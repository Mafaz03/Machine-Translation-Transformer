import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import torch.nn.functional as F

from dataset import Multi30kDataset
from model import Transformer, make_src_mask, make_tgt_mask

from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

from dataset import *
from lr_scheduler import *

import wandb
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int = 1, smoothing: float = 0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        # logits: (B*T, V)
        # target: (B*T)

        log_probs = F.log_softmax(logits, dim=-1)   # (B*T, V)

        with torch.no_grad():
            # Create smoothed distribution
            true_dist = torch.zeros_like(log_probs)

            # Fill with smoothing value
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))

            # Put confidence at correct indices
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

            # Zero out pad positions
            true_dist[target == self.pad_idx] = 0

        # Compute loss
        loss = -(true_dist * log_probs).sum(dim=1)   # (B*T)

        # Ignore pad tokens in loss
        non_pad_mask = target != self.pad_idx
        loss = loss[non_pad_mask].mean()

        return loss
    

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:

    model.train() if is_train else model.eval()
    losses = []
    for _ in range(epoch_num):
        total_loss = 0

        for src, tgt in tqdm(data_iter):

            src = src.to(device)
            tgt = tgt.to(device)

            # masks
            src_mask = make_src_mask(src)
            tgt_mask = make_tgt_mask(tgt)

            # shift for teacher forcing
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_mask = make_tgt_mask(tgt_input)

            # forward
            logits = model(src, tgt_input, src_mask, tgt_mask)

            # reshape
            logits = logits.contiguous().view(-1, logits.shape[-1])
            tgt_output = tgt_output.contiguous().view(-1)

            # loss
            loss = loss_fn(logits, tgt_output)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                encoder_weights_WV = 0
                encoder_weights_WK = 0
                encoder_weights_WQ = 0

                decoder_weights_WV = 0
                decoder_weights_WK = 0
                decoder_weights_WQ = 0

                ## Gradient tracking ##
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    if "WQ" in name or "WK" in name or "WV" in name:
                        param_grad = param.grad.detach().norm(2).item()     
                        # encoder
                        if "encoder" in name and ".weight" in name:
                            if "WQ" in name:
                                encoder_weights_WQ += param_grad
                            if "WK" in name:
                                encoder_weights_WK += param_grad
                            if "WV" in name:
                                encoder_weights_WV += param_grad
                            
                        if "decoder" in name and ".weight" in name:
                            if "WQ" in name:
                                decoder_weights_WQ += param_grad
                            if "WK" in name:
                                decoder_weights_WK += param_grad
                            if "WV" in name:
                                decoder_weights_WV += param_grad

                grad_norm = param.grad.detach().norm(2).item()
                print(f"encoder_weights_WV --> {encoder_weights_WV}")
                print(f"encoder_weights_WK --> {encoder_weights_WK}")
                print(f"encoder_weights_WQ --> {encoder_weights_WQ}\n")

                print(f"decoder_weights_WV --> {decoder_weights_WV}")
                print(f"decoder_weights_WK --> {decoder_weights_WK}")
                print(f"decoder_weights_WQ --> {decoder_weights_WQ}\n")

                wandb.log({
                    f"grad_norm/encoder_weights_WV": encoder_weights_WV,
                    f"grad_norm/encoder_weights_WK": encoder_weights_WK,
                    f"grad_norm/encoder_weights_WQ": encoder_weights_WQ,
                    f"grad_norm/decoder_weights_WV": decoder_weights_WV,
                    f"grad_norm/decoder_weights_WK": decoder_weights_WK,
                    f"grad_norm/decoder_weights_WQ": decoder_weights_WQ,
                })
                #######################

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
        losses.append(total_loss / len(data_iter))

    return sum(losses)/len(losses)


def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device = "cpu", break_at_eos = True):
    src = src.to(device)
    src_mask = src_mask.to(device)
    
    # encode
    ys = torch.ones(src.size(0), 1).fill_(start_symbol).long().to(device)
    memory = model.encode(src, src_mask)

    # loop
    for _ in range(max_len):
        # decoding
        tgt_mask = make_tgt_mask(ys).to(device)
        out = model.decode(memory, src_mask, ys, tgt_mask)

        prob = out[:, -1, :] # (B, vocab)
        next_word = prob.argmax(dim=-1)

        ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

        if (next_word == end_symbol).all() and break_at_eos:
            break
    return ys



def evaluate_bleu(
    model: Transformer,
    test_dataloader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:

    model.eval()

    references = []
    hypotheses = []

    token_to_idx = {v: i for i, v in tgt_vocab.items()}
    sos_idx = token_to_idx["<sos>"]
    eos_idx = token_to_idx["<eos>"]
    pad_idx = token_to_idx["<pad>"]

    itos = tgt_vocab  # idx -> token

    with torch.no_grad():
        for num, (src, tgt) in enumerate(test_dataloader):
            # if num == 5: break
            B = src.size(0)

            ys = greedy_decode(model = model, src = src, 
                          src_mask = make_src_mask(src).to(device),
                          max_len = max_len, start_symbol = sos_idx, end_symbol = eos_idx,
                          device = device, break_at_eos = False 
                          )

            # Convert predictions and targets to words
            for i in tqdm(range(B)):

                pred_tokens = ys[i].tolist()
                tgt_tokens  = tgt[i].tolist()

                # Remove special tokens
                pred_sentence = []
                for tok in pred_tokens:
                    if tok == eos_idx:
                        break
                    if tok not in [sos_idx, pad_idx]:
                        pred_sentence.append(itos[tok])

                ref_sentence = []
                for tok in tgt_tokens:
                    if tok == eos_idx:
                        break
                    if tok not in [sos_idx, pad_idx]:
                        ref_sentence.append(itos[tok])

                hypotheses.append(pred_sentence)
                references.append([ref_sentence])  # list of references

    bleu = corpus_bleu(references, hypotheses) * 100
    return bleu


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    torch.save(
        {
            "epoch"               : epoch,
            "model_state_dict"    : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": {
                "src_vocab_size": model.src_vocab_size,
                "tgt_vocab_size": model.tgt_vocab_size,
                "d_model"       : model.d_model,
                "N"             : model.N,
                "num_heads"     : model.num_heads,
                "d_ff"          : model.d_ff,
                "dropout"       : model.dropout,
            }
         }
    ,path
    )

def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device = "cpu"
) -> int:

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"]


def run_training_experiment() -> None:
    # 2. Build dataset / vocabs from dataset.py
    language_dataset = Multi30kDataset(split = "train")

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    config = {
        "src_vocab_size"   : len(language_dataset.de_vocab),
        "tgt_vocab_size"   : len(language_dataset.en_vocab),
        "d_model"          : 512,
        "N"                : 6,
        "num_heads"        : 8,
        "d_ff"             : 2048,
        "dropout"          : 0.1,
        "train_batch_size" : 32,
        "test_batch_size"  : 32,
        "epochs"           : 1,
        "device"           : device,
        'save_every'       : 4
    }


    # 1. Init W&B
    wandb.init(project="Machine Translation Transformer", config = config)

    # 3. Create DataLoaders for train / val 
    processed_dataset = language_dataset.process_data()
    train_dataset_obj = TranslationDataset(processed_dataset)
    train_dataloader = DataLoader(
        train_dataset_obj,
        batch_size=config["train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    language_dataset.split = "test"
    processed_dataset = language_dataset.process_data()
    test_dataset_obj = TranslationDataset(processed_dataset)
    test_dataloader = DataLoader(
        test_dataset_obj,
        batch_size=config["test_batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    # 4. Instantiate Transformer with hyperparameters from config
    transformer = Transformer(src_vocab_size = config["src_vocab_size"], tgt_vocab_size = config['tgt_vocab_size'],
                            d_model = 512, N = 6, num_heads = 8, d_ff = 2048,
                            dropout = 0.1).to(config["device"])

    # 5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
    optimizer = optim.Adam(transformer.parameters(), betas = [0.9, 0.98], lr=1)

    # 6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
    scheduler = NoamScheduler(optimizer, d_model = config["d_model"], warmup_steps = 5000)

    # 7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
    loss_fn = LabelSmoothingLoss(vocab_size = config["tgt_vocab_size"], pad_idx = 1, smoothing = 0.1)

    # 8. Training loop:
    for epoch in range(config['epochs']):
        transformer.train()
        train_loss = run_epoch(train_dataloader, transformer, loss_fn,
                        optimizer, scheduler, 1, is_train=True, device=config['device'])
        transformer.eval()
        test_loss = run_epoch(test_dataloader, transformer, loss_fn,
                        None, None, 1, is_train=False, device=config['device'])
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'test_loss': test_loss})
        print(f"EPOCH: {epoch} => Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}")
        
        if (epoch % config['save_every'] == 0) or (epoch == config["epochs"]-1):
            print(f"Saving at epoch: {epoch}")
            save_checkpoint(transformer, optimizer, scheduler, epoch)

    # 9. Final BLEU on test set
    bleu = evaluate_bleu(transformer, test_dataloader, language_dataset.en_itos, device = config['device'])
    print(f"Blue score: {bleu}")
    wandb.log({'test_bleu': bleu})
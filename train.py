import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import torch.nn.functional as F

from dataset import Multi30kDataset
from model import Transformer, make_src_mask, make_tgt_mask

from tqdm import tqdm

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
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
        losses.append(total_loss / len(data_iter))

    return sum(losses)/len(losses)
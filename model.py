import math
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Multi30kDataset





def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # Q : Query tensor,  shape (B, Heads, seq_q, d_k)
    # K : Key tensor,    shape (B, Heads, seq_k, d_k)
    # V : Value tensor,  shape (B, Heads, seq_k, d_v)

    # seq_x -> List of tokens
    # d_x   -> Features each token can use to represent itself

    # d_k = d_q
    # seq_k = seq_v

    d_k = K.shape[-1]
    KT  = K.transpose(-2, -1)
    scores = torch.matmul(Q, KT) / math.sqrt(d_k)          # (B, H, seq_q, seq_k)
    # scores = torch.matmul(Q, KT)# / math.sqrt(d_k)          # (B, H, seq_q, seq_k)
    
    if mask is not None: 
        scores = scores.masked_fill(mask, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)            # (B, H, seq_q, seq_k)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
        
    output = torch.matmul(attn_weights, V)                  # (B, H, seq_q, d_v)

    return output, attn_weights
    
    

def make_src_mask(src: torch.Tensor, pad_idx: int = 1):
    mask = (src == pad_idx)
    return mask.unsqueeze(1).unsqueeze(2)

def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1):
    B, tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, tgt_len)

    causal_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=tgt.device),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)                            # (1, 1, tgt_len, tgt_len)
    mask = pad_mask | causal_mask
    return mask


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        
        
        self.WQ = torch.nn.Linear(d_model, d_model)
        self.WK = torch.nn.Linear(d_model, d_model)
        self.WV = torch.nn.Linear(d_model, d_model)
        self.WO = torch.nn.Linear(d_model, d_model)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_v, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]
        """

        B, seq_q, _ = query.shape
        B, seq_k, _ = key.shape
        B, seq_v, _ = value.shape # seq_k == seq_v anyways 
                                  # dim_v == dim_model

        # Linear Projection
        Q = self.WQ(query) # [batch, seq_q, d_model] x [d_model, d_model] -> [batch, seq_q, d_model]
        K = self.WK(key)   # [batch, seq_q, d_model] x [d_model, d_model] -> [batch, seq_k, d_model]
        V = self.WV(value) # [batch, seq_q, d_model] x [d_model, d_model] -> [batch, seq_v, d_model]
        
        # Split into heads
        Q = Q.view(B, seq_q, self.num_heads, self.d_k).transpose(1, 2)  # [batch, seq_q, num_heads, d_k]
                                                        # After transpose [batch, num_heads, seq_q, d_k]

        K = K.view(B, seq_k, self.num_heads, self.d_k).transpose(1, 2)  # [batch, seq_k, num_heads, d_k]
                                                        # After transpose [batch, num_heads, seq_q, d_k]

        V = V.view(B, seq_k, self.num_heads, self.d_k).transpose(1, 2)  # [batch, seq_v, num_heads, d_k]
                                                        # After transpose [batch, num_heads, seq_v, d_k]

        output, attn_weights = scaled_dot_product_attention(Q, K, V, mask, self.dropout)     # output: (B, num_heads, seq_q, d_k), attn: (B, H, seq_q, seq_k)

        # Merging heads
        output = output.transpose(1, 2).contiguous()                # output: (B, seq_q, num_heads, d_k)
        output = output.view(B, seq_q, self.d_model)                # output: (B, seq_q, d_model)

        # Final linear
        output = self.WO(output)
        self.attn_weights = attn_weights
        return output
    
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        B, seq_len, _ = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, seq_len)

        pos_embed = self.embedding(positions)

        return self.dropout(x + pos_embed)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)                                  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # positions: (max_len, 1)
        div_term = 1 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))

        # apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # reshape to (1, max_len, d_model) for broadcasting
        pe = pe.unsqueeze(0)

        # register as buffer (not a parameter)
        self.register_buffer("pe", pe)

        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.mha_self_attn = MultiHeadAttention(d_model = d_model, num_heads = num_heads, dropout = dropout)

        self.d_model = d_model
        
        self.dropout     = torch.nn.Dropout(dropout)
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.linear1     = torch.nn.Linear(d_model, d_ff)
        self.relu        = torch.nn.ReLU()
        self.linear2     = torch.nn.Linear(d_ff, d_model)

        
    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:

        # Self attention
        attn_out = self.mha_self_attn(x, x, x, src_mask)    # [B, src_len, d_model]

        # Add and Norm
        x = x + self.dropout(attn_out)
        x = self.layer_norm1(x)

        # Feed forward
        ffn_out = self.linear2(self.relu(self.linear1(x)))  # [B, src_len, d_model]

        # Add and Norm
        x = x + self.dropout(ffn_out)
        x = self.layer_norm2(x)

        return x                                            # [B, src_len, d_model]
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(d_model = d_model, num_heads = num_heads, dropout = dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.pos_encoding  = PositionalEncoding(d_model = d_model, dropout = dropout, max_len = 5000)
        # self.pos_encoding  = LearnedPositionalEncoding(d_model = d_model, dropout = dropout, max_len = 5000)

        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.layer_norm3 = torch.nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)


    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:

        # Masked self attention
        attn_out = self.self_attn(x, x, x, tgt_mask)               # [B, tgt_len, d_model]
        x = self.layer_norm1(x + self.dropout(attn_out))           # [B, tgt_len, d_model]

        # Cross attension
        attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.layer_norm2(x + self.dropout(attn_out))           # [B, tgt_len, d_model]

        # feed forward
        ffn_out = self.ffn(x)                                      # [B, tgt_len, d_model]
        x = self.layer_norm3(x + self.dropout(ffn_out))            # [B, tgt_len, d_model]

        return x

class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.encoder_layers = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.layer_norm = torch.nn.LayerNorm(layer.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        for layer in self.encoder_layers:
            x = layer(x, mask)
        x = self.layer_norm(x)

        return x
    

class Decoder(nn.Module):

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.decoder_layers = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.layer_norm = torch.nn.LayerNorm(layer.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x = self.layer_norm(x)

        return x
    

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        if not (src_vocab_size or tgt_vocab_size):
            language_dataset = Multi30kDataset(split = "train")
            src_vocab_size = len(language_dataset.de_vocab)
            tgt_vocab_size = len(language_dataset.en_vocab)
        self.language_dataset = language_dataset

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        self.positional_encodings = PositionalEncoding(d_model = d_model, dropout = dropout, max_len = 5000)
        # self.positional_encodings = LearnedPositionalEncoding(d_model = d_model, dropout = dropout, max_len = src_vocab_size)

        encoder_layer = EncoderLayer(d_model = d_model, num_heads = num_heads, d_ff = d_ff, dropout = dropout)
        self.encoder  = Encoder(layer = encoder_layer, N = N)

        decoder_layer = DecoderLayer(d_model = d_model, num_heads = num_heads, d_ff = d_ff, dropout = dropout)
        self.decoder  = Decoder(layer = decoder_layer, N = N)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encodings(src)
        self.memory = self.encoder(src, src_mask)   # [B, src_len, d_model]
        return self.memory      


    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encodings(tgt)

        out = self.decoder(tgt, memory, src_mask, tgt_mask) # [B, tgt_len, d_model]
        return self.fc_out(out)               # [B, tgt_len, d_model] -> [B, tgt_len, tgt_vocab_size]

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        logits = self.decode(memory, src_mask, tgt, tgt_mask)
        return logits

    # def infer(self, src_sentence: str) -> str:
    #     # build vocab
    #     language_dataset = Multi30kDataset(split = "train")
    #     src_vocab = len(language_dataset.de_vocab)
    #     tgt_vocab = len(language_dataset.en_vocab)

    #     # text to tokens
    #     # src = torch.tensor([language_dataset.de_vocab[i] for i in src_sentence.split()]).unsqueeze(0)
    #     tokens = language_dataset.tokenize_de(src_sentence)
    #     src = torch.tensor([language_dataset.de_vocab.get(tok, language_dataset.de_vocab["<unk>"])for tok in tokens]).unsqueeze(0)

    #     src_mask = make_src_mask(src).to(src.device)

    #     # text to tokens
    #     transformer = Transformer(src_vocab_size = src_vocab, tgt_vocab_size = tgt_vocab,
    #                       d_model = 512, N = 6, num_heads = 8, d_ff = 2048,
    #                       dropout = 0.1)

    #     load_checkpoint("checkpoint.pt", transformer)

    #     # decode
    #     sos_idx = 2
    #     eos_idx = 3
    #     prediction_idx = greedy_decode(transformer, src, src_mask, 40, sos_idx, eos_idx, src.device, break_at_eos = True)
    #     no_clean = ' '.join([language_dataset.en_itos[i.item()] for i in prediction_idx[0]])
    #     clean = no_clean.split("<sos>")[-1].strip().split("<eos>")[0].strip()

    #     return clean

    def infer(self, src_sentence: str) -> str:

        language_dataset = self.language_dataset

        tokens = language_dataset.tokenize_de(src_sentence)

        src = torch.tensor([
            language_dataset.de_vocab.get(tok, language_dataset.de_vocab["<unk>"])
            for tok in tokens
        ]).unsqueeze(0)

        src_mask = make_src_mask(src).to(src.device)

        sos_idx = language_dataset.en_vocab["<sos>"]
        eos_idx = language_dataset.en_vocab["<eos>"]

        with torch.no_grad():
            prediction_idx = greedy_decode(
                self,
                src,
                src_mask,
                max_len=40,
                start_symbol=sos_idx,
                end_symbol=eos_idx,
                device=src.device,
                break_at_eos=True
            )

        tokens = []

        for idx in prediction_idx[0]:
            word = language_dataset.en_itos[idx.item()]

            if word == "<sos>":
                continue

            if word == "<eos>":
                break

            tokens.append(word)

        print(f"src_sentence:\n{src_sentence}")
        print(f"result:\n{' '.join(tokens)}")
        return " ".join(tokens)
        return clean
    


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
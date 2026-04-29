from datasets import load_dataset
from collections import Counter

import spacy
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader, Dataset

import torch
class Multi30kDataset:
    def __init__(self, split: str = "train"):
        self.dataset = load_dataset("bentrevett/multi30k")
        self.split = split

        self.spacy_de = spacy.load("de_core_news_sm", disable=["parser", "ner", "tagger"]) # German
        self.spacy_en = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])  # english

        self.specials = ["<unk>", "<pad>", "<sos>", "<eos>"] 
        self.build_vocab()

    # def tokenize_de(self, text): # fancy way of lowering text but it have more info ike pos etc
    #     return [tok.text.lower() for tok in self.spacy_de(text)]

    def tokenize_de(self, text):
        return text.lower().split()
    def tokenize_en(self, text):
        return text.lower().split()
   
    # def tokenize_en(self, text):
    #     return [tok.text.lower() for tok in self.spacy_en(text)]  
    
    
    def build_vocab(self):
        print("Building vocab, please wait...")
        de_counter = Counter() # word count
        en_counter = Counter()

        for split_name in ["train", "validation", "test"]:
            split_data = self.dataset[split_name]
            for word in tqdm(split_data, total=split_data.num_rows, desc=f"Processing {split_name}"):
                de_tokens = self.tokenize_de(word["de"])
                en_tokens = self.tokenize_en(word["en"])

                de_counter.update(de_tokens)
                en_counter.update(en_tokens)
        
        # bluiding vocab mappping from the counters

        # word: idx mappings
        self.de_vocab = {}
        self.en_vocab = {}
        idx = 0
        for _ in self.specials:
            self.de_vocab[self.specials[idx]] = idx
            self.en_vocab[self.specials[idx]] = idx
            idx += 1

        de_idx = idx
        en_idx = idx

        for entry, freq in de_counter.items():
            self.de_vocab[entry] = de_idx
            de_idx += 1
        for entry, freq in en_counter.items():
            self.en_vocab[entry] = en_idx
            en_idx += 1
        
        # idx: word mappings
        self.de_itos = {idx: entry for entry, idx in self.de_vocab.items()}
        self.en_itos = {idx: entry for entry, idx in self.en_vocab.items()}
    
    def get_idx_from_tokens(self, tokens, vocab):
        return [vocab.get(tok, vocab["<unk>"]) for tok in tokens]

    def process_data(self):
        processed_data = []

        split_data = self.dataset[self.split]
        for word in tqdm(split_data, total=split_data.num_rows):
            de_tokens = self.tokenize_de(word["de"])
            en_tokens = self.tokenize_en(word["en"])

            de_idx = [self.de_vocab["<sos>"]] + self.get_idx_from_tokens(de_tokens, self.de_vocab) + [self.de_vocab["<eos>"]]
            en_idx = [self.en_vocab["<sos>"]] + self.get_idx_from_tokens(en_tokens, self.en_vocab) + [self.en_vocab["<eos>"]]
        
            processed_data.append((de_idx, en_idx))
        
        return processed_data
    
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(src), torch.tensor(tgt)
    

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=1, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=1, batch_first=True)
    return src_batch, tgt_batch

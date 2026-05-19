# Machine Translation Transformer

Github:https://github.com/Mafaz03/Machine-Translation-Transformer
WandB report: https://wandb.ai/mafaz03/Machine%20Translation%20Transformer/reports/Machine-Translation-Transformer--VmlldzoxNjkyMTU4Mg?accessToken=ujyxun87k4we9sz1e6hddckc2fohwmbodoeoyig56uzrfmhuruj9j2uqqisnkkbe

A PyTorch implementation of a seq2seq Transformer model for German-to-English translation using the Multi30k dataset.

## Overview

This project builds a Transformer from scratch (without `torch.nn.MultiheadAttention`) and trains it on the Multi30k dataset. It includes:

- `dataset.py` — dataset loading, tokenization, vocabulary building, and batching
- `model.py` — Transformer encoder-decoder architecture, attention, positional encoding, and inference utilities
- `train.py` — training loop, label smoothing loss, checkpoint saving/loading, BLEU evaluation, and WandB logging
- `lr_scheduler.py` — Noam learning rate scheduler used for Transformer training

## Project Structure

- `dataset.py`
  - `Multi30kDataset`
  - `TranslationDataset`
  - `collate_fn`
- `model.py`
  - `Transformer`
  - `Encoder`, `Decoder`, `EncoderLayer`, `DecoderLayer`
  - `MultiHeadAttention`, `PositionalEncoding`, `LearnedPositionalEncoding`
  - `greedy_decode`, `load_checkpoint`
- `train.py`
  - `LabelSmoothingLoss`
  - `run_epoch`, `evaluate_bleu`
  - `save_checkpoint`, `load_checkpoint`
  - `run_training_experiment`
- `lr_scheduler.py`
  - `NoamScheduler`

## Requirements

Install the required Python packages:

```bash
pip install torch torchvision torchaudio datasets spacy nltk tqdm wandb
```

Install the spaCy language models used for tokenization:

```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Training

The training experiment is defined in `train.py` as `run_training_experiment()`.

Run training with:

```bash
python -c "from train import run_training_experiment; run_training_experiment()"
```

The script will:

- build the Multi30k vocabulary
- load German-English sentence pairs
- create dataloaders with padding
- initialize the Transformer model
- train with label smoothing and a Noam scheduler
- save checkpoints to `checkpoint.pt`
- evaluate final BLEU score on the test set

## Inference

The `Transformer` class in `model.py` provides an `infer()` helper that:

- loads the trained checkpoint from `checkpoint.pt`
- tokenizes a German sentence
- performs greedy decoding
- returns the English translation string

## Notes

- The dataset uses `bentrevett/multi30k` from the Hugging Face `datasets` library.
- Padding token index is `1` and special tokens include: `<unk>`, `<pad>`, `<sos>`, `<eos>`.
- WandB logging is enabled in `train.py` and will create a run under the project name `Machine Translation Transformer`.

## Optional Improvements

- add a script entrypoint or CLI wrapper for easier training/inference
- support beam search decoding for better translation quality
- add model checkpoint loading from a custom path
- add a `requirements.txt` for exact dependency versions

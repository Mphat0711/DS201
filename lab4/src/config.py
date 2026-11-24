from dataclasses import dataclass
import random
import numpy as np
import torch

PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIAL = ['<pad>', '<unk>', '<bos>', '<eos>']

@dataclass
class ExperimentConfig:
    emb: int = 256
    hidden: int = 256
    layers: int = 3
    dropout: float = 0.3
    batch: int = 32
    lr: float = 1e-3
    teacher_ratio: float = 0.5
    epochs: int = 10
    max_words: int = 50
    vocab_limit: int = 10000
    min_freq: int = 2
    train_limit: int | None = None
    dev_limit: int | None = None
    seed: int = 42

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

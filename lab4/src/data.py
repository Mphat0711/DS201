import json
import unicodedata
from collections import Counter
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from .config import PAD, UNK, BOS, EOS, SPECIAL

def tokens(text):
    return unicodedata.normalize('NFC', text.replace('\ufeff', '')).lower().split()

def load_splits(data_dir, max_words):
    result = {}
    for split in ('train', 'dev', 'test'):
        path = Path(data_dir) / f'small-{split}.json'
        rows = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(rows, list):
            raise ValueError(f'{path} must contain a JSON array')
        result[split] = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in ('english', 'vietnamese')):
                raise ValueError(f'Invalid row {i} in {path}')
            src, tgt = tokens(row['english']), tokens(row['vietnamese'])
            if 1 <= len(src) <= max_words and 1 <= len(tgt) <= max_words:
                result[split].append(dict(id=i, src=src, tgt=tgt))
        if not result[split]:
            raise ValueError(f'No usable rows in {path}')
    return result

def make_vocab(sequences, min_freq, vocab_limit):
    freq = Counter(t for seq in sequences for t in seq)
    ranked = sorted(freq, key=lambda t: (-freq[t], t))
    keep = [t for t in ranked if freq[t] >= min_freq and t not in SPECIAL][:vocab_limit-4]
    return {token: i for i, token in enumerate(SPECIAL + keep)}

def encode(seq, vocab):
    return [vocab.get(t, UNK) for t in seq]

def collate(rows, src_vocab, tgt_vocab):
    src = [torch.tensor(encode(r['src'], src_vocab)+[EOS]) for r in rows]
    tgt = [torch.tensor([BOS]+encode(r['tgt'], tgt_vocab)+[EOS]) for r in rows]
    return dict(src=pad_sequence(src, batch_first=True, padding_value=PAD),
                lengths=torch.tensor([len(x) for x in src]),
                tgt=pad_sequence(tgt, batch_first=True, padding_value=PAD),
                refs=[' '.join(r['tgt']) for r in rows], ids=[r['id'] for r in rows],
                sources=[' '.join(r['src']) for r in rows])

def choose(rows, limit, seed=42):
    if limit is None or len(rows) <= limit: return rows
    return [rows[int(i)] for i in np.random.default_rng(seed).permutation(len(rows))[:limit]]

def loader(rows, config, src_vocab, tgt_vocab, shuffle=False):
    return DataLoader(rows, batch_size=config.batch, shuffle=shuffle,
                      collate_fn=partial(collate, src_vocab=src_vocab, tgt_vocab=tgt_vocab),
                      generator=torch.Generator().manual_seed(config.seed))

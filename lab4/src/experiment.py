from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import seed_all
from .data import load_splits, make_vocab, choose, loader
from .model import make_seq2seq
from .train import fit, evaluate
from .artifacts import create_run_dirs, save_json, plot_learning_curves, save_examples, plot_attention_heatmaps

@dataclass
class ExperimentResult:
    name: str
    best_epoch: int
    dev_metrics: dict
    test_metrics: dict | None
    parameter_count: int
    seconds_per_epoch: float
    output_dir: str

def run_experiment(model_name, data_dir, output_dir, config, evaluate_test=False):
    if config.epochs < 1: raise ValueError('epochs must be positive')
    root = create_run_dirs(output_dir)
    log = logging.getLogger(f'lab4.{model_name}.{root.resolve()}')
    log.setLevel(logging.INFO)
    log.propagate = False
    for handler in log.handlers[:]:
        handler.close()
        log.removeHandler(handler)
    for handler in (logging.FileHandler(root/'logs/train.log', encoding='utf-8'), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        log.addHandler(handler)
    seed_all(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits = load_splits(data_dir, config.max_words)
    src_vocab = make_vocab((r['src'] for r in splits['train']), config.min_freq, config.vocab_limit)
    tgt_vocab = make_vocab((r['tgt'] for r in splits['train']), config.min_freq, config.vocab_limit)
    tgt_itos = {i:t for t,i in tgt_vocab.items()}
    train_rows = choose(splits['train'], config.train_limit, config.seed)
    dev_rows = choose(splits['dev'], config.dev_limit, config.seed)
    def make_loader(rows, shuffle=False): return loader(rows, config, src_vocab, tgt_vocab, shuffle)
    log.info('model=%s device=%s config=%s split_counts=%s', model_name, device, asdict(config), {k:len(v) for k,v in splits.items()})
    model = make_seq2seq(model_name, len(src_vocab), len(tgt_vocab), config).to(device)
    history, best_epoch = fit(model, make_loader(train_rows, True), make_loader(dev_rows), tgt_itos, config, device, root/'checkpoints/best.pt', log)
    dev, examples = evaluate(model, make_loader(dev_rows), tgt_itos, config, device, collect=True)
    test = evaluate(model, make_loader(splits['test']), tgt_itos, config, device)[0] if evaluate_test else None
    pd.DataFrame(history).to_csv(root/'metrics/history.csv', index=False)
    pd.DataFrame([dev]).to_csv(root/'metrics/dev_metrics.csv', index=False)
    if test is not None: pd.DataFrame([test]).to_csv(root/'metrics/test_metrics.csv', index=False)
    save_examples(examples, root/'metrics/sample_translations.csv')
    plot_learning_curves(history, root/'plots/learning_curves.png')
    if model_name == 'bahdanau': plot_attention_heatmaps(examples, root/'plots/attention_heatmaps', tgt_itos)
    save_json(asdict(config), root/'config.json')
    save_json(dict(config=asdict(config), model=model_name, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
                   normalization='remove BOM; NFC; lowercase; whitespace', best_epoch=best_epoch,
                   selected_train_ids=[r['id'] for r in train_rows], selected_dev_ids=[r['id'] for r in dev_rows]), root/'checkpoints/metadata.json')
    log.info('best_epoch=%s dev=%s test=%s', best_epoch, dev, test)
    return ExperimentResult(model_name, best_epoch, dev, test, sum(p.numel() for p in model.parameters()), float(np.mean([x['seconds'] for x in history])), str(root))

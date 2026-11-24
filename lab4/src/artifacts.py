import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from .config import EOS
from .metrics import rouge

def create_run_dirs(output_dir):
    root = Path(output_dir)
    for name in ('logs', 'metrics', 'plots', 'checkpoints'):
        (root/name).mkdir(parents=True, exist_ok=True)
    return root

def save_json(data, path):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def plot_learning_curves(history, path):
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in zip(axes, ('train_loss', 'dev_rougeL_f1'), ('Train loss', 'Dev ROUGE-L F1')):
        ax.plot(frame.epoch, frame[col], marker='o')
        ax.set(xlabel='Epoch', title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def save_examples(examples, path):
    rows = [dict(source=e['source'], reference=e['reference'], hypothesis=e['hypothesis'],
                 rougeL_f1=rouge.score(e['reference'], e['hypothesis'])['rougeL'].fmeasure) for e in examples]
    pd.DataFrame(rows).to_csv(path, index=False)

def plot_attention_heatmaps(examples, directory, tgt_itos):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for i, ex in enumerate(examples, 1):
        ids = ex['ids']
        stop = ids.index(EOS)+1 if EOS in ids else len(ids)
        matrix = ex['weights'][:stop]
        fig, ax = plt.subplots(figsize=(max(6, len(ex['source'].split())*.4), max(4, stop*.35)))
        im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(matrix.shape[1]), ex['source'].split()+['<eos>'], rotation=90)
        ax.set_yticks(range(stop), [tgt_itos[x] for x in ids[:stop]])
        ax.set(xlabel='Source token', ylabel='Generated token')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(directory/f'attention_{i}.png', dpi=150)
        plt.close(fig)

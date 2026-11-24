import time
import torch
from torch import nn
from .config import PAD, seed_all
from .decode import generate, decode
from .metrics import translation_metrics

def evaluate(model, data_loader, tgt_itos, config, device, collect=False):
    model.eval()
    hypotheses, references, examples = [], [], []
    nll, count = 0., 0
    with torch.no_grad():
        for b in data_loader:
            src, tgt = b['src'].to(device), b['tgt'].to(device)
            target = tgt[:, 1:]
            logits = model(src, b['lengths'], tgt, teacher_ratio=1.0)
            nll += nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=PAD, reduction='sum').item()
            count += int((target != PAD).sum())
            ids, weights = generate(model, src, b['lengths'], config.max_words+1)
            hyps = [decode(row, tgt_itos) for row in ids.cpu()]
            hypotheses.extend(hyps)
            references.extend(b['refs'])
            if collect and len(examples) < 3:
                for i in range(min(len(hyps), 3-len(examples))):
                    examples.append(dict(source=b['sources'][i], reference=b['refs'][i], hypothesis=hyps[i],
                        ids=ids[i].cpu().tolist(), weights=weights[i, :, :int(b['lengths'][i])].cpu().numpy() if weights is not None else None))
    metrics = translation_metrics(hypotheses, references)
    metrics['dev_nll'] = nll/count
    return metrics, examples

def train_epoch(model, train_loader, optimizer, config, device):
    model.train()
    total, count = 0., 0
    for b in train_loader:
        src, tgt = b['src'].to(device), b['tgt'].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(src, b['lengths'], tgt, config.teacher_ratio)
        gold = tgt[:, 1:]
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), gold.reshape(-1), ignore_index=PAD)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n = int((gold != PAD).sum())
        total += loss.item()*n
        count += n
    return total/count

def fit(model, train_loader, dev_loader, tgt_itos, config, device, checkpoint, log):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_key, best_epoch, history = (-1., float('-inf')), 0, []
    for epoch in range(1, config.epochs+1):
        seed_all(config.seed+epoch)
        start = time.perf_counter()
        loss = train_epoch(model, train_loader, optimizer, config, device)
        if device.type == 'cuda': torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        metrics, _ = evaluate(model, dev_loader, tgt_itos, config, device)
        key = (metrics['rougeL_f1'], -metrics['dev_nll'])
        if key > best_key:
            best_key, best_epoch = key, epoch
            torch.save(model.state_dict(), checkpoint)
        row = dict(epoch=epoch, train_loss=loss, seconds=seconds, **{f'dev_{k}' if k != 'dev_nll' else k:v for k,v in metrics.items()})
        history.append(row)
        log.info('epoch=%s loss=%.4f dev_ROUGE-L=%.4f seconds=%.2f', epoch, loss, metrics['rougeL_f1'], seconds)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return history, best_epoch

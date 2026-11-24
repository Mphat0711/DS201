import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from .data import tokens

class UnicodeTokens:
    def tokenize(self, text): return tokens(text)

rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, tokenizer=UnicodeTokens())

def translation_metrics(hypotheses, references):
    scores = [rouge.score(ref, hyp)['rougeL'] for hyp, ref in zip(hypotheses, references)]
    bleu = BLEU(tokenize='none', lowercase=False)
    value = bleu.corpus_score(hypotheses, [references])
    return dict(rougeL_f1=float(np.mean([s.fmeasure for s in scores])),
                rougeL_precision=float(np.mean([s.precision for s in scores])),
                rougeL_recall=float(np.mean([s.recall for s in scores])),
                bleu=value.score, bleu_signature=str(bleu.get_signature()))

"""Set-based precision/recall/F1 with document-level bootstrap intervals."""
import random

from . import config


def counts(pred, gold):
    return len(pred & gold), len(pred - gold), len(gold - pred)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def micro(per_doc):
    """per_doc: list of (tp, fp, fn)."""
    tp, fp, fn = (sum(x[i] for x in per_doc) for i in range(3))
    return prf(tp, fp, fn)


def bootstrap_ci(per_doc, reps=1000, alpha=0.05):
    """95% intervals for (P, R, F1), resampling documents with replacement."""
    rng = random.Random(config.SEED)
    samples = []
    for _ in range(reps):
        draw = [per_doc[rng.randrange(len(per_doc))] for _ in per_doc]
        samples.append(micro(draw))
    out = []
    for i in range(3):
        vals = sorted(s[i] for s in samples)
        out.append((vals[int(reps * alpha / 2)], vals[int(reps * (1 - alpha / 2)) - 1]))
    return out


def paired_diff(per_doc_a, per_doc_b, reps=1000, alpha=0.05):
    """Paired bootstrap of (P, R, F1) of b minus a. Both lists are per document, in the
    same order, and each draw resamples the same documents for both. Returns, per metric,
    (difference, low, high, share of draws where b is not better)."""
    rng = random.Random(config.SEED)
    base = [y - x for x, y in zip(micro(per_doc_a), micro(per_doc_b))]
    samples = []
    for _ in range(reps):
        idx = [rng.randrange(len(per_doc_a)) for _ in per_doc_a]
        a, b = micro([per_doc_a[i] for i in idx]), micro([per_doc_b[i] for i in idx])
        samples.append([y - x for x, y in zip(a, b)])
    out = []
    for i in range(3):
        vals = sorted(s[i] for s in samples)
        out.append((base[i], vals[int(reps * alpha / 2)], vals[int(reps * (1 - alpha / 2)) - 1],
                    sum(v <= 0 for v in vals) / reps))
    return out

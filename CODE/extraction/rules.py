"""Ontology-directed extraction rules learned from gold labels.

Two rule shapes are learned from the training folds only:
  context  ("experience", "with") -> take the list that follows the trigger
  segment  ("highlights",)        -> a short piece in this section is a skill
Each candidate rule is applied back to the training documents and kept only
if it is precise enough. A rule's precision becomes its confidence. Every
extraction carries a span, so rules are provenance-checked by construction.
A negation rule removes skills preceded by "no", "not", "without", etc.
"""
import re
from collections import defaultdict

from .extractors import canonical_id, row
from .provenance import find_quote

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#.'\-]*[A-Za-z0-9+#]|[A-Za-z0-9]")
_SEP = re.compile(r"\s*(?:[,;/&()|]|\band\b|\bor\b|\bas well as\b|\bincluding\b)\s*", re.I)
_NEG = re.compile(r"\b(no|not|without|lack|lacking|never|none)\b|n't\b", re.I)
_DET = re.compile(r"^(a|an|the|of|in|with|using)\s+", re.I)
MAX_ITEM_WORDS = 4
MAX_ITEMS = 10
SEGMENT_MAX_WORDS = 4
MIN_TP = 2
MIN_PRECISION = 0.6


def _items_after(text, offset):
    """Split the text after a trigger into list items with spans."""
    items, pos = [], offset
    for m in list(_SEP.finditer(text, offset)) + [None]:
        end = m.start() if m else len(text)
        piece = text[pos:end]
        lead = len(piece) - len(piece.lstrip())
        piece = piece.strip().rstrip(".:")
        det = _DET.match(piece)
        if det:
            lead += det.end()
            piece = piece[det.end():]
        if piece:
            if len(piece.split()) > MAX_ITEM_WORDS:
                break
            items.append((pos + lead, pos + lead + len(piece), piece))
            if len(items) >= MAX_ITEMS:
                break
        if m is None:
            break
        pos = m.end()
    return items


def negated(text, start):
    window = text[max(0, start - 40):start]
    return bool(_NEG.search(window))


def _gold_spans(sentence, skills):
    spans = []
    for sk in skills:
        sp = find_quote(sk, sentence.text)
        if sp:
            spans.append(sp)
    return spans


def _apply_rule(rule, sentence):
    kind, key = rule
    text = sentence.text
    if kind == "segment":
        words = _WORD.findall(text)
        if sentence.section == key[0] and 0 < len(words) <= SEGMENT_MAX_WORDS:
            return [(0, len(text), text)]
        return []
    pattern = r"\b" + r"\s+".join(re.escape(w) for w in key) + r"\b"
    out = []
    for m in re.finditer(pattern, text, re.I):
        out.extend(_items_after(text, m.end()))
    return out


class RuleExtractor:
    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.rules = {}  # rule -> precision

    def fit(self, labeled):
        """labeled: list of (sentences, {sent_id: [gold skill strings]})."""
        candidates = set()
        for sents, gold in labeled:
            for s in sents:
                for start, end in _gold_spans(s, gold.get(s.sent_id, [])):
                    words = [w.lower() for w in _WORD.findall(s.text[:start])]
                    if words:
                        candidates.add(("context", tuple(words[-1:])))
                        if len(words) >= 2:
                            candidates.add(("context", tuple(words[-2:])))
                    if start == 0 and end >= len(s.text.rstrip(". ")):
                        candidates.add(("segment", (s.section,)))

        stats = defaultdict(lambda: [0, 0])
        for sents, gold in labeled:
            for s in sents:
                gold_ids = {canonical_id(g, self.lexicon, g) for g in gold.get(s.sent_id, [])}
                for rule in candidates:
                    for _, _, item in _apply_rule(rule, s):
                        hit = canonical_id(item, self.lexicon, item) in gold_ids
                        stats[rule][0 if hit else 1] += 1
        self.rules = {r: tp / (tp + fp) for r, (tp, fp) in stats.items()
                      if tp >= MIN_TP and tp / (tp + fp) >= MIN_PRECISION}
        return self

    def extract(self, doc_id, doc_type, sentences):
        best = {}
        for s in sentences:
            if s.section == "header":
                continue
            for rule, prec in self.rules.items():
                for start, end, item in _apply_rule(rule, s):
                    if negated(s.text, start):
                        continue
                    k = (s.sent_id, start, end)
                    if k not in best or best[k][1] < prec:
                        best[k] = (item, prec, rule)
        out = []
        sent_by_id = {s.sent_id: s for s in sentences}
        for (sid, start, end), (item, prec, rule) in best.items():
            out.append(row(doc_id, doc_type, "rules", item, self.lexicon, sent_by_id[sid],
                           (start, end), confidence=round(prec, 3), negated=False,
                           check=f"rule {rule[0]}: {' '.join(rule[1])}"))
        return out

"""Documents, the 50 gold documents, labeling sheets, and annotator agreement.

Workflow
  1. select_gold.py picks 25 resumes and 25 jobs from DATA/slice -> DATA/gold/selection.csv
  2. make_gold_sheets.py writes one CSV per gold document to DATA/gold/sheets/.
     Each row is one sentence (same sent_id as everywhere else in the project).
     "suggestions" lists what the extractors found; "skills" is for the annotator.
  3. An annotator saves a filled copy to DATA/gold/labels/A/ (or B/ for the
     second labeler).
  4. run_e1.py scores against labels/A and computes kappa where B exists.
"""
import csv
import random
from pathlib import Path

import pipeline

from . import config
from .extractors import canonical_id

SHEET_FIELDS = ["doc_id", "sent_id", "section", "sentence", "suggestions", "skills"]
SEP = ";"
N_GOLD_EACH = 25


def all_docs():
    """Every slice document as dicts with doc_id, doc_type, title, text."""
    docs = [{"doc_id": r["resume_id"], "doc_type": "resume", "title": r["category"], "text": r["text"]}
            for r in pipeline.read("resumes.csv")]
    docs += [{"doc_id": r["job_id"], "doc_type": "job", "title": r["title"], "text": r["description"]}
             for r in pipeline.read("postings.csv")]
    return docs


def select_gold():
    """25 resumes and 25 jobs from the slice, fixed seed. Writes selection.csv."""
    docs = all_docs()
    rng = random.Random(config.SEED)
    chosen = []
    for kind in ("resume", "job"):
        pool = sorted((d for d in docs if d["doc_type"] == kind), key=lambda d: d["doc_id"])
        chosen += rng.sample(pool, N_GOLD_EACH)
    config.GOLD.mkdir(parents=True, exist_ok=True)
    with open(config.GOLD / "selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "doc_type", "title"])
        for d in chosen:
            w.writerow([d["doc_id"], d["doc_type"], d["title"]])
    return chosen


def gold_docs():
    with open(config.GOLD / "selection.csv", newline="", encoding="utf-8") as f:
        ids = [r["doc_id"] for r in csv.DictReader(f)]
    by_id = {d["doc_id"]: d for d in all_docs()}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} gold documents are not in DATA/slice anymore: {missing[:5]}")
    return [by_id[i] for i in ids]


def write_sheet(path, doc_id, sentences, suggestions, prefill=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
        w.writeheader()
        for s in sentences:
            sug = f"{SEP} ".join(suggestions.get(s.sent_id, []))
            w.writerow({"doc_id": doc_id, "sent_id": s.sent_id, "section": s.section,
                        "sentence": s.text, "suggestions": sug, "skills": sug if prefill else ""})


def read_sheet(path, column="skills"):
    """Return {sent_id: [skill strings]} from a sheet's skills (or suggestions) column."""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            skills = [x.strip() for x in (r.get(column) or "").split(SEP) if x.strip()]
            if skills:
                out[int(r["sent_id"])] = skills
    return out


def labeled(annotator="A"):
    """{doc_id: {sent_id: [skills]}} for every sheet the annotator has saved."""
    folder = config.GOLD / "labels" / annotator
    return {p.stem: read_sheet(p) for p in sorted(Path(folder).glob("*.csv"))}


def doc_skill_set(sent_skills, lex):
    return {canonical_id(s, lex, s) for skills in sent_skills.values() for s in skills}


def cohen_kappa(a_labels, b_labels, lex, suggestions):
    """Kappa over (doc, sentence, skill) items either annotator could have marked.

    The item pool is everything either annotator marked plus every system
    suggestion, so a suggestion both annotators rejected counts as agreement on "no".
    """
    items, marked = set(), {"a": set(), "b": set()}
    for doc_id in set(a_labels) & set(b_labels):
        for who, labels in (("a", a_labels[doc_id]), ("b", b_labels[doc_id])):
            for sid, skills in labels.items():
                for s in skills:
                    key = (doc_id, sid, canonical_id(s, lex, s))
                    items.add(key)
                    marked[who].add(key)
        for sid, skills in suggestions.get(doc_id, {}).items():
            for s in skills:
                items.add((doc_id, sid, canonical_id(s, lex, s)))
    if not items:
        return None, 0
    a, b, n = marked["a"], marked["b"], len(items)
    agree = sum((i in a) == (i in b) for i in items) / n
    pa, pb = len(a) / n, len(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return ((agree - expected) / (1 - expected) if expected < 1 else 1.0), n

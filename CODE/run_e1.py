"""E1: does provenance-checked, ontology-directed extraction beat the baselines?

    python CODE/run_e1.py                              # extractor output in DATA/extracted/gold
    python CODE/run_e1.py --set gold_scout --name scout # output of run_extraction.py --out gold_scout

Arms, all scored against DATA/gold/labels/A at the document level (the set of
canonical skills in each document, negated mentions excluded):
  1 lexicon           the team's word matcher (classical baseline)
  2 llm_alone         plain LLM, no sentence and no quote
  3 llm_prov          LLM fills ontology slots and must quote its span
  4 llm_prov+rules    3 plus rules learned with 5-fold CV over the gold docs
  5 final             3 after the cleanup in extraction/clean.py (what skills_llm.csv holds)
  (rules_only is reported for reference)
Also reports how often the plain LLM names skills that are not in the document,
what the provenance check dropped, and Cohen's kappa where a second
annotator's labels exist (DATA/gold/labels/B).

Writes EVALUATIONS/e1/results[_<name>].csv and .md.
"""
import argparse
import csv
import random
from collections import Counter, defaultdict

from extraction import config
from extraction.clean import clean
from extraction.extractors import canonical_id, lexicon
from extraction.gold import cohen_kappa, doc_skill_set, gold_docs, labeled, read_sheet
from extraction.lexicon import skill_key
from extraction.metrics import bootstrap_ci, counts, micro, paired_diff
from extraction.provenance import GENERIC
from extraction.rules import RuleExtractor
from extraction.sentences import split

K_FOLDS = 5
OUT = config.EVAL / "e1"
ARMS = ("lexicon", "llm_alone", "llm_prov", "final", "llm_prov+rules", "rules_only")
N_GOLD = 50


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def skill_sets(rows):
    out = defaultdict(set)
    for r in rows:
        if str(r.get("negated")) != "True":
            out[r["doc_id"]].add(r["canonical_id"])
    return out


def folds(docs):
    """Stratified by document type, fixed seed."""
    rng = random.Random(config.SEED)
    out = [[] for _ in range(K_FOLDS)]
    for kind in ("resume", "job"):
        group = sorted(i for i, d in docs.items() if d["doc_type"] == kind)
        rng.shuffle(group)
        for n, i in enumerate(group):
            out[n % K_FOLDS].append(i)
    return out


def in_text(skill, text_words, text_key):
    words = [w for w in skill_key(skill).split() if w not in GENERIC]
    return bool(words) and (skill_key(skill) in text_key or all(w in text_words for w in words))


def present(row, lex, doc_skill_ids, text_words, text_key):
    """Is a plain-LLM skill written anywhere in the document, under any of its names?"""
    if row["canonical_id"] in doc_skill_ids:
        return True
    names = [row["skill"]] + [a for a, c in lex.alias_to_id.items() if c == row["canonical_id"]]
    return any(in_text(n, text_words, text_key) for n in names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="gold", help="folder under DATA/extracted with the arm outputs")
    ap.add_argument("--name", default="", help="suffix for the results files, e.g. scout")
    args = ap.parse_args()
    src = config.EXTRACTED / args.set
    lex = lexicon()
    gold = labeled("A")
    docs = {d["doc_id"]: d for d in gold_docs() if d["doc_id"] in gold}
    if not docs:
        raise SystemExit("No labeled gold docs in DATA/gold/labels/A yet. See DATA/gold/LABELING_GUIDE.md.")
    gold_sets = {i: doc_skill_set(gold[i], lex) for i in docs}
    sents = {i: split(d["text"]) for i, d in docs.items()}

    extracted = {a: read_csv(src / f"{a}.csv") for a in ("lexicon", "llm_alone", "llm_prov")}
    extracted["final"] = clean([r for r in extracted["llm_prov"]
                                if r["doc_id"] in docs and str(r.get("negated")) != "True"])
    preds = {a: skill_sets(rows) for a, rows in extracted.items()}
    models = sorted({r.get("model") for r in extracted["llm_prov"] if r.get("model")})

    # Rules: learn on 4 folds, apply to the held-out fold.
    preds["rules_only"], n_rules = defaultdict(set), []
    for fold in folds(docs):
        rx = RuleExtractor(lex).fit([(sents[i], gold[i]) for i in docs if i not in fold])
        n_rules.append(len(rx.rules))
        for i in fold:
            preds["rules_only"][i] = {r["canonical_id"] for r in
                                      rx.extract(i, docs[i]["doc_type"], sents[i])}
    preds["llm_prov+rules"] = {i: preds["llm_prov"][i] | preds["rules_only"][i] for i in docs}

    rows = []
    for arm in ARMS:
        for subset in ("all", "resume", "job"):
            ids = [i for i in docs if subset == "all" or docs[i]["doc_type"] == subset]
            if not ids:
                continue
            per_doc = [counts(preds[arm].get(i, set()), gold_sets[i]) for i in ids]
            p, r, f = micro(per_doc)
            (pl, ph), (rl, rh), (fl, fh) = bootstrap_ci(per_doc)
            rows.append({"arm": arm, "docs": subset, "n_docs": len(ids),
                         "precision": round(p, 3), "p_low": round(pl, 3), "p_high": round(ph, 3),
                         "recall": round(r, 3), "r_low": round(rl, 3), "r_high": round(rh, 3),
                         "f1": round(f, 3), "f1_low": round(fl, 3), "f1_high": round(fh, 3)})

    # Paired test: same resampled documents for both arms.
    pairs = []
    for base, arm in (("llm_alone", "llm_prov"), ("llm_alone", "final"), ("final", "llm_prov+rules")):
        ids = list(docs)
        a = [counts(preds[base].get(i, set()), gold_sets[i]) for i in ids]
        b = [counts(preds[arm].get(i, set()), gold_sets[i]) for i in ids]
        pairs.append((base, arm, paired_diff(a, b)))

    # Idea A evidence: what the plain LLM invents, and what the check removed.
    alone_total = alone_missing = 0
    for i, d in docs.items():
        key = skill_key(d["text"])
        words = set(key.split())
        found = {h[3] for h in lex.find(d["text"])}
        for r in extracted["llm_alone"]:
            if r["doc_id"] == i:
                alone_total += 1
                alone_missing += not present(r, lex, found, words, key)
    reasons, dropped_wrong = Counter(), 0
    for r in read_csv(src / "llm_prov_rejected.csv"):
        if r["doc_id"] in docs:
            reasons[r["reason"]] += 1
            cid = canonical_id(r["skill"] or "", lex, r["quote"] or None)
            dropped_wrong += cid not in gold_sets[r["doc_id"]]
    dropped = sum(reasons.values())

    kappa, n_items = None, 0
    second = labeled("B")
    if second:
        sugg = {p.stem: read_sheet(p, "suggestions") for p in (config.GOLD / "sheets").glob("*.csv")}
        kappa, n_items = cohen_kappa(gold, second, lex, sugg)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = "results" + (f"_{args.name}" if args.name else "")
    with open(OUT / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n_res = sum(d["doc_type"] == "resume" for d in docs.values())
    lines = [f"# E1 results: {len(docs)} of {N_GOLD} gold documents labeled ({n_res} resumes, {len(docs) - n_res} jobs)", "",
             f"Skill list: {lex.source}. LLM: {', '.join(models) or config.LLM_MODEL}. Rules: {K_FOLDS}-fold CV "
             f"({min(n_rules)}-{max(n_rules)} rules per fold). 95% bootstrap intervals over documents.",
             "", "| Extractor | Docs | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['docs']} | {r['precision']:.2f} [{r['p_low']:.2f}, {r['p_high']:.2f}] "
                     f"| {r['recall']:.2f} [{r['r_low']:.2f}, {r['r_high']:.2f}] "
                     f"| {r['f1']:.2f} [{r['f1_low']:.2f}, {r['f1_high']:.2f}] |")
    lines += ["", "## Paired differences (all documents)", "",
              "Each bootstrap draw resamples the same documents for both extractors. "
              "p = share of draws where the second is not better.", "",
              "| Comparison | Precision | Recall | F1 |", "|---|---|---|---|"]
    for base, arm, d in pairs:
        lines.append(f"| {arm} vs {base} | " + " | ".join(
            f"{x:+.2f} [{lo:+.2f}, {hi:+.2f}] p={pv:.3f}" for x, lo, hi, pv in d) + " |")
    lines += ["", "## Provenance (Idea A)", "",
              f"- Plain LLM skills not found anywhere in the document: {alone_missing}/{alone_total}"
              f" ({alone_missing / max(alone_total, 1):.1%})",
              f"- Items the provenance check dropped: {dropped}"
              f" ({', '.join(f'{k}: {v}' for k, v in reasons.most_common()) or 'none'})",
              f"- Dropped items that are not in the gold labels (correct drops): "
              f"{dropped_wrong}/{dropped} ({dropped_wrong / max(dropped, 1):.1%})"]
    if kappa is not None:
        lines += ["", f"Cohen's kappa (annotators A vs B, {n_items} items): {kappa:.3f}"]
    (OUT / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

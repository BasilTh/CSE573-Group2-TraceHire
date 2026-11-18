"""Run the skill extractors over a document set and save rows in the shared skills format.

    python CODE/run_extraction.py --set gold                       # the E1 arms on the 50 gold docs
    python CODE/run_extraction.py --set resumes --arms final       # graph input, needs gold labels for rules
    python CODE/run_extraction.py --set jobs --limit 200 --arms final
    python CODE/run_extraction.py --set slice --arms final --first 83816738 26746496
                                   # every resume and job -> DATA/slice/skills_llm.csv

Writes DATA/extracted/<set>/<arm>.csv and DATA/extracted/<set>/llm_prov_rejected.csv
(what the provenance check dropped, with the reason). LLM answers are cached in
DATA/cache/llm, so reruns only pay for new text. See CODE/extraction/README.md.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from extraction import config, llm
from extraction.extractors import (FIELDS, extract_final, extract_lexicon, extract_llm_alone,
                                   extract_llm_prov)
from extraction.gold import all_docs, gold_docs
from extraction.sentences import split
from extraction.service import learned_rules

ARMS = ["lexicon", "llm_alone", "llm_prov", "final"]
REJECT_FIELDS = ["doc_id", "doc_type", "sent_id", "sentence", "skill", "quote", "reason"]


def load_set(name):
    if name == "gold":
        return gold_docs()
    docs = all_docs()
    if name == "slice":
        return docs
    if name == "resumes":
        return [d for d in docs if d["doc_type"] == "resume"]
    if name == "jobs":
        return [d for d in docs if d["doc_type"] == "job"]
    raise SystemExit(f"unknown set {name}; use gold, resumes, jobs or slice")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="gold", choices=["gold", "resumes", "jobs", "slice"])
    ap.add_argument("--arms", nargs="+", default=["lexicon", "llm_alone", "llm_prov"], choices=ARMS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4, help="documents processed in parallel")
    ap.add_argument("--first", nargs="+", default=[], help="doc_ids to process before the rest")
    ap.add_argument("--out", help="folder name under DATA/extracted (default: the set name), e.g. gold_scout")
    ap.add_argument("--no-rules", action="store_true",
                    help="final = llm_prov without negated mentions, even if gold labels exist")
    args = ap.parse_args()

    docs = load_set(args.set)
    docs = sorted(docs, key=lambda d: d["doc_id"] not in args.first)[:args.limit]
    rules = learned_rules() if "final" in args.arms and not args.no_rules else None
    if "final" in args.arms:
        print(f"rules: {len(rules.rules)} learned from gold labels" if rules
              else "rules: off, final = llm_prov without negated mentions")
    print(f"{len(docs)} docs, model {config.LLM_MODEL} at {config.LLM_BASE_URL}")

    def run(d):
        sents = split(d["text"])
        args_ = (d["doc_id"], d["doc_type"], d["text"])
        out = {}
        if "lexicon" in args.arms:
            out["lexicon"] = extract_lexicon(*args_, sentences=sents)
        if "llm_alone" in args.arms:
            out["llm_alone"] = extract_llm_alone(*args_, sentences=sents)
        if "llm_prov" in args.arms:
            out["llm_prov"], out["rejected"] = extract_llm_prov(*args_, sentences=sents)
        if "final" in args.arms:
            out["final"] = extract_final(*args_, rules=rules, sentences=sents)
        return out

    out_dir = config.EXTRACTED / (args.out or args.set)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers, files = {}, []
    for name in [a for a in args.arms] + (["rejected"] if "llm_prov" in args.arms else []):
        path = out_dir / ("llm_prov_rejected.csv" if name == "rejected" else f"{name}.csv")
        if args.set == "slice" and name == "final":
            path = config.SLICE / "skills_llm.csv"   # the team's file, next to skills.csv
        f = open(path, "w", newline="", encoding="utf-8")
        files.append(f)
        writers[name] = csv.DictWriter(f, fieldnames=REJECT_FIELDS if name == "rejected" else FIELDS)
        writers[name].writeheader()

    with ThreadPoolExecutor(args.workers) as pool:
        for out in tqdm(pool.map(run, docs), total=len(docs)):
            for name, rows in out.items():
                writers[name].writerows(rows)
            for f in files:
                f.flush()   # rows land on disk document by document
    for f in files:
        f.close()
    print(f"wrote {', '.join(writers)} to {out_dir}")
    if llm.FAILURES:
        print(f"WARNING: {len(llm.FAILURES)} LLM batches failed and were skipped; rerun to retry them")


if __name__ == "__main__":
    main()

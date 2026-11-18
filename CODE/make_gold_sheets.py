"""Pick the 50 gold documents and write one labeling sheet per document.

    python CODE/make_gold_sheets.py --select     # once: 25 resumes + 25 jobs -> DATA/gold/selection.csv
    python CODE/run_extraction.py --set gold     # fills the suggestions
    python CODE/make_gold_sheets.py              # -> DATA/gold/sheets/<doc_id>.csv

"suggestions" holds the words the lexicon and the LLM+provenance extractor
marked in each sentence, so an annotator can copy instead of typing. With
--prefill the "skills" column starts as a copy of the suggestions (faster, but
it nudges annotators toward the systems being tested; say so in the report).
See DATA/gold/LABELING_GUIDE.md.
"""
import argparse
import csv
from collections import defaultdict

from extraction import config
from extraction.gold import gold_docs, select_gold, write_sheet
from extraction.sentences import split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true", help="choose the gold documents")
    ap.add_argument("--prefill", action="store_true")
    args = ap.parse_args()

    if args.select:
        chosen = select_gold()
        print(f"selected {len(chosen)} gold documents -> {config.GOLD / 'selection.csv'}")
        return

    sugg = defaultdict(lambda: defaultdict(list))
    for arm in ("lexicon", "llm_prov"):
        path = config.EXTRACTED / "gold" / f"{arm}.csv"
        if not path.exists():
            raise SystemExit(f"missing {path}; run run_extraction.py --set gold first")
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["span"] and r["negated"] != "True":
                    seen = sugg[r["doc_id"]][int(r["sent_id"])]
                    if r["span"].lower() not in (s.lower() for s in seen):
                        seen.append(r["span"])

    docs = gold_docs()
    for d in docs:
        write_sheet(config.GOLD / "sheets" / f"{d['doc_id']}.csv", d["doc_id"], split(d["text"]),
                    sugg.get(d["doc_id"], {}), prefill=args.prefill)
    print(f"wrote {len(docs)} sheets to {config.GOLD / 'sheets'}")


if __name__ == "__main__":
    main()

# Extraction (E1): Mathew

This folder replaces the inside of `pipeline.extract_skills` with the method from our proposal:
an LLM reads each sentence, fills the ontology slots, and **must quote the words it used**.
Anything it can't quote from that sentence is dropped. It also holds the baselines and the
scoring for experiment E1.

Nothing else in the project has to change. The rows come out in the shared skills format
from `DATA/README.md`, with the same `sent_id` numbers as the rest of the code (it uses
`pipeline.split_sentences`).

## Using it

```python
import pipeline
rows = pipeline.extract_skills(resume_text)                    # the word matcher, as before
rows = pipeline.extract_skills(resume_text, extractor="llm")   # the LLM + provenance check
```

Or set `TRACEHIRE_EXTRACTOR=llm` before `streamlit run CODE/app.py` to switch the whole app.
The app still uses the word matcher by default, because the LLM takes about 10-30 seconds per
new resume on a laptop. Answers are cached, so a document already seen is instant.

The LLM runs locally with [Ollama](https://ollama.com):

```sh
brew install ollama
OLLAMA_NUM_PARALLEL=4 ollama serve      # leave running in its own terminal
ollama pull llama3.1:8b                 # once, about 5 GB
pip install -r requirements.txt
```

Llama 3.1 8B (the model in our proposal) is not on the ASU API, so extraction runs it locally.
It has its own settings, `EXTRACTION_BASE_URL` and `EXTRACTION_MODEL`, so the team's
`OPENAI_BASE_URL` (for the agents) doesn't redirect it by accident. To run extraction on the ASU API,
set those two in `.env` to the API's URL and one of its models. The key is read from the team's
`OPENAI_API_KEY`. See `.env.example`.

You don't need Ollama to *use* the results: the extractor output files are in the repo, and
`extractor="lexicon"` needs no LLM at all.

## How it works

```
document ─▶ pipeline.split_sentences ─▶ numbered sentences (batches of up to 8)
                                              │
                         Llama 3.1 8B fills the ontology for each skill:
                         skill, quote, class (tool/technique/domain), negated,
                         proficiency, required/preferred
                                              │
                     provenance check: is the quote in that sentence,
                     and does it actually name the skill?
                          │ yes                                  │ no
                          ▼                                      ▼
            map to the skill list (canonical_id)      dropped, logged with the reason
                          │
            + skills found by rules learned from the gold labels
            − negated mentions ("no Java experience")
                          ▼
            rows in the shared skills format ─▶ graph, matching, app
```

| File | What it does |
|---|---|
| `sentences.py` | Wraps the shared splitter and records which section each sentence is in |
| `lexicon.py` | The skill list (O*NET now, Lightcast later) with IDs and aliases ("Spark" → "Apache Spark") |
| `llm.py` | Talks to Ollama or Voyager, with a disk cache (`DATA/cache/llm`) |
| `provenance.py` | The provenance check |
| `extractors.py` | The extractors and the shared row format |
| `rules.py` | Learns patterns such as "experience with [list]" from gold labels, plus a negation filter |
| `service.py` | What `pipeline.extract_skills(..., extractor="llm")` calls |
| `gold.py` | Gold document selection, labeling sheets, Cohen's kappa |
| `metrics.py` | Precision, recall, F1 and bootstrap confidence intervals |

## The provenance check

A skill is kept only if both are true:
1. **The quote is in the sentence** the LLM pointed to: whole words, ignoring capital letters
   and extra spaces. If the LLM gave the wrong sentence number but the quote occurs in exactly
   one sentence of the batch, the row is kept and marked `sent_id_corrected`.
2. **The quote names the skill.** Every word of the skill appears in the quote (allowing
   "analyses" for "analysis"), or the skill list maps both to the same skill.

Real examples from the 50 gold documents (resume 18752129 and others):

| LLM said | Its quote | The sentence | Result |
|---|---|---|---|
| SAP | "SAP" | "Company Name" | dropped: invented. SAP and GAAP are examples in our prompt, and the LLM copied them. |
| Java | "Java" | "Software Engineer" | dropped: invented |
| vulnerability assessment | "vulnerability and risk" | "Analyzed complex computer systems to assess vulnerability and risk." | dropped: the quote doesn't name the skill |
| network software | "Network software" | "Upgrading Network hardware, software and accessories." | dropped, but this one was real. The quote must be continuous text, so split phrases cost some recall. |

On the gold run the check dropped 351 of 3,464 LLM items: 164 quotes not in the sentence and
186 quotes that don't name the skill. E1 measures how many of those drops were right. For
comparison, 8.7% of the plain LLM's skills (266 of 3,056) appear nowhere in their document.

Long answers: if a reply is cut off at the token limit, it is not cached. The batch is split in half
and asked again, so skill-heavy sections such as resume skill lists aren't lost. A single sentence
too long even on its own is skipped and reported at the end of the run (3 on the gold set).
Every rejection goes to `DATA/extracted/<set>/llm_prov_rejected.csv`.

## Output columns

The shared columns are `doc_id, doc_type, sent_id, sentence, span, skill, vocab, extractor, confidence`.
Extra columns:

| Column | Meaning |
|---|---|
| `canonical_id` | Stable skill ID (`onet:...`). `raw:...` means the skill isn't in the list yet. Use it as the Skill node key. |
| `span_start`, `span_end` | Where the span sits inside `sentence`, for highlighting |
| `skill_class` | `tool`, `technique` or `domain` |
| `negated` | `True` for "no experience with X". The final output already removes these. |
| `proficiency` | `basic`, `intermediate`, `advanced` when stated |
| `requirement` | `required` or `preferred` (jobs) |
| `check` | `ok`, `sent_id_corrected`, or the rule that found it |
| `model` | which LLM produced it |

`skill` uses the O*NET name when the skill is in the list (so it lines up with the word matcher),
otherwise the LLM's name with `vocab = none`.

## E1: how we prove it works

```sh
python CODE/make_gold_sheets.py --select     # done: 25 resumes + 25 jobs from DATA/slice
python CODE/run_extraction.py --set gold     # done: runs lexicon, llm_alone, llm_prov
python CODE/make_gold_sheets.py              # done: one labeling sheet per gold document
# label the sheets (DATA/gold/LABELING_GUIDE.md), then:
python CODE/run_e1.py                        # -> EVALUATIONS/e1/results.md
```

| Arm | Method |
|---|---|
| `lexicon` | the team's word matcher. Classical baseline. |
| `llm_alone` | the LLM lists skills with no sentence and no quote |
| `llm_prov` | ontology slots + provenance check |
| `llm_prov+rules` | plus rules, learned with 5-fold cross-validation so no document is scored by rules learned from it |

`run_e1.py` reports precision, recall and F1 (overall, resumes, jobs) with 95% bootstrap intervals.
It also reports how often the plain LLM names skills that appear nowhere in the document, what
the provenance check dropped and how many of those drops were correct, and Cohen's kappa
between two labelers.

## Building the graph input

After gold labels exist, so rules can be learned:

```sh
python CODE/run_extraction.py --set resumes --arms final     # 238 resumes, about 1 hour
python CODE/run_extraction.py --set jobs --arms final        # 2,000 jobs, a few hours; try --limit 200 first
```

Output: `DATA/extracted/<set>/final.csv`.

## Tests

```sh
python CODE/tests/test_extraction.py
```

## Open items
- **Gold labels**: the 50 sheets are in `DATA/gold/sheets/`. Assignments are in `DATA/gold/assignments.md`.
- **Lightcast**: drop the skills list at `DATA/raw/lightcast/skills.json` and the code switches to it.
- **Gold jobs depend on the slice**: if the slice is regenerated, the 50 gold job IDs must stay in it.
  `gold.py` stops with an error if any are missing.

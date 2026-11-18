# E1: skill extraction (all 50 gold documents)

All 50 gold documents are labeled (25 resumes, 25 job postings) in `DATA/gold/labels/A`.
12 of them have a second, independent label in `DATA/gold/labels/B`.

Main run: `llama4-scout-17b` on Voyager, the model the app uses. Full tables: `results_scout.md`.
Same arms with Llama 3.1 8B (the model in the proposal): `results_llama31.md`.
95% bootstrap intervals over documents in brackets.

| Extractor (Scout) | Precision | Recall | F1 |
|---|---|---|---|
| Word matcher (lexicon baseline) | 0.90 [0.86, 0.94] | 0.10 | 0.18 [0.15, 0.22] |
| Plain LLM | 0.47 [0.43, 0.51] | 0.65 | 0.55 [0.51, 0.58] |
| **LLM + ontology slots + provenance check** | **0.65 [0.61, 0.68]** | 0.57 | **0.60 [0.57, 0.63]** |
| ... + cleanup (what `skills_llm.csv` holds) | 0.65 | 0.57 | 0.61 [0.57, 0.64] |
| ... + learned rules | 0.51 | 0.61 | 0.56 |

Paired comparison (each bootstrap draw resamples the same documents for both extractors):

| Scout | Precision | Recall | F1 |
|---|---|---|---|
| Provenance check vs plain LLM | +0.18 [+0.14, +0.21] | -0.08 [-0.12, -0.05] | **+0.06 [+0.02, +0.09]** |
| Learned rules added vs final | -0.14 [-0.17, -0.12] | +0.04 [+0.02, +0.07] | -0.05 [-0.06, -0.04] |

What it shows:
- **The provenance check makes the LLM's skills much more trustworthy.** Precision rises from
  0.47 to 0.65. F1 rises from 0.55 to 0.60. The separate intervals overlap a little, but the
  paired test, which compares both extractors on the same documents, puts the F1 gain at
  +0.06 with a 95% interval of +0.02 to +0.09, so the gain is real for Scout.
- **The gain comes from resumes.** On resumes F1 goes 0.53 to 0.62. On job postings
  precision still rises (0.58 to 0.71) but recall falls more (0.60 to 0.45), so F1 is about
  the same (0.59 vs 0.55, intervals overlap). Postings phrase skills in long lists and
  requirement sentences that the LLM paraphrases instead of quoting.
- **The plain LLM names skills that are not in the document:** 13.1% of its skills
  (440 of 3,359) appear nowhere in the text, under any of their names.
- **The word matcher is precise but finds about 1 skill in 10.**
- **The cost:** the check gives up recall (0.65 to 0.57). Of the 146 items it dropped, 82 (56%)
  were not in the gold labels, so close to half of the drops were real skills. Most of those are
  phrases the LLM paraphrased instead of quoting, or skills split across a sentence.
- **Learned rules hurt.** They add recall but lower precision more, so F1 falls by 0.05
  (paired interval -0.06 to -0.04). The final extractor does not use them.
- **The cleanup** (soft skills, degrees, employer names) changes the numbers only slightly here,
  because it mostly affects job postings outside the gold set.
- **With Llama 3.1 8B the gain does not show.** F1 is 0.51 for both the plain LLM and the
  checked version (paired F1 difference +0.01, interval -0.03 to +0.04), and precision barely
  moves (0.45 to 0.47). The check does reject 351 items, and 70% of them are rightly rejected,
  but the smaller model's slot answers miss about as many skills as the check removes.
  Its plain answers invent fewer skills (9.0%). So the method helps the stronger model the app
  uses, and we cannot claim it for the 8B model.

Agreement between labelers: Cohen's kappa 0.37 over 900 skill decisions on the 12 documents
that have second labels. That is "fair" agreement: deciding what counts as a skill is subjective
(for example, whether "vendor management" or "RFP" is a skill). `DATA/gold/notes.md` lists the
cases we settled together.

Skill names are matched through the O*NET list with a few alias fixes
(generic acronyms such as "AI" no longer map to a product; React.js and React share one ID).

Reproduce:
```sh
python CODE/run_extraction.py --set gold --out gold_scout --arms lexicon llm_alone llm_prov   # needs EXTRACTION_* set to Voyager
python CODE/run_e1.py --set gold_scout --name scout
python CODE/run_e1.py --set gold --name llama31
```

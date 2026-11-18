# E1 results: 50 of 50 gold documents labeled (25 resumes, 25 jobs)

Skill list: onet. LLM: llama3.1:8b. Rules: 5-fold CV (366-431 rules per fold). 95% bootstrap intervals over documents.

| Extractor | Docs | Precision | Recall | F1 |
|---|---|---|---|---|
| lexicon | all | 0.90 [0.86, 0.94] | 0.10 [0.08, 0.12] | 0.18 [0.15, 0.22] |
| lexicon | resume | 0.89 [0.83, 0.94] | 0.10 [0.08, 0.12] | 0.17 [0.14, 0.21] |
| lexicon | job | 0.93 [0.86, 1.00] | 0.11 [0.06, 0.16] | 0.20 [0.12, 0.28] |
| llm_alone | all | 0.45 [0.41, 0.49] | 0.58 [0.53, 0.62] | 0.51 [0.47, 0.54] |
| llm_alone | resume | 0.46 [0.43, 0.50] | 0.61 [0.56, 0.66] | 0.53 [0.49, 0.56] |
| llm_alone | job | 0.42 [0.33, 0.52] | 0.48 [0.39, 0.56] | 0.45 [0.36, 0.53] |
| llm_prov | all | 0.47 [0.43, 0.50] | 0.56 [0.51, 0.61] | 0.51 [0.48, 0.54] |
| llm_prov | resume | 0.45 [0.42, 0.49] | 0.61 [0.56, 0.67] | 0.52 [0.49, 0.55] |
| llm_prov | job | 0.52 [0.42, 0.63] | 0.42 [0.34, 0.51] | 0.47 [0.39, 0.54] |
| final | all | 0.47 [0.44, 0.51] | 0.56 [0.51, 0.61] | 0.51 [0.48, 0.54] |
| final | resume | 0.46 [0.42, 0.49] | 0.61 [0.56, 0.67] | 0.52 [0.50, 0.55] |
| final | job | 0.54 [0.44, 0.64] | 0.42 [0.34, 0.51] | 0.48 [0.40, 0.55] |
| llm_prov+rules | all | 0.41 [0.39, 0.44] | 0.62 [0.57, 0.67] | 0.50 [0.47, 0.52] |
| llm_prov+rules | resume | 0.41 [0.38, 0.44] | 0.67 [0.62, 0.73] | 0.51 [0.48, 0.54] |
| llm_prov+rules | job | 0.44 [0.36, 0.52] | 0.46 [0.39, 0.53] | 0.45 [0.38, 0.51] |
| rules_only | all | 0.40 [0.33, 0.44] | 0.20 [0.13, 0.27] | 0.27 [0.19, 0.34] |
| rules_only | resume | 0.42 [0.37, 0.47] | 0.25 [0.17, 0.34] | 0.31 [0.23, 0.39] |
| rules_only | job | 0.22 [0.09, 0.35] | 0.06 [0.02, 0.11] | 0.10 [0.03, 0.17] |

## Paired differences (all documents)

Each bootstrap draw resamples the same documents for both extractors. p = share of draws where the second is not better.

| Comparison | Precision | Recall | F1 |
|---|---|---|---|
| llm_prov vs llm_alone | +0.01 [-0.02, +0.05] p=0.241 | -0.02 [-0.07, +0.03] p=0.734 | +0.00 [-0.03, +0.04] p=0.455 |
| final vs llm_alone | +0.02 [-0.02, +0.06] p=0.126 | -0.02 [-0.07, +0.03] p=0.742 | +0.01 [-0.03, +0.04] p=0.368 |
| llm_prov+rules vs final | -0.06 [-0.07, -0.04] p=1.000 | +0.06 [+0.02, +0.10] p=0.000 | -0.02 [-0.04, +0.01] p=0.919 |

## Provenance (Idea A)

- Plain LLM skills not found anywhere in the document: 275/3056 (9.0%)
- Items the provenance check dropped: 351 (skill_not_in_quote: 186, quote_not_in_sentence: 164, no_skill: 1)
- Dropped items that are not in the gold labels (correct drops): 244/351 (69.5%)

Cohen's kappa (annotators A vs B, 900 items): 0.368

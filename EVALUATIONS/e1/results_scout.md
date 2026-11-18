# E1 results: 50 of 50 gold documents labeled (25 resumes, 25 jobs)

Skill list: onet. LLM: llama4-scout-17b. Rules: 5-fold CV (366-431 rules per fold). 95% bootstrap intervals over documents.

| Extractor | Docs | Precision | Recall | F1 |
|---|---|---|---|---|
| lexicon | all | 0.90 [0.86, 0.94] | 0.10 [0.08, 0.12] | 0.18 [0.15, 0.22] |
| lexicon | resume | 0.89 [0.83, 0.94] | 0.10 [0.08, 0.12] | 0.17 [0.14, 0.21] |
| lexicon | job | 0.93 [0.86, 1.00] | 0.11 [0.06, 0.16] | 0.20 [0.12, 0.28] |
| llm_alone | all | 0.47 [0.43, 0.51] | 0.65 [0.62, 0.68] | 0.55 [0.51, 0.58] |
| llm_alone | resume | 0.45 [0.41, 0.48] | 0.67 [0.63, 0.71] | 0.53 [0.50, 0.57] |
| llm_alone | job | 0.58 [0.47, 0.67] | 0.60 [0.52, 0.68] | 0.59 [0.50, 0.67] |
| llm_prov | all | 0.65 [0.61, 0.68] | 0.57 [0.52, 0.60] | 0.60 [0.57, 0.63] |
| llm_prov | resume | 0.63 [0.59, 0.67] | 0.60 [0.57, 0.64] | 0.62 [0.58, 0.65] |
| llm_prov | job | 0.71 [0.61, 0.79] | 0.45 [0.37, 0.53] | 0.55 [0.47, 0.62] |
| final | all | 0.65 [0.62, 0.69] | 0.57 [0.52, 0.60] | 0.61 [0.57, 0.64] |
| final | resume | 0.64 [0.60, 0.67] | 0.60 [0.57, 0.64] | 0.62 [0.59, 0.65] |
| final | job | 0.71 [0.62, 0.80] | 0.45 [0.37, 0.53] | 0.56 [0.47, 0.62] |
| llm_prov+rules | all | 0.51 [0.48, 0.54] | 0.61 [0.56, 0.65] | 0.56 [0.52, 0.58] |
| llm_prov+rules | resume | 0.50 [0.48, 0.54] | 0.65 [0.60, 0.70] | 0.57 [0.54, 0.59] |
| llm_prov+rules | job | 0.55 [0.46, 0.64] | 0.48 [0.39, 0.56] | 0.51 [0.43, 0.58] |
| rules_only | all | 0.40 [0.33, 0.44] | 0.20 [0.13, 0.27] | 0.27 [0.19, 0.34] |
| rules_only | resume | 0.42 [0.37, 0.47] | 0.25 [0.17, 0.34] | 0.31 [0.23, 0.39] |
| rules_only | job | 0.22 [0.09, 0.35] | 0.06 [0.02, 0.11] | 0.10 [0.03, 0.17] |

## Paired differences (all documents)

Each bootstrap draw resamples the same documents for both extractors. p = share of draws where the second is not better.

| Comparison | Precision | Recall | F1 |
|---|---|---|---|
| llm_prov vs llm_alone | +0.18 [+0.14, +0.21] p=0.000 | -0.08 [-0.12, -0.05] p=1.000 | +0.06 [+0.02, +0.09] p=0.000 |
| final vs llm_alone | +0.18 [+0.15, +0.22] p=0.000 | -0.08 [-0.12, -0.05] p=1.000 | +0.06 [+0.03, +0.09] p=0.000 |
| llm_prov+rules vs final | -0.14 [-0.17, -0.12] p=1.000 | +0.04 [+0.02, +0.07] p=0.000 | -0.05 [-0.06, -0.04] p=1.000 |

## Provenance (Idea A)

- Plain LLM skills not found anywhere in the document: 440/3359 (13.1%)
- Items the provenance check dropped: 146 (skill_not_in_quote: 78, quote_not_in_sentence: 64, no_quote: 4)
- Dropped items that are not in the gold labels (correct drops): 82/146 (56.2%)

Cohen's kappa (annotators A vs B, 900 items): 0.368

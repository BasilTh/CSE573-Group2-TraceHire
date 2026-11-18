# E2 results: matching, skills from `skills_llm.csv`

20 resumes (10 IT, 10 Engineering, no gold documents), ranked against 2000 postings. Top 10 per method, 541 pooled pairs in this run.
Hybrid uses alpha = 0.5, not tuned yet. The graph arm (LightGCN) is not built yet; skill overlap stands in for the graph score.

### Automatic relevance, strict: same O*NET broad occupation (preliminary)

| Method | P@5 (95% CI) | nDCG@10 (95% CI) |
|---|---|---|
| tfidf | 0.09 (0.00 to 0.20) | 0.08 (0.01 to 0.16) |
| sbert | 0.06 (0.01 to 0.13) | 0.07 (0.02 to 0.13) |
| skills | 0.09 (0.02 to 0.18) | 0.09 (0.02 to 0.19) |
| hybrid | 0.12 (0.04 to 0.23) | 0.10 (0.04 to 0.18) |

### Automatic relevance, lenient: same O*NET minor group (preliminary)

| Method | P@5 (95% CI) | nDCG@10 (95% CI) |
|---|---|---|
| tfidf | 0.20 (0.08 to 0.34) | 0.17 (0.07 to 0.27) |
| sbert | 0.16 (0.07 to 0.27) | 0.18 (0.09 to 0.29) |
| skills | 0.19 (0.07 to 0.33) | 0.19 (0.09 to 0.32) |
| hybrid | 0.22 (0.09 to 0.37) | 0.20 (0.09 to 0.32) |

How each resume title was mapped is in `onet_labels.csv`. Titles such as "Objective" or "ENGINEERING TECH" map poorly, so treat these numbers as a rough first look.

### Judged relevance (judged by rishi; an unjudged pair counts as not a fit)

| Method | P@5 (95% CI) | nDCG@10 (95% CI) | Top-10 pairs judged |
|---|---|---|---|
| tfidf | 0.65 (0.53 to 0.76) | 0.65 (0.54 to 0.76) | 200 of 200 |
| sbert | 0.56 (0.41 to 0.70) | 0.57 (0.43 to 0.70) | 200 of 200 |
| skills | 0.43 (0.29 to 0.57) | 0.44 (0.31 to 0.56) | 200 of 200 |
| hybrid | 0.50 (0.37 to 0.62) | 0.52 (0.40 to 0.64) | 200 of 200 |

541 of this run's 541 pooled pairs are judged.

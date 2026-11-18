# E2 results: matching, skills from `skills.csv`

20 resumes (10 IT, 10 Engineering, no gold documents), ranked against 2000 postings. Top 10 per method, 586 pooled pairs in this run.
Hybrid uses alpha = 0.5, not tuned yet. The graph arm (LightGCN) is not built yet; skill overlap stands in for the graph score.

### Automatic relevance, strict: same O*NET broad occupation (preliminary)

| Method | P@5 (95% CI) | nDCG@10 (95% CI) |
|---|---|---|
| tfidf | 0.09 (0.00 to 0.20) | 0.08 (0.01 to 0.16) |
| sbert | 0.06 (0.01 to 0.13) | 0.07 (0.02 to 0.13) |
| skills | 0.05 (0.01 to 0.10) | 0.06 (0.01 to 0.12) |
| hybrid | 0.07 (0.02 to 0.13) | 0.08 (0.03 to 0.15) |

### Automatic relevance, lenient: same O*NET minor group (preliminary)

| Method | P@5 (95% CI) | nDCG@10 (95% CI) |
|---|---|---|
| tfidf | 0.20 (0.08 to 0.34) | 0.17 (0.07 to 0.27) |
| sbert | 0.16 (0.07 to 0.27) | 0.18 (0.09 to 0.29) |
| skills | 0.13 (0.04 to 0.24) | 0.15 (0.05 to 0.28) |
| hybrid | 0.15 (0.05 to 0.28) | 0.17 (0.07 to 0.31) |

How each resume title was mapped is in `onet_labels.csv`. Titles such as "Objective" or "ENGINEERING TECH" map poorly, so treat these numbers as a rough first look.

### Judged relevance (judged by rishi; an unjudged pair counts as not a fit)

| Method | P@5 (95% CI) | nDCG@10 (95% CI) | Top-10 pairs judged |
|---|---|---|---|
| tfidf | 0.65 (0.53 to 0.76) | 0.67 (0.56 to 0.77) | 200 of 200 |
| sbert | 0.56 (0.41 to 0.70) | 0.58 (0.45 to 0.71) | 200 of 200 |
| skills | 0.28 (0.17 to 0.40) | 0.28 (0.17 to 0.40) | 200 of 200 |
| hybrid | 0.43 (0.30 to 0.56) | 0.45 (0.33 to 0.57) | 200 of 200 |

586 of this run's 586 pooled pairs are judged.

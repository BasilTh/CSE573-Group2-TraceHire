# E3 results: skill advice, hidden-skill test

Skills from `skills.csv`: 348 distinct skills in the graph. 158 resumes with at least 3 skills, 1216 hidden-skill trials (each skill of each resume hidden once).
Recovered means the hidden skill is in the method's top 10. The deck's target is 0.70.

96% of the hidden skills appear in some other document. The rest cannot be recovered by any method.

### All skills

| Method | Recovery in top 10 (95% CI) | O*NET hot share of the top 10 |
|---|---|---|
| popularity | 0.29 (0.26 to 0.32) | 0.90 |
| advisor | 0.49 (0.45 to 0.52) | 0.91 |
| graph | 0.48 (0.45 to 0.51) | 0.97 |

### Only skills on the O*NET list (1216 trials, 158 resumes)

These are the same kind of target in every skills file, so this table is the one to compare across files.

| Method | Recovery in top 10 (95% CI) | O*NET hot share of the top 10 |
|---|---|---|
| popularity | 0.29 (0.26 to 0.32) | 0.90 |
| advisor | 0.49 (0.45 to 0.52) | 0.91 |
| graph | 0.48 (0.45 to 0.51) | 0.97 |

For scale: 118 of the 348 skills in the graph are O*NET hot technologies (0.34).
The graph method here is personalized PageRank. It stands in for the trained graph model (LightGCN), which is not built yet.

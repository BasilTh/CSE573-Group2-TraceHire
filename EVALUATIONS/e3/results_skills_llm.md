# E3 results: skill advice, hidden-skill test

Skills from `skills_llm.csv`: 19190 distinct skills in the graph. 237 resumes with at least 3 skills, 13854 hidden-skill trials (each skill of each resume hidden once).
Recovered means the hidden skill is in the method's top 10. The deck's target is 0.70.

63% of the hidden skills appear in some other document. The rest cannot be recovered by any method.

### All skills

| Method | Recovery in top 10 (95% CI) | O*NET hot share of the top 10 |
|---|---|---|
| popularity | 0.03 (0.03 to 0.04) | 0.56 |
| advisor | 0.08 (0.07 to 0.09) | 0.49 |
| graph | 0.08 (0.07 to 0.09) | 0.60 |

### Only skills on the O*NET list (2700 trials, 233 resumes)

These are the same kind of target in every skills file, so this table is the one to compare across files.

| Method | Recovery in top 10 (95% CI) | O*NET hot share of the top 10 |
|---|---|---|
| popularity | 0.12 (0.10 to 0.14) | 0.56 |
| advisor | 0.27 (0.25 to 0.30) | 0.49 |
| graph | 0.27 (0.25 to 0.30) | 0.60 |

For scale: 151 of the 19190 skills in the graph are O*NET hot technologies (0.01).
The graph method here is personalized PageRank. It stands in for the trained graph model (LightGCN), which is not built yet.

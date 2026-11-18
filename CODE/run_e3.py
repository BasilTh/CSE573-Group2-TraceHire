"""E3: skill advice, the hidden-skill test.

For every resume with at least 3 skills, each skill is hidden in turn. The advice
method sees the remaining skills and suggests 10 to learn. The hidden skill counts
as recovered if it is among the 10. Recovery is averaged per resume and reported
with a 95% bootstrap interval over resumes.

    popularity  the skills most postings ask for, the same list for everyone
    advisor     pipeline.skills_to_learn: counts over the 100 closest postings
    graph       personalized PageRank on the skill graph: a random walk that starts
                at the person's skills, steps skill -> document -> skill, and keeps
                jumping back to the start. The test resume is left out of the graph.

Also reported: the share of each method's 10 suggestions (nothing hidden) that
O*NET flags as a hot technology.

Run from the repo root:
    python CODE/run_e3.py                   # skills from skills.csv
    python CODE/run_e3.py skills_llm.csv    # skills from the LLM extractor
Writes results.md and per_resume.csv in EVALUATIONS/e3 (results_skills_llm.md and so on for another file).
"""
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import pipeline

OUT = pipeline.SLICE.parent.parent / "EVALUATIONS" / "e3"
K, RESTART, STEPS, SEED = 10, 0.15, 20, 573
ARMS = ["popularity", "advisor", "graph"]


def load(file):
    """Returns each resume's skills, and the document x skill matrix of the graph."""
    rows = pipeline.skill_rows(file)
    skills = sorted({r["skill"] for r in rows})
    col = {s: i for i, s in enumerate(skills)}
    docs = sorted({(r["doc_type"], r["doc_id"]) for r in rows})
    row = {d: i for i, d in enumerate(docs)}
    pairs = {(row[(r["doc_type"], r["doc_id"])], col[r["skill"]]) for r in rows}
    i, j = zip(*pairs)
    A = csr_matrix((np.ones(len(pairs)), (i, j)), shape=(len(docs), len(skills)))
    resumes = {}
    for r in rows:
        if r["doc_type"] == "resume":
            resumes.setdefault(r["doc_id"], set()).add(r["skill"])
    job_counts = Counter(r["skill"] for r in rows if r["doc_type"] == "job")
    return resumes, A, skills, col, row, job_counts


def graph_advice(have, resume_row, A, skills, col):
    """Personalized PageRank from the skills in `have`, with the test resume's row removed."""
    own = A[resume_row].toarray().ravel()
    skill_deg = np.maximum(np.asarray(A.sum(0)).ravel() - own, 1)
    doc_deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    start = np.zeros(len(skills))
    start[[col[s] for s in have]] = 1 / len(have)
    p = start.copy()
    for _ in range(STEPS):
        d = A @ (p / skill_deg)
        d[resume_row] = 0
        p = RESTART * start + (1 - RESTART) * (A.T @ (d / doc_deg))
    ranked = [skills[i] for i in np.argsort(-p, kind="stable") if skills[i] not in have and p[i] > 0]
    return ranked[:K]


def bootstrap(values):
    rng = np.random.default_rng(SEED)
    means = [rng.choice(values, len(values)).mean() for _ in range(2000)]
    return f"{np.mean(values):.2f} ({np.percentile(means, 2.5):.2f} to {np.percentile(means, 97.5):.2f})"


if __name__ == "__main__":
    file = sys.argv[1] if len(sys.argv) > 1 else "skills.csv"
    pipeline.SKILLS_FILE = file
    resumes, A, skills, col, row, job_counts = load(file)
    popular = [s for s, _ in job_counts.most_common()]
    hot = {r["skill"] for r in pipeline.read("skills_vocab.csv") if r["hot"] == "Y"}

    def advise(arm, have, rid):
        if arm == "popularity":
            return [s for s in popular if s not in have][:K]
        if arm == "advisor":
            return [s["skill"] for s in pipeline.skills_to_learn(have, k=K)]
        return graph_advice(have, row[("resume", rid)], A, skills, col)

    # A hidden skill can only come back if some other document has it, and a long list of
    # rare skills makes the test harder. So recovery is reported for all skills and for the
    # skills on the O*NET list alone, which are the same kind of target in every skills file.
    onet = {r["skill"] for r in pipeline.read("skills_vocab.csv")}
    docs_with = dict(zip(skills, np.asarray(A.sum(0)).ravel()))
    trials, hot_share = [], {arm: [] for arm in ARMS}
    for rid, have in sorted(resumes.items()):
        for arm in ARMS:
            hot_share[arm].append(np.mean([s in hot for s in advise(arm, have, rid)] or [0]))
        if len(have) < 3:
            continue
        for hidden in sorted(have):
            trials.append({"resume_id": rid, "hidden": hidden, "on_onet_list": hidden in onet,
                           "in_another_document": docs_with[hidden] > 1,
                           **{arm: hidden in advise(arm, have - {hidden}, rid) for arm in ARMS}})

    trials = pd.DataFrame(trials)
    suffix = "" if file == "skills.csv" else "_" + file.removesuffix(".csv")
    OUT.mkdir(parents=True, exist_ok=True)
    trials.groupby("resume_id")[ARMS].mean().assign(skills=trials.groupby("resume_id").size()).to_csv(OUT / f"per_resume{suffix}.csv")

    def rows(subset):
        per = subset.groupby("resume_id")[ARMS].mean()  # one number per resume, then the mean over resumes
        return [f"| {arm} | {bootstrap(per[arm].to_numpy())} | {np.mean(hot_share[arm]):.2f} |" for arm in ARMS], len(per)

    head = [f"| Method | Recovery in top {K} (95% CI) | O*NET hot share of the top {K} |", "|---|---|---|"]
    all_rows, n_all = rows(trials)
    onet_rows, n_onet = rows(trials[trials.on_onet_list])
    lines = ["# E3 results: skill advice, hidden-skill test", "",
             f"Skills from `{file}`: {len(skills)} distinct skills in the graph. {n_all} resumes with at least 3 skills, "
             f"{len(trials)} hidden-skill trials (each skill of each resume hidden once).",
             f"Recovered means the hidden skill is in the method's top {K}. The deck's target is 0.70.", "",
             f"{trials.in_another_document.mean():.0%} of the hidden skills appear in some other document. "
             "The rest cannot be recovered by any method.", "",
             "### All skills", ""] + head + all_rows + ["",
             f"### Only skills on the O*NET list ({int(trials.on_onet_list.sum())} trials, {n_onet} resumes)", "",
             "These are the same kind of target in every skills file, so this table is the one to compare across files.", ""] + head + onet_rows
    lines += ["", f"For scale: {len(hot & set(skills))} of the {len(skills)} skills in the graph are O*NET hot technologies "
              f"({len(hot & set(skills)) / len(skills):.2f}).",
              "The graph method here is personalized PageRank. It stands in for the trained graph model (LightGCN), which is not built yet."]
    (OUT / f"results{suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

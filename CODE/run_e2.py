"""E2: matching. Ranks the 2,000 postings for each E2 resume with four methods
and scores the top 10 with P@5 and nDCG@10, with 95% bootstrap intervals.

    tfidf   TF-IDF and cosine similarity on the full texts (classical baseline)
    sbert   Sentence-BERT all-MiniLM-L6-v2, texts cut into chunks and averaged
    skills  overlap of extracted skill sets (pipeline.match_jobs)
    hybrid  0.5 * sbert + 0.5 * skills, both min-max scaled per resume

Relevance comes from two sources:
    judged  the sheet EVALUATIONS/e2/judging_sheet.csv, filled in by two judges
            (1 = a reasonable fit, 0 = not). When they disagree, the O*NET rule decides.
    onet    automatic: the resume title and the job title map to the same O*NET
            group. Strict = same broad occupation (first six characters of the code,
            such as 15-125); lenient = same minor group (first five, such as 15-12).

Run from the repo root:
    python CODE/run_e2.py                   # skills from skills.csv, writes results.md
    python CODE/run_e2.py skills_llm.csv    # skills from the LLM extractor, writes results_skills_llm.md
The tfidf and sbert arms read only the texts, so they are the same in both runs.

judging_sheet.csv is one sheet for every run. A run adds the pairs it has not seen, with empty
judge columns, and never removes a row or a judgment. A pair nobody has judged counts as "not a
fit", so an arm with unjudged pairs is scored too low until they are judged; the tables say how
many of each arm's top 10 are judged.
"""
import random
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pipeline

OUT = pipeline.SLICE.parent.parent / "EVALUATIONS" / "e2"
ONET_TITLES = pipeline.SLICE.parent / "raw" / "onet" / "job_titles.csv"
JUDGES = ["judge_rishi", "judge_basil"]
METHODS = ["tfidf", "sbert", "skills", "hybrid"]
ALPHA, K, SEED = 0.5, 10, 573


def minmax(x):
    return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)


def chunks(text, words=180):
    # all-MiniLM-L6-v2 reads at most 256 tokens, so long texts are cut into pieces of about 180 words.
    w = text.split()
    return [" ".join(w[i:i + words]) for i in range(0, max(len(w), 1), words)]


def sbert_vectors(model, texts):
    pieces = [chunks(t) for t in texts]
    flat = model.encode([p for ps in pieces for p in ps], batch_size=64, normalize_embeddings=True)
    out, i = [], 0
    for ps in pieces:
        v = flat[i:i + len(ps)].mean(0)
        out.append(v / np.linalg.norm(v))
        i += len(ps)
    return np.array(out)


def skills_by_doc(file):
    have = {}
    for row in pipeline.skill_rows(file):
        have.setdefault(row["doc_id"], set()).add(row["skill"])
    return have


def rank_all(resumes, jobs, have):
    """Returns {method: {resume_id: [job_id, ...] ranked best first}}."""
    from sentence_transformers import SentenceTransformer

    texts = [r["text"] for r in resumes]
    descs = [f'{j["title"]}. {j["description"]}' for j in jobs]
    tf = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit(descs + texts)
    scores = {"tfidf": cosine_similarity(tf.transform(texts), tf.transform(descs))}
    model = SentenceTransformer("all-MiniLM-L6-v2")
    scores["sbert"] = sbert_vectors(model, texts) @ sbert_vectors(model, descs).T

    job_index = {j["job_id"]: i for i, j in enumerate(jobs)}
    overlap = np.zeros((len(resumes), len(jobs)))
    for a, r in enumerate(resumes):
        for m in pipeline.match_jobs(have.get(r["resume_id"], set()), k=len(jobs)):
            overlap[a, job_index[m["job_id"]]] = m["score"]
    scores["skills"] = overlap
    scores["hybrid"] = np.array([ALPHA * minmax(s) + (1 - ALPHA) * minmax(o) for s, o in zip(scores["sbert"], overlap)])

    ids = [j["job_id"] for j in jobs]
    ranked = {}
    for m, s in scores.items():
        # skill overlap gives many ties at zero; ties break by posting order, which is random
        ranked[m] = {r["resume_id"]: [ids[i] for i in np.argsort(-s[a], kind="stable")[:K]] for a, r in enumerate(resumes)}
    return ranked


def onet_codes(titles):
    """Maps each title to the O*NET code of its closest lay job title (TF-IDF on
    words and character trigrams). Returns codes and the matched O*NET titles."""
    onet = pd.read_csv(ONET_TITLES, dtype=str)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True).fit(onet["Job Title"])
    sims = cosine_similarity(vec.transform([t.lower() for t in titles]), vec.transform(onet["Job Title"].str.lower()))
    best = sims.argmax(1)
    return list(onet["O*NET-SOC Code"].iloc[best]), list(onet["Job Title"].iloc[best])


def title_of(resume):
    return re.split(r"\s{2,}", resume["text"].strip())[0][:80]


def metrics(ranked, rel):
    """P@5 and nDCG@10 for one resume. rel maps job_id to 0 or 1; n_rel is all
    the relevant jobs known for that resume (for the ideal ranking)."""
    gains = [rel.get(j, 0) for j in ranked[:K]]
    n_rel = sum(rel.values())
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(n_rel, K)))
    return sum(gains[:5]) / 5, (dcg / idcg if idcg else 0.0)


def bootstrap(values, n=2000):
    rng = np.random.default_rng(SEED)
    means = [rng.choice(values, len(values)).mean() for _ in range(n)]
    return np.mean(values), np.percentile(means, 2.5), np.percentile(means, 97.5)


def table(ranked, labels, title, judged=None):
    """judged: the set of (resume, job) pairs somebody has judged. When given, a column
    says how much of each method's top 10 is judged."""
    extra = " Top-10 pairs judged |" if judged is not None else ""
    lines = [f"### {title}", "", "| Method | P@5 (95% CI) | nDCG@10 (95% CI) |" + extra, "|---|---|---|" + ("---|" if extra else "")]
    for m in METHODS:
        per = [metrics(ranked[m][rid], labels.get(rid, {})) for rid in ranked[m]]
        p, n = bootstrap(np.array([x[0] for x in per])), bootstrap(np.array([x[1] for x in per]))
        row = f"| {m} | {p[0]:.2f} ({p[1]:.2f} to {p[2]:.2f}) | {n[0]:.2f} ({n[1]:.2f} to {n[2]:.2f}) |"
        if judged is not None:
            pairs = [(rid, j) for rid, js in ranked[m].items() for j in js]
            row += f" {sum(pair in judged for pair in pairs)} of {len(pairs)} |"
        lines.append(row)
    return lines


if __name__ == "__main__":
    chosen = set(pd.read_csv(OUT / "resumes.csv", dtype=str).resume_id)
    file = sys.argv[1] if len(sys.argv) > 1 else "skills.csv"
    pipeline.SKILLS_FILE = file  # the skills and hybrid arms match on this file's skills
    resumes = [r for r in pipeline.read("resumes.csv") if r["resume_id"] in chosen]
    jobs = pipeline.read("postings.csv")
    have = skills_by_doc(file)
    ranked = rank_all(resumes, jobs, have)

    # The judging sheet: every pooled (resume, job) pair once, with no method shown.
    sheet_path = OUT / "judging_sheet.csv"
    by_job = {j["job_id"]: j for j in jobs}
    titles = {r["resume_id"]: title_of(r) for r in resumes}
    sheet = pd.read_csv(sheet_path, dtype=str).fillna("") if sheet_path.exists() else pd.DataFrame()
    seen = set(zip(sheet.resume_id, sheet.job_id)) if len(sheet) else set()
    pool = {(rid, j) for m in METHODS for rid, js in ranked[m].items() for j in js}
    new = sorted(pool - seen)
    random.Random(SEED).shuffle(new)
    added = pd.DataFrame([{"resume_id": rid, "resume_title": titles[rid],
                           "resume_skills": ", ".join(sorted(have.get(rid, ()))[:25]),
                           "job_id": j, "job_title": by_job[j]["title"],
                           "company": by_job[j]["company_name"], "location": by_job[j]["location"],
                           "job_text": " ".join(by_job[j]["description"].split())[:400],
                           **{c: "" for c in JUDGES}} for rid, j in new])
    sheet = pd.concat([sheet, added], ignore_index=True).fillna("")
    sheet.to_csv(sheet_path, index=False)

    lines = [f"# E2 results: matching, skills from `{file}`", "",
             f"{len(resumes)} resumes (10 IT, 10 Engineering, no gold documents), ranked against "
             f"{len(jobs)} postings. Top {K} per method, {len(pool)} pooled pairs in this run.",
             f"Hybrid uses alpha = {ALPHA}, not tuned yet. The graph arm (LightGCN) is not built yet; "
             "skill overlap stands in for the graph score.", ""]

    onet = {}
    if ONET_TITLES.exists():
        rids = list(titles)
        rcodes, rmatch = onet_codes([titles[r] for r in rids])
        jcodes, _ = onet_codes([j["title"] for j in jobs])
        def same(n):
            return {rid: {j["job_id"]: int(jc[:n] == code[:n]) for j, jc in zip(jobs, jcodes)}
                    for rid, code in zip(rids, rcodes)}
        onet, lenient = same(6), same(5)
        pd.DataFrame({"resume_id": rids, "resume_title": [titles[r] for r in rids], "onet_title": rmatch,
                      "onet_code": rcodes,
                      "matching_postings": [sum(onet[r].values()) for r in rids]}).to_csv(OUT / "onet_labels.csv", index=False)
        lines += table(ranked, onet, "Automatic relevance, strict: same O*NET broad occupation (preliminary)")
        lines += [""] + table(ranked, lenient, "Automatic relevance, lenient: same O*NET minor group (preliminary)")
        lines += ["", "How each resume title was mapped is in `onet_labels.csv`. Titles such as "
                  "\"Objective\" or \"ENGINEERING TECH\" map poorly, so treat these numbers as a rough first look.", ""]

    # One judge on a pair: their label. Two judges: their label if they agree, else O*NET decides.
    labels, judged, judges = {}, set(), set()
    for _, row in sheet.iterrows():
        votes = {j: int(row[j]) for j in JUDGES if row[j] != ""}
        if votes:
            judges |= set(votes)
            judged.add((row.resume_id, row.job_id))
            label = next(iter(votes.values())) if len(set(votes.values())) == 1 else onet.get(row.resume_id, {}).get(row.job_id, 0)
            labels.setdefault(row.resume_id, {})[row.job_id] = label
    if judged:
        who = " and ".join(sorted(j.removeprefix("judge_") for j in judges))
        lines += table(ranked, labels, f"Judged relevance (judged by {who}; an unjudged pair counts as not a fit)", judged)
        unjudged = len(pool - judged)
        lines += ["", f"{len(pool) - unjudged} of this run's {len(pool)} pooled pairs are judged."
                  + (f" {unjudged} are new and still need a judgment (empty rows in `judging_sheet.csv`), "
                     "so arms with unjudged pairs are scored too low for now." if unjudged else "")]
    else:
        lines += ["### Judged relevance", "", "Not scored yet: nobody has judged any pair."]

    name = "results.md" if file == "skills.csv" else f"results_{file.removesuffix('.csv')}.md"
    (OUT / name).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

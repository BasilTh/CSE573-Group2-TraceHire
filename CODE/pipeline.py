"""The working parts behind the TraceHire page.

Each function is a simple first version with fixed inputs and outputs, so the
inside can be replaced (see the TODO) while the page keeps working.

Run this file to check the functions:

    python CODE/pipeline.py
"""
import os
import re
from collections import Counter
from functools import cache
from pathlib import Path

import pandas as pd

SLICE = Path(__file__).parent.parent / "DATA" / "slice"

# What people write -> the name O*NET uses for it. Add more as you find them.
SHORT_FORMS = {
    "SQL": "Structured query language SQL", "AWS": "Amazon Web Services AWS software",
    "HTML": "Hypertext markup language HTML", "XML": "Extensible markup language XML",
    "CSS": "Cascading style sheets CSS", "Java": "Oracle Java", "Excel": "Microsoft Excel",
    "Azure": "Microsoft Azure software", "MATLAB": "The MathWorks MATLAB", "JIRA": "Atlassian JIRA",
    "Spark": "Apache Spark", "Hadoop": "Apache Hadoop", "Kafka": "Apache Kafka", "Hive": "Apache Hive",
    "Airflow": "Apache Airflow", "Maven": "Apache Maven", "Cassandra": "Apache Cassandra",
    "Photoshop": "Adobe Photoshop", "AutoCAD": "Autodesk AutoCAD", "Revit": "Autodesk Revit",
    "SolidWorks": "Dassault Systemes SolidWorks", "CATIA": "Dassault Systemes CATIA",
    "LabVIEW": "National Instruments LabVIEW", "Simulink": "MathWorks Simulink",
    "PowerPoint": "Microsoft PowerPoint", "SharePoint": "Microsoft SharePoint", "Visio": "Microsoft Visio",
    "SQL Server": "Microsoft SQL Server", "Power BI": "Microsoft Power BI",
    "Active Directory": "Microsoft Active Directory", "Microsoft Office": "Microsoft Office software",
    "Windows Server": "Microsoft Windows Server", "ASP.NET": "Microsoft ASP.NET",
    "PowerShell": "Microsoft PowerShell", "SAP": "SAP software", "Salesforce": "Salesforce software",
    "Google Cloud": "Google Cloud software", "Angular": "Google Angular", "Android": "Google Android",
    "Terraform": "IBM Terraform", "Ansible": "Ansible software", "Confluence": "Atlassian Confluence",
    "macOS": "Apple macOS", "iOS": "Apple iOS",
}
IGNORE = {"reduce", "act", "analyze", "ada"}  # O*NET names that also mean something else in normal text
WORD = re.compile(r"[A-Za-z0-9+#]+(?:\.[A-Za-z0-9+#]+)*")  # keeps C++, C# and Node.js in one piece


@cache
def read(name):
    return pd.read_csv(SLICE / name, dtype=str).fillna("").to_dict("records")


@cache
def skill_names():
    names = {r["skill"].lower(): r["skill"] for r in read("skills_vocab.csv")}
    names.update({short.lower(): full for short, full in SHORT_FORMS.items()})
    # One and two letter names ("C", "R", "Go") match normal text far too often.
    return {k: v for k, v in names.items() if k not in IGNORE and (len(k) > 2 or k == "c#")}


def split_sentences(text):
    # Resumes have few full stops, so line breaks and wide gaps also end a sentence.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+|\s{3,}", text) if s.strip()]


def extract_skills(text, doc_type="resume", extractor=None):
    """Returns one row per skill per sentence. Every row carries the sentence it
    came from and the exact words (span) that named the skill.

    extractor is "lexicon" (fast word matching) or "llm" (Llama 3.1 8B with the
    provenance check, see CODE/extraction). The default comes from the
    TRACEHIRE_EXTRACTOR environment variable and is "lexicon" when it is not set.
    """
    extractor = extractor or os.getenv("TRACEHIRE_EXTRACTOR", "lexicon")
    if extractor == "llm":
        from extraction.service import extract_rows
        return extract_rows(text, doc_type)
    return extract_skills_lexicon(text)


def extract_skills_lexicon(text):
    """The word-matching extractor: the E1 lexicon baseline."""
    # TODO: swap the O*NET names for Lightcast names once we have access.
    names, rows = skill_names(), []
    for sent_id, sentence in enumerate(split_sentences(text)):
        words, seen = list(WORD.finditer(sentence)), set()
        for i in range(len(words)):
            for j in range(i, min(i + 4, len(words))):  # try spans of 1 to 4 words
                span = sentence[words[i].start():words[j].end()]
                skill = names.get(span.lower())
                # "excel at teamwork" is not Excel: a skill name has a capital letter somewhere.
                if skill and span != span.lower() and skill not in seen:
                    seen.add(skill)
                    rows.append({"sent_id": sent_id, "sentence": sentence, "span": span, "skill": skill,
                                 "vocab": "onet", "extractor": "lexicon", "confidence": 1.0})
    return rows


# The skills file that matching and advice read: the LLM extractor's skills. Reading
# the file here keeps the page working without a database connection.
SKILLS_FILE = os.getenv("TRACEHIRE_SKILLS_FILE", "skills_llm.csv")


@cache
def _skill_names(file):
    """For a skills file with a canonical_id column: the most used name of each ID."""
    names = {}
    for r in read(file):
        if r.get("canonical_id"):
            names.setdefault(r["canonical_id"], Counter())[r["skill"]] += 1
    return {cid: counts.most_common(1)[0][0] for cid, counts in names.items()}


def same_names(rows, file=None):
    """Gives each row the name its skill goes by in the skills file, so "Agile" and
    "agile methodology" with one canonical_id count as one skill. Certifications
    (skill_class = cert) are not skills and are left out."""
    names = _skill_names(file or SKILLS_FILE)
    return [{**r, "skill": names.get(r.get("canonical_id"), r["skill"])}
            for r in rows if r.get("skill_class") != "cert"]


@cache
def skill_rows(file=None):
    """The rows of a skills file that count as skills, each under its one name."""
    return same_names(read(file or SKILLS_FILE), file)


@cache
def _job_skills(file):
    skills = {}
    for r in skill_rows(file):
        if r["doc_type"] == "job":
            skills.setdefault(r["doc_id"], set()).add(r["skill"])
    return skills


def job_skills():
    return _job_skills(SKILLS_FILE)


@cache
def _demand(file):
    """How many postings ask for each skill."""
    return Counter(s for wanted in _job_skills(file).values() for s in wanted)


def evidence(doc_id, skill, file=None):
    """Returns the source sentences that tie a resume or a job to a skill.
    file picks the skills file to read; the default is SKILLS_FILE."""
    # TODO: read this from the skill edge in the Neo4j knowledge graph.
    return [r["sentence"] for r in skill_rows(file or SKILLS_FILE) if r["doc_id"] == doc_id and r["skill"] == skill]


def match_jobs(skills, k=10):
    """Returns the k closest postings. Each one has a score and the skills the
    person shares with it and is missing for it, the most asked-for missing skill first."""
    # TODO: this is plain skill overlap. Add TF-IDF with cosine similarity
    # (the E2 baseline) and Sentence-BERT (all-MiniLM-L6-v2) on the full text, then
    # the hybrid score with the graph. Keep "score", "shared" and "missing".
    skills, jobs, demand = set(skills), {r["job_id"]: r for r in read("postings.csv")}, _demand(SKILLS_FILE)
    scored = [(len(skills & wanted) / len(skills | wanted), job_id, wanted)
              for job_id, wanted in job_skills().items() if skills & wanted]
    return [{**jobs[job_id], "score": round(score, 3), "shared": sorted(skills & wanted),
             "missing": sorted(wanted - skills, key=lambda s: (-demand[s], s))} for score, job_id, wanted in sorted(scored, reverse=True)[:k]]


def skills_to_learn(skills, k=10):
    """Returns the missing skills that the 100 closest postings ask for most,
    with the O*NET hot technology and in demand flags."""
    # TODO: this count is the E3 baseline. Add the role and city filters
    # and the salary per skill, then the graph recommendation.
    flags = {r["skill"]: r for r in read("skills_vocab.csv")}
    counts = Counter(s for job in match_jobs(skills, k=100) for s in job["missing"])
    # Skills the LLM found that are not in the O*NET list (vocab = none) have no flags.
    return [{"skill": s, "postings": n, "hot": flags.get(s, {}).get("hot", ""),
             "in_demand": flags.get(s, {}).get("in_demand", "")}
            for s, n in counts.most_common(k)]


def company_profile(company_id):
    """Returns the company's row from the dataset, how many of our postings it
    has, and the skills those postings ask for most."""
    # TODO: add live web search for what the dataset does not know.
    company = next((r for r in read("companies.csv") if r["company_id"] == company_id), {})
    posted = [r["job_id"] for r in read("postings.csv") if r["company_id"] == company_id]
    wanted = job_skills()
    hiring = Counter(s for job_id in posted for s in wanted.get(job_id, ()))
    return {**company, "postings": len(posted), "hiring_for": [s for s, _ in hiring.most_common(10)]}


def answer_baseline(question, resume_text):
    """The plain LLM: the question and the resume, nothing else (CODE/agents.py)."""
    import agents  # imported here: agents imports this module
    return agents.answer_baseline(question, resume_text)


def answer_tracehire(question, resume_text):
    """The LangGraph orchestrator: routes the question to one agent, which answers
    only from cited sources (CODE/agents.py)."""
    import agents
    return agents.answer_tracehire(question, resume_text)


if __name__ == "__main__":
    rows = extract_skills("Built data pipelines in Python and Spark on AWS.   I excel at teamwork.")
    assert {r["skill"] for r in rows} == {"Python", "Apache Spark", "Amazon Web Services AWS software"}, rows
    assert all(r["span"] in r["sentence"] for r in rows)  # no source sentence, no skill
    assert set(SHORT_FORMS.values()) <= {r["skill"] for r in read("skills_vocab.csv")}

    docs = [("resume", r["resume_id"], r["text"]) for r in read("resumes.csv")]
    docs += [("job", r["job_id"], r["description"]) for r in read("postings.csv")]
    found = [{"doc_id": doc_id, "doc_type": doc_type, **row}
             for doc_type, doc_id, text in docs[:20] for row in extract_skills(text)]
    assert all(r["span"] in r["sentence"] for r in found)

    top = match_jobs({"Python", "Apache Spark"}, k=3)
    assert top and all(set(j["shared"]) <= {"Python", "Apache Spark"} for j in top)
    job = top[0]
    assert evidence(job["job_id"], job["shared"][0]) and not evidence("no such id", "Python")
    print("checks passed")

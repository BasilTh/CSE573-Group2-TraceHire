"""TraceHire's agents: a LangGraph orchestrator that sends each question to one agent.

    read_resume -> route -> one of: match, advice, outreach, company, resume, other -> check

Every agent answers only from numbered sources it is handed (resume sentences S,
job postings J, skill counts A, company facts C) and must cite them like [S1].
The check step drops citations that point nowhere and lists every cited source
in full under the answer, so each claim can be traced back to its evidence.
answer_baseline is the plain LLM with no sources, for the side-by-side screen (E4).

The LLM is the ASU Research Computing API (see .env.example). The model is
llama4-scout-17b unless TRACEHIRE_MODEL says otherwise.
"""
import os
import re
from functools import cache
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

import pipeline

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
MODEL = os.getenv("TRACEHIRE_MODEL", "llama4-scout-17b")
RESUME_LIMIT = 12000  # characters of resume text the baseline sees (about 3,000 tokens)

ROUTES = {
    "match": "which jobs fit the person, or how well they fit a kind of role",
    "advice": "which skills to learn next to get more or better jobs",
    "outreach": "drafting a message to a company, recruiter or hiring manager",
    "company": "facts about a company: what it does, where it is, what it hires for",
    "resume": "what skills or experience the resume shows, or how to improve the resume itself",
    "other": "a greeting, thanks or small talk, or a question that is not about jobs, skills, "
             "companies, the resume or a message to send",
}
HELLO = ("Hi! I'm TraceHire. Ask me which jobs fit your resume, which skills to learn next, how to improve "
         "your resume, about a company in our data, or for a message to a recruiter.")
OFF_TOPIC = ("I can't answer that: it is outside our job and resume data. I can tell you which jobs fit your "
             "resume, which skills to learn next, how to improve your resume, about a company in our data, "
             "or draft a message to a recruiter.")
GREETING = re.compile(r"\W*(hi|hello|hey|yo|thanks|thank you|good (morning|afternoon|evening))\b[\w\s]{0,20}\W*", re.I)
# One citation group: [S1], [S1, S8], [A1-A8], [S2 and S5].
GROUP = re.compile(r"\[\s*([SJAC]\d+(?:\s*(?:,|-|–|and)\s*[SJAC]?\d+)*)\s*\]")

RULES = """You are TraceHire, a career coach. Use ONLY the numbered sources below.
Speak to the person directly as "you" and "your", in plain, friendly words.
After every sentence that states a fact about the person, a job, a company or a skill,
cite its sources at the END of the sentence, each in its own brackets, for example [S2] or
[S2][J1]. Never write ranges such as [S1-S5]. A citation is never a word in the sentence:
always write the job title, company or skill by name, never "[J1] is a good fit".
Use everyday skill names (say SQL, AWS, Java), not long catalogue names.
Never say the person has a skill unless an S source shows it. If the sources do not answer
the question, say so in a plain sentence with no brackets. Square brackets are ONLY for source
ids: never write placeholders or notes in brackets, such as [Your Name] or [no source].
Do not use outside knowledge. No closing remarks. Keep it under 150 words."""


class State(TypedDict, total=False):
    question: str
    resume_text: str
    rows: list
    route: str
    sources: dict
    answer: str
    text: str      # the answer alone, with only its valid citations
    cited: list    # the sources it cites, in the order they first appear
    dropped: list  # citations that pointed to no source and were removed


@cache
def llm():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("No OPENAI_API_KEY. Copy .env.example to .env and add your Voyager key.")
    return ChatOpenAI(model=MODEL, api_key=key, temperature=0, max_tokens=900, timeout=120, max_retries=2,
                      base_url=os.getenv("OPENAI_BASE_URL", "https://openai.rc.asu.edu/v1"))


def ask(system, user):
    return llm().invoke([("system", system), ("user", user)]).content.strip()


def answer_baseline(question, resume_text):
    """The plain LLM: the question and the resume, no sources, no checks."""
    return ask("You are a career coach. Answer the user's question about their resume.",
               f"My resume:\n{resume_text[:RESUME_LIMIT]}\n\nQuestion: {question}")


# ---------------------------------------------------------------- sources

def sentence_sources(rows, only=None):
    """S sources: one per resume sentence that names a skill, with the skills it names."""
    by_sentence = {}
    for r in rows:
        if only is None or r["skill"] in only:
            by_sentence.setdefault(r["sentence"], []).append(r["skill"])
    return {f"S{i}": f'"{s[:300]}" (names: {", ".join(dict.fromkeys(skills))})'
            for i, (s, skills) in enumerate(by_sentence.items(), 1)}


def job_source(job, text_chars=0):
    company = job.get("company_name") or "unknown company"
    text = f'{job["title"]} at {company}, {job["location"]}. Shares with the person: {", ".join(job["shared"]) or "none"}.'
    text += f' Asks for that the person lacks: {", ".join(job["missing"][:8]) or "none"}.'
    if text_chars:
        text += f' Posting text: "{" ".join(job["description"].split())[:text_chars]}"'
    return text


def company_for(question, jobs):
    """The company the question names, if any, else the top match's company."""
    q = question.lower()
    for job in jobs:
        if job.get("company_name") and job["company_name"].lower() in q:
            return job["company_id"], job
    for c in pipeline.read("companies.csv"):
        if len(c["name"]) > 3 and c["name"].lower() in q:
            return c["company_id"], None
    top = next((j for j in jobs if j.get("company_id")), None)
    return (top["company_id"], top) if top else (None, None)


def company_source(company_id):
    c = pipeline.company_profile(company_id)
    if not c.get("name"):
        return None
    place = ", ".join(x for x in (c.get("city"), c.get("state"), c.get("country")) if x)
    about = " ".join(c.get("description", "").split())[:600]
    return (f'{c["name"]} ({place}). About: "{about}". Postings in our data: {c["postings"]}. '
            f'Those postings ask most for: {", ".join(c["hiring_for"][:8]) or "no skills found"}.')


# ---------------------------------------------------------------- graph nodes

def read_resume(state):
    rows = state.get("rows") or pipeline.extract_skills(state["resume_text"])
    return {"rows": rows}


def route(state):
    menu = "\n".join(f"{name}: {what}" for name, what in ROUTES.items())
    reply = ask(f"Pick the one agent that should answer the question. Reply with its name only.\n{menu}",
                state["question"]).lower()
    picked = next((name for name in ROUTES if name in reply), None)
    if not picked:  # the LLM did not name an agent: fall back on keywords
        q = state["question"].lower()
        picked = ("outreach" if any(w in q for w in ("message", "email", "reach out", "draft")) else
                  "resume" if "resume" in q else
                  "advice" if any(w in q for w in ("learn", "improve", "next")) else
                  "match" if any(w in q for w in ("job", "role", "fit", "match")) else "resume")
    return {"route": picked}


def skills_of(state):
    return {r["skill"] for r in state["rows"]}


def match_agent(state):
    jobs = pipeline.match_jobs(skills_of(state), k=8)
    sources = {f"J{i}": job_source(j) for i, j in enumerate(jobs, 1)}
    sources |= sentence_sources(state["rows"], only={s for j in jobs for s in j["shared"]})
    task = ("First say what kinds of roles fit the person: name two or three role types, taken from the "
            "job titles in the J sources, and say in one sentence each why, using the skills they share. "
            "Then name the two or three postings that fit best, with the company, and one skill each "
            "asks for that the person lacks. These postings are from our stored data, not live listings.")
    return answer(state, sources, task)


def advice_agent(state):
    learn = pipeline.skills_to_learn(skills_of(state), k=8)
    sources = {f"A{i}": f'{s["skill"]}: asked for in {s["postings"]} of the 100 postings closest to this resume'
                        f'{" (an O*NET hot technology)" if s["hot"] == "Y" else ""}'
               for i, s in enumerate(learn, 1)}
    sources |= sentence_sources(state["rows"])
    task = ("Recommend three skills to learn next. Write one short opening sentence, then a bulleted list "
            "with one bullet per skill: the skill name in bold, then one sentence saying how many of the "
            "closest postings ask for it (A sources) and which current skill it builds on (S sources).")
    return answer(state, sources, task)


def outreach_agent(state):
    jobs = pipeline.match_jobs(skills_of(state), k=20)
    company_id, job = company_for(state["question"], jobs)
    sources = {}
    if job:
        sources["J1"] = job_source(job, text_chars=500)
    if company_id and (c := company_source(company_id)):
        sources["C1"] = c
    sources |= sentence_sources(state["rows"], only=set(job["shared"]) if job else None)
    task = ("Draft a short LinkedIn message (under 120 words) from the person to a recruiter at the "
            "company about the role. Mention only experience shown in S sources and facts about the role "
            "or company from J or C sources, and cite each one. The person sends it themselves, so write "
            "it ready to send: open with \"Hi,\" and end with \"Best regards\" alone. No names, no blanks "
            "to fill in.")
    return answer(state, sources, task)


def company_agent(state):
    jobs = pipeline.match_jobs(skills_of(state), k=20)
    company_id, job = company_for(state["question"], jobs)
    sources = {}
    if company_id and (c := company_source(company_id)):
        sources["C1"] = c
    if job:
        sources["J1"] = job_source(job)
    task = "Answer the question about the company. Our data has only what the dataset says about it."
    return answer(state, sources, task)


def resume_agent(state):
    task = ("Answer from the resume sentences. If the person asks how to improve the resume, point to "
            "specific sentences and say how to make each one stronger.")
    return answer(state, sentence_sources(state["rows"]), task)


def other_agent(state):
    """Greetings and off-topic questions get a fixed reply: no LLM call, no sources."""
    return {"sources": {}, "answer": HELLO if GREETING.fullmatch(state["question"]) else OFF_TOPIC}


def answer(state, sources, task):
    if not sources:
        return {"sources": {}, "answer": "I could not find any evidence in our data for this question."}
    listing = "\n".join(f"[{k}] {v}" for k, v in sources.items())
    text = ask(f"{RULES}\n\n{task}", f"Sources:\n{listing}\n\nQuestion: {state['question']}")
    return {"sources": sources, "answer": text}


def expand(group):
    """'S1, S8' -> [S1, S8]; 'A1-A3' -> [A1, A2, A3]."""
    ids = []
    for part in re.split(r"\s*(?:,|\band\b)\s*", group.strip()):
        if m := re.fullmatch(r"([SJAC])(\d+)\s*[-–]\s*[SJAC]?(\d+)", part):
            ids += [f"{m[1]}{i}" for i in range(int(m[2]), int(m[3]) + 1)]
        elif re.fullmatch(r"[SJAC]\d+", part):
            ids.append(part)
    return ids


def check(state):
    """The provenance check on the answer: each citation is split into single
    sources, citations to unknown sources are removed, every cited source is listed
    in full, and an answer with no citations is flagged."""
    sources, cited, dropped = state.get("sources", {}), [], set()

    def keep_known(m):
        ids = expand(m.group(1))
        cited.extend(i for i in ids if i in sources)
        dropped.update(i for i in ids if i not in sources)
        return "".join(f"[{i}]" for i in ids if i in sources)

    text = GROUP.sub(keep_known, state["answer"])
    text = re.sub(r" ?\[[^\]\[]*\bsources?\b[^\]\[]*\]", "", text, flags=re.I)  # notes such as [no source]
    cited = list(dict.fromkeys(cited))
    if state["route"] == "other":
        return {"answer": text, "text": text, "cited": [], "dropped": []}
    parts = [text]
    if sources and not cited:
        parts.insert(0, "> **Warning:** this answer cites no sources, so it cannot be checked.")
    if state["route"] == "outreach" and cited:
        clean = re.sub(r"\s*\[[SJAC]\d+\]", "", text).strip()
        parts.append("**Clean copy to send:**\n\n> " + clean.replace("\n", "\n> "))
    if cited:
        parts.append("**Sources**\n" + "\n".join(f"- [{c}] {sources[c]}" for c in cited))
    if dropped:
        parts.append(f"_Removed citations to sources that do not exist: {', '.join(sorted(dropped))}._")
    parts.append(f"_Answered by the {state['route']} agent ({MODEL})._")
    return {"answer": "\n\n".join(parts), "text": text, "cited": cited, "dropped": sorted(dropped)}


AGENTS = {"match": match_agent, "advice": advice_agent, "outreach": outreach_agent,
          "company": company_agent, "resume": resume_agent, "other": other_agent}


@cache
def graph():
    g = StateGraph(State)
    g.add_node("read_resume", read_resume)
    g.add_node("route", route)
    g.add_node("check", check)
    for name, fn in AGENTS.items():
        g.add_node(name, fn)
        g.add_edge(name, "check")
    g.set_entry_point("read_resume")
    g.add_edge("read_resume", "route")
    g.add_conditional_edges("route", lambda s: s["route"], {name: name for name in AGENTS})
    g.add_edge("check", END)
    return g.compile()


def run(question, resume_text, rows=None):
    """Runs the graph and returns the whole final state (answer, route, sources)."""
    return graph().invoke({"question": question, "resume_text": resume_text, "rows": rows or []})


def answer_tracehire(question, resume_text, rows=None):
    """The orchestrator's answer as Markdown. rows: skill rows already extracted
    (for example from skills_llm.csv); without them the resume text is extracted."""
    return run(question, resume_text, rows)["answer"]

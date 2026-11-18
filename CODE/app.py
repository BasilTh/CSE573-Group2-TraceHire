"""The TraceHire page: upload a resume, then chat. Run from the repo root:

    streamlit run CODE/app.py

The answers need OPENAI_API_KEY in .env (copy .env.example).
Add ?test=1 to the address to load one of the dataset's test resumes instead of uploading.
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor
from html import escape

import pandas as pd
import streamlit as st

import agents
import pipeline

# Uploaded resumes are read by the model that made skills_llm.csv, on the team's API,
# unless .env sets these two (agents has loaded .env by now).
os.environ.setdefault("EXTRACTION_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://openai.rc.asu.edu/v1"))
os.environ.setdefault("EXTRACTION_MODEL", agents.MODEL)

DEMO_RESUMES = ["83816738", "26746496", "22890839"]  # listed first in test mode
STARTERS = ["Which jobs fit me best?", "What should I learn next?", "How can I improve my resume?",
            "Tell me about the company in my top match", "Draft a message to a recruiter at my top match"]
AGENT = {"match": "job matching", "advice": "skill advice", "outreach": "outreach",
         "company": "company research", "resume": "resume"}
SOURCE_KIND = {"S": "From your resume", "J": "From job postings", "A": "From counts over the 100 postings closest to you",
               "C": "About the company"}
SOURCE_COUNT = {"S": "lines of your resume", "J": "job postings", "A": "posting counts", "C": "company facts"}
SHORT_NAME = {full: name for name, full in pipeline.SHORT_FORMS.items()}
COMPARE_HEIGHT = 800  # pixels, both cards of a side by side answer

st.set_page_config(page_title="TraceHire", layout="wide")
# The page colours (background, text, buttons) are in .streamlit/config.toml. LOOK holds the
# rest of the palette for our own markup: light is the cream page, dark the pine green one.
LOOKS = {
    "light": {"strong": "#204242", "accent": "#AF5B46", "soft": "rgba(52, 21, 21, 0.68)",
              "line": "#A9BBBC", "track": "rgba(169, 187, 188, 0.45)"},
    "dark": {"strong": "#F9C982", "accent": "#F9C982", "soft": "#A9BBBC",
             "line": "rgba(169, 187, 188, 0.4)", "track": "rgba(242, 239, 225, 0.16)"},
}
LOOK = LOOKS.get(st.get_option("theme.base"), LOOKS["light"])
# No boxes: colour comes from the page, the type and a few marks. A sentence quoted from a
# document is set in a serif face, and the words that named a skill are marked like a highlighter.
st.markdown("""<style>
@import url("https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400&display=swap");
:root { LOOK }
.block-container { max-width: 1320px; padding-top: 3rem; }
.brand { font-size: 3rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.05; color: var(--strong); }
.tagline { color: var(--accent); font-size: 1.1rem; margin-top: 0.3rem; }
[data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2, [data-testid="stChatMessage"] h3 {
  font-size: 1.05rem; letter-spacing: 0; padding: 0.4rem 0 0.1rem; }
/* No bubbles and no icons: the question is a line of large type and the answer sits under it. */
[data-testid="stChatMessage"] { background: transparent; padding: 0.6rem 0 0; }
[data-testid="stChatMessageAvatarCustom"] { display: none; }
[class*="st-key-ask-"] [data-testid="stChatMessage"] p { font-size: 1.35rem; font-weight: 600; color: var(--strong); }
/* Buttons that open something (your skills, an answer's sources) carry the accent. */
[data-testid="stPopover"] button { border-color: var(--accent); color: var(--accent); font-weight: 600; }
/* Every list inside an answer (jobs, skills to learn, evidence) shares one rhythm: a hairline, then the row. */
.evidence, .job, .learn { border-top: 1px solid var(--line); padding: 0.75rem 0; }
.evidence { display: grid; gap: 0.2rem 1.5rem; grid-template-columns: minmax(8rem, 13rem) 1fr; }
.source { padding-bottom: 0.8rem; }
.job-head { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }
.job p { margin: 0.25rem 0 0; }
.skill { font-weight: 600; }
.pill { white-space: nowrap; font-size: 0.85rem; background: #F9C982; color: #341515; border-radius: 99px;
  padding: 0.15rem 0.75rem; }
.count, .kind, .evidence summary { color: var(--soft); font-size: 0.9rem; }
.learn { display: grid; grid-template-columns: minmax(6rem, 11rem) 1fr 6.5rem; gap: 1rem; align-items: center;
  padding: 0.55rem 0; }
.learn.narrow { grid-template-columns: 1fr auto; gap: 0.35rem 1rem; }
.learn.narrow .track { grid-column: 1 / -1; grid-row: 2; }
.track { height: 0.5rem; background: var(--track); border-radius: 99px; overflow: hidden; }
.fill { height: 100%; background: var(--strong); border-radius: 99px; }
.n { text-align: right; color: var(--soft); font-size: 0.9rem; font-variant-numeric: tabular-nums; }
.hot { font-size: 0.75rem; font-weight: 400; color: var(--accent); border: 1px solid var(--accent);
  border-radius: 4px; padding: 0 0.3rem; margin-left: 0.4rem; vertical-align: 0.1em; }
/* The two sides of a compared answer: a name over a rule, and no frame. */
.side { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem;
  border-bottom: 2px solid var(--line); padding-bottom: 0.5rem; }
.side b { font-size: 1.3rem; }
.side span { color: var(--soft); font-size: 0.9rem; }
.side.ours { border-bottom-color: var(--accent); }
.side.ours b { color: var(--accent); }
.quote { font-family: "Newsreader", Georgia, serif; font-size: 1.06rem; line-height: 1.5; margin: 0 0 0.3rem; max-width: 70ch; }
.fact { margin: 0; max-width: 75ch; }
mark { background: #F9C982; color: #341515; padding: 0.05em 0.2em; border-radius: 3px; }
.evidence summary { cursor: pointer; margin-bottom: 0.3rem; }
[data-testid="stPopoverBody"] { min-width: min(44rem, 92vw); }
@media (max-width: 640px) { .evidence { grid-template-columns: 1fr; } .learn { grid-template-columns: 1fr 1fr 5rem; } }
</style>""".replace("LOOK", " ".join(f"--{k}: {v};" for k, v in LOOK.items())), unsafe_allow_html=True)


# ------------------------------------------------------------------ reading the resume

def read_upload(file):
    if file.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(file).pages)
    return file.read().decode("utf-8", errors="replace")


def read_with_llm(text, seconds=60):
    """The LLM extractor on an uploaded resume, run the way skills_llm.csv was made (no
    learned rules). It asks about a few sentences per request and one request takes about
    8 seconds, so the requests go out side by side. Raises when the model has not answered
    every sentence within the time limit."""
    from extraction import extractors, llm, sentences
    failed, pool = len(llm.FAILURES), ThreadPoolExecutor(8)
    try:
        parts = list(pool.map(lambda batch: extractors.extract_final(None, "resume", text, sentences=batch),
                              extractors.batches(sentences.split(text)), timeout=seconds))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if len(llm.FAILURES) > failed:
        raise RuntimeError("the model left some sentences unanswered")
    return [row for part in parts for row in part]


def skills_for(resume_id, text):
    """Skill rows for a resume, and a note for the person when the reading fell short.
    A test resume's rows are already in the skills file. An uploaded resume is read by the
    LLM, and by the word matcher when the model has no key, is down or is too slow."""
    if resume_id:
        rows = [r for r in pipeline.read(pipeline.SKILLS_FILE) if r["doc_id"] == resume_id]
        if rows:
            return rows, ""
    if os.getenv("OPENAI_API_KEY"):
        try:
            return read_with_llm(text), ""
        except Exception:
            pass  # no full answer in time: fall through to the word matcher
    return pipeline.extract_skills(text, extractor="lexicon"), (
        "The language model did not answer in time, so your resume was read with a simple word list. "
        "Some of your skills may be missing. Choose Change resume to try again.")


def load_resume(name, text, resume_id=None):
    with st.spinner("Reading your resume. A long one can take half a minute."):
        rows, note = skills_for(resume_id, text)
    # Certifications (skill_class = cert) are listed apart and are not skills, so they stay
    # out of the matching, the advice and the agents' sources. same_names leaves them out
    # and gives every skill the name the job postings use for it.
    st.session_state["resume"] = {"name": name, "text": text, "note": note,
                                  "rows": pipeline.same_names(rows),
                                  "certs": [r for r in rows if r.get("skill_class") == "cert"]}
    st.session_state["chat"] = []


# ------------------------------------------------------------------ showing evidence

def marked(sentence, spans):
    """The sentence as HTML with every span marked. The text comes from a resume or a
    posting, so each piece is escaped before it goes on the page."""
    spans = sorted({s for s in spans if s}, key=len, reverse=True)
    if not spans:
        return escape(sentence)
    out, last = [], 0
    for m in re.finditer("|".join(map(re.escape, spans)), sentence):
        out += [escape(sentence[last:m.start()]), f"<mark>{escape(m[0])}</mark>"]
        last = m.end()
    return "".join(out) + escape(sentence[last:])


def evidence_list(rows):
    """Every skill once, with the sentences it came from. The first is shown; the rest open on a click."""
    by_skill = {}
    for r in rows:
        by_skill.setdefault(r["skill"], []).append(r)
    parts = []
    for skill, found_in in by_skill.items():
        quotes = [f'<p class="quote">{marked(r["sentence"], [r["span"]])}</p>' for r in found_in]
        more = len(quotes) - 1
        rest = (f'<details><summary>{more} more {"sentence" if more == 1 else "sentences"}</summary>'
                f'{"".join(quotes[1:])}</details>') if more else ""
        parts.append(f'<div class="evidence"><div class="skill">{escape(skill)}</div>'
                     f'<div>{quotes[0]}{rest}</div></div>')
    st.html("".join(parts))


def evidence_box(message, rows):
    """A small button that opens the sources an answer cites, grouped by where they come from.
    The box scrolls, so a long list never pushes the chat down the page."""
    spans, groups = {}, {}
    for r in rows:
        spans.setdefault(r["sentence"][:300], []).append(r["span"])
    for key in message["cited"]:
        text = message["sources"][key]
        quoted = re.fullmatch(r'"(.*)" \(names: .*\)', text, re.S)
        groups.setdefault(key[0], []).append(
            f'<p class="quote">{marked(quoted[1], spans.get(quoted[1], []))}</p>' if quoted
            else f'<p class="fact">{escape(to_you(text))}</p>')
    label = ", ".join(f"{len(groups[k])} {SOURCE_COUNT[k]}" for k in "SJAC" if k in groups)
    with st.popover(f"Based on {label}"):
        with st.container(height=360):
            st.html("".join(f'<div class="source"><div class="kind">{SOURCE_KIND[k]}</div>{"".join(groups[k])}</div>'
                            for k in "SJAC" if k in groups))


def plain(text):
    return text.replace("$", "\\$")  # Streamlit would read $...$ as a formula


def readable(text):
    """The answer without the agents' citation codes. The evidence box shows the sources instead."""
    return plain(re.sub(r"\s*\[[SJAC]\d+\]", "", text))


def to_you(source):
    """A posting source as the person reads it. The agents' sources say "the person"."""
    return (source.replace("Shares with the person:", "You share:")
            .replace("Asks for that the person lacks:", "It also asks for:"))


def named_in(text, skills, k=3):
    """The skills an answer names, so the bars beside it show the same ones. The top k if it names none."""
    def named(s):
        return any(re.search(rf"(?<![\w.]){re.escape(n.lower())}(?![\w.])", text.lower())
                   for n in {short(s["skill"]), s["skill"]})
    return ([s for s in skills if named(s)] or skills)[:k]


def opening(text, limit=1100):
    """A long answer cut at a line break: the first lines, up to about `limit` characters, and
    the rest. The plain LLM writes far more than fits beside TraceHire's answer."""
    lines, start = text.split("\n"), []
    while lines and (not start or len("\n".join(start + lines[:1])) <= limit):
        start.append(lines.pop(0))
    return "\n".join(start), "\n".join(lines).strip()


def short(skill):
    """The everyday name of a skill: "Structured query language SQL" is shown as "SQL"."""
    return SHORT_NAME.get(skill, re.sub(r" software$", "", skill))


def job_rows(jobs):
    """Postings as rows, each with the skills the person has and lacks for it."""
    parts = []
    for job in jobs:
        company = job["company_name"] or "Company not named"
        asked, shared = len(job["shared"]) + len(job["missing"]), [short(s) for s in job["shared"]]
        more = f" and {len(shared) - 6} more" if len(shared) > 6 else ""
        lacks = (f'<p class="fact kind">Not in your resume: {escape(", ".join(short(s) for s in job["missing"][:6]))}</p>'
                 if job["missing"] else "")
        parts.append(f'<div class="job"><div class="job-head"><span class="skill">{escape(job["title"])}</span>'
                     f'<span class="pill">{len(job["shared"])} of {asked} skills</span></div>'
                     f'<div class="kind">{escape(company)}, {escape(job["location"])}</div>'
                     f'<p class="fact">You have: {escape(", ".join(shared[:6]))}{more}</p>{lacks}</div>')
    return "".join(parts)


def learn_rows(skills, narrow=False):
    """Skills to learn as rows with a bar: how many of the 100 closest postings ask for each.
    narrow: the bar goes under the name, for the half-width compare card."""
    most = max((s["postings"] for s in skills), default=1)
    return "".join(
        f'<div class="learn{" narrow" if narrow else ""}"><div class="skill">{escape(short(s["skill"]))}'
        f'{"<span class=hot>hot</span>" if s["hot"] == "Y" and not narrow else ""}</div>'
        f'<div class="track"><div class="fill" style="width:{100 * s["postings"] / most:.0f}%"></div></div>'
        f'<div class="n">{s["postings"]} of 100</div></div>' for s in skills)


# ------------------------------------------------------------------ the chat

def ask(question, resume, compare):
    message = {"question": question}
    try:
        state = agents.run(question, resume["text"], resume["rows"])
        message.update({k: state.get(k, []) for k in ("text", "cited", "sources", "route", "dropped")})
    except Exception as e:  # network, key or API errors: show them, don't crash the page
        message["error"] = str(e)
    if compare:
        try:
            message["baseline"] = agents.answer_baseline(question, resume["text"])
        except Exception as e:
            message["baseline"] = f"The plain LLM could not answer: {e}"
    return message


def show_answer(m, rows, side_by_side=False):
    """One answer. side_by_side: it sits in the narrower compare card, so its lists are shorter."""
    if "error" in m:
        st.error(f'TraceHire could not answer: {m["error"]}')
        return
    have = {r["skill"] for r in rows}
    st.markdown(readable(m["text"]))
    if m["route"] == "match" and have:
        jobs = pipeline.match_jobs(have, k=5)
        st.caption("The closest of our 2,000 stored postings. They are a 2024 snapshot, not live listings.")
        st.html(job_rows(jobs[:3]))
        if jobs[3:] and not side_by_side:
            with st.expander(f"{len(jobs) - 3} more postings"):
                st.html(job_rows(jobs[3:]))
    if m["route"] == "advice" and have:
        learn = pipeline.skills_to_learn(have, k=8)
        if side_by_side:
            st.caption("How many of your 100 closest postings ask for it")
            st.html(learn_rows(named_in(m["text"], learn), narrow=True))
        else:
            st.caption("Of the 100 postings closest to you, how many ask for each skill. "
                       "Hot marks an O*NET hot technology.")
            st.html(learn_rows(learn))
    if m["route"] == "outreach" and m["cited"]:
        with st.expander("Copy to send"):
            st.code(readable(m["text"]).strip(), language=None, wrap_lines=True)
    if m["sources"] and not m["cited"]:
        st.warning("This answer cites no sources, so it cannot be checked.")
    if m["cited"]:
        evidence_box(m, rows)
    if m["route"] not in AGENT or side_by_side:  # "other": a greeting or an off-topic question, answered by no agent
        return
    note = f'Answered by the {AGENT[m["route"]]} agent.'
    if m["dropped"]:
        note += f' {len(m["dropped"])} citations pointed to no source and were removed.'
    st.caption(note)


def show(m, rows, n):
    """One question and its answer. n numbers the message, to name its two compare cards."""
    st.container(key=f"ask-{n}").chat_message("user", avatar=":material/person:").markdown(plain(m["question"]))
    with st.chat_message("assistant", avatar=":material/fact_check:"):
        if "baseline" not in m:
            show_answer(m, rows)
            return
        # Two columns of one height, so the answers sit level whatever their length. Each scrolls.
        theirs, ours = st.columns(2, gap="large")
        with theirs.container(border=False, height=COMPARE_HEIGHT, key=f"side-plain-{n}"):
            st.html('<div class="side"><b>Plain LLM</b><span>No sources to check</span></div>')
            start, rest = opening(plain(m["baseline"]))
            st.markdown(start)
            if rest:
                with st.expander("Read the rest of this answer"):
                    st.markdown(rest)
        with ours.container(border=False, height=COMPARE_HEIGHT, key=f"side-ours-{n}"):
            st.html('<div class="side ours"><b>TraceHire</b><span>Every claim has a source</span></div>')
            show_answer(m, rows, side_by_side=True)


st.html('<div class="brand">TraceHire</div>'
        '<div class="tagline">A LinkedIn Assistant whose every claim points back to evidence</div>')

resume = st.session_state.get("resume")
if not resume:
    st.write("Upload your resume, then ask about jobs that fit you, what to learn next, a company, or a "
             "message to a recruiter. Every answer shows the resume sentences and job postings it is based on.")
    file = st.file_uploader("Your resume, as a PDF or a text file", type=["pdf", "txt"])
    pasted = st.text_area("Or paste the text of your resume", height=160)
    st.caption("Your resume text is sent to the language model at ASU, which reads it and answers your questions. "
               "The computer running this page keeps the model's reading of it, so the same resume loads faster next time.")
    if file:
        text = read_upload(file)
        if text.strip():
            load_resume(file.name, text)
            st.rerun()
        st.error("We could not read any text in that file. It may be a scanned image. Paste the text instead.")
    if st.button("Use this text", type="primary", disabled=not pasted.strip()):
        load_resume("Pasted resume", pasted)
        st.rerun()
    if st.query_params.get("test"):
        tests = {r["resume_id"]: r for r in pipeline.read("resumes.csv")}
        order = [i for i in DEMO_RESUMES if i in tests] + [i for i in tests if i not in DEMO_RESUMES]
        picked = st.selectbox("Test mode: a resume from the dataset", order,
                              format_func=lambda i: f'{i} ({tests[i]["category"].title()})')
        if st.button("Load this test resume"):
            load_resume(f"Test resume {picked}", tests[picked]["text"], picked)
            st.rerun()
    st.stop()

rows, chat = resume["rows"], st.session_state["chat"]
skills = len({r["skill"] for r in rows})
# One slim row for the resume: its name, a small button that opens what was read in it, and a way to change it.
about, read, change = st.columns([5, 1.6, 1.4], vertical_alignment="center")
about.markdown(f'**{resume["name"]}**')
if rows:
    with read.popover(f"{skills} skills found", use_container_width=True):
        st.caption("Every skill we read in your resume, next to the sentence it came from")
        with st.container(height=420):
            evidence_list(rows)
            if resume["certs"]:
                st.markdown("**Certifications**, listed apart and not counted as skills")
                evidence_list(resume["certs"])
if change.button("Change resume", use_container_width=True):
    del st.session_state["resume"]
    st.rerun()
if resume["note"]:
    st.info(resume["note"])
if not rows:
    st.warning("We found no skills we recognise in this resume, so the answers will have little to go on.")
compare = st.toggle("Compare with a plain LLM", help="Shows the same question answered by the language model "
                    "alone, with no evidence, next to TraceHire's answer.")

for n, message in enumerate(chat):
    show(message, rows, n)

starter = None if chat else st.pills("Try asking", STARTERS)
question = st.chat_input("Ask about jobs, skills to learn, a company, or a message to send") or starter
if question:
    st.container(key="ask-new").chat_message("user", avatar=":material/person:").markdown(plain(question))
    with st.chat_message("assistant", avatar=":material/fact_check:"), st.spinner("Checking the evidence"):
        chat.append(ask(question, resume, compare))
    st.rerun()

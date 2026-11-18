"""The extractors compared in E1, all writing the team's shared skills format.

  lexicon    the team's word matcher (pipeline.extract_skills_lexicon), classical baseline
  llm_alone  the LLM lists skills freely: no sentence, no span (E1 only, never stored)
  llm_prov   the LLM fills ontology slots per sentence and must quote its span;
             the provenance check drops anything unsupported
  final      llm_prov with negated mentions removed and the cleanup in clean.py;
             this is what the page uses

Shared columns (DATA/README.md): doc_id, doc_type, sent_id, sentence, span,
skill, vocab, extractor, confidence. Extra columns: canonical_id, span_start,
span_end (character offsets of the span inside the sentence), skill_class,
negated, proficiency, requirement, check, model.
"""
import re

import pipeline

from . import config, llm
from .lexicon import Lexicon, skill_key
from .provenance import GENERIC, check
from .sentences import split

BATCH_SENTENCES = 8
BATCH_CHARS = 1800
FIELDS = ["doc_id", "doc_type", "sent_id", "sentence", "span", "skill", "vocab", "extractor",
          "confidence", "canonical_id", "span_start", "span_end", "skill_class", "negated",
          "proficiency", "requirement", "check", "model"]

ONTOLOGY_PROMPT = """You fill the TraceHire skill ontology from numbered sentences of a {doc_type}.

Skill classes:
- tool: a named software product, programming language, library, platform or system (e.g. Python, Excel, AWS, SAP)
- technique: a method or practice (e.g. machine learning, ETL, regression analysis, budgeting)
- domain: a field of knowledge (e.g. accounting, GAAP, network security, payroll)
Do NOT extract soft skills or personality traits (e.g. teamwork, communication, leadership, motivated).
Do NOT extract job titles, company names, degrees or certifications.

For every skill mention return an object with:
- "sent_id": the number of the sentence it is in
- "skill": a short standard name for the skill
- "quote": the exact words copied from that sentence that name the skill (copy them, do not paraphrase)
- "class": "tool", "technique" or "domain"
- "negated": true only if the sentence says the skill is missing or not needed
- "proficiency": "basic", "intermediate", "advanced" or null (only if the sentence states it)
- "requirement": "required", "preferred" or null (job postings only)

Only extract skills that are written in the sentences. Return JSON: {{"skills": [ ... ]}}"""

ALONE_PROMPT = """List the skills in this text from a {doc_type}: technical skills, tools, software, programming languages, methods and areas of domain knowledge. Do not include soft skills or personality traits. Return JSON: {{"skills": ["skill name", ...]}}"""

_LEXICON = None


def lexicon():
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = Lexicon.load()
    return _LEXICON


def canonical_id(skill, lex, quote=None):
    """Canonical skill ID: the skill list's ID when known, otherwise a normalized key."""
    cid = lex.resolve(skill)
    if not cid.startswith("raw:"):
        return cid
    if quote:
        # Only when the matched name covers the skill's words: "API Connect" must not
        # become "Adobe Connect", nor "AI/ML" a product named "... AI".
        need = set(re.split(r"[\s\-/]+", skill_key(skill))) - GENERIC - {""}
        hits = {h[3] for h in lex.find(quote)
                if need <= set(re.split(r"[\s\-/]+", skill_key(h[2])))}
        if len(hits) == 1:
            return hits.pop()
    words = [w for w in skill_key(skill).split() if w not in GENERIC]
    return "raw:" + " ".join(words or skill_key(skill).split())


CERT = re.compile(
    r"certifi|licen[cs]e\b|licensure|\b(CompTIA|CCNA|CCNP|CCIE|CCENT|CISSP|CISA|CISM|CEH|PMP|CAPM|"
    r"MCSE|MCSA|MCSD|MCP|MCTS|MCITP|OCA|OCP|RHCE|RHCSA|LPIC|CSM|PMI-ACP|GIAC|"
    r"ITIL\s*v?\d? *(Foundation|Practitioner|Expert)|Six Sigma (Green|Black|Yellow) Belt|"
    r"(Security|Network|Linux|Server|Cloud|Project)\+|A\+)(?!\w)", re.I)


# Skills that mention certificates or licences but are not credentials ("SSL certificates",
# "software licensing", "Certification and Accreditation").
NOT_CERT = re.compile(r"software|accreditation|maturity model|process|digital|SSL|X\.509|public key|"
                      r"management|authentication|workstation|drawing|corner|known exploited|"
                      r"compliance|utilization|rules|modeling program", re.I)


def is_cert(skill, span=""):
    """Certifications and licences (CompTIA A+, CCNA, PMP, "Certified ..."). The app
    lists them apart from skills."""
    skill = skill or ""
    return bool(CERT.search(skill)) and not NOT_CERT.search(skill)


def row(doc_id, doc_type, extractor, skill, lex, sent=None, span=None, quote_hint=None, **extra):
    """One record in the shared format. Known skills use the skill list's name."""
    quote = sent.text[span[0]:span[1]] if sent is not None and span else None
    cid = canonical_id(skill, lex, quote or quote_hint)
    known = not cid.startswith("raw:")
    r = dict.fromkeys(FIELDS)
    r.update(doc_id=doc_id, doc_type=doc_type, extractor=extractor, canonical_id=cid,
             skill=lex.label(cid) if known else skill, vocab=lex.source if known else "none")
    if quote is not None:
        r.update(sent_id=sent.sent_id, sentence=sent.text, span=quote,
                 span_start=span[0], span_end=span[1])
    r.update(extra)
    if is_cert(r["skill"], r["span"]):
        r["skill_class"] = "cert"
    return r


def batches(sentences):
    batch, size = [], 0
    for s in sentences:
        if s.section == "header":
            continue
        if batch and (len(batch) >= BATCH_SENTENCES or size + len(s.text) > BATCH_CHARS):
            yield batch
            batch, size = [], 0
        batch.append(s)
        size += len(s.text)
    if batch:
        yield batch


def _kind(doc_type):
    return "job posting" if doc_type == "job" else "resume"


def _ask(system, batch, render):
    """The "skills" list the LLM returns for a batch. A cut-off reply means the
    batch had too many skills for one answer, so it is split in half and asked again."""
    try:
        resp = llm.chat_json(system, render(batch))
    except llm.Truncated as e:
        if len(batch) == 1:
            llm.FAILURES.append((system[:60], render(batch)[:200], str(e)))
            return []
        half = len(batch) // 2
        return _ask(system, batch[:half], render) + _ask(system, batch[half:], render)
    skills = resp.get("skills", []) if isinstance(resp, dict) else []
    return skills if isinstance(skills, list) else []


def extract_lexicon(doc_id, doc_type, text, sentences=None):
    lex = lexicon()
    sents = {s.sent_id: s for s in sentences or split(text)}
    out = []
    for r in pipeline.extract_skills_lexicon(text):
        s = sents[r["sent_id"]]
        start = s.text.find(r["span"])
        out.append(row(doc_id, doc_type, "lexicon", r["skill"], lex, s, (start, start + len(r["span"])),
                       confidence=1.0, negated=False))
    return out


def extract_llm_alone(doc_id, doc_type, text, sentences=None):
    lex = lexicon()
    out = []
    for batch in batches(sentences or split(text)):
        names = _ask(ALONE_PROMPT.format(doc_type=_kind(doc_type)), batch,
                     lambda b: "\n".join(s.text for s in b))
        for name in names:
            if isinstance(name, dict):
                name = name.get("skill") or name.get("name") or ""
            name = str(name).strip()
            if name:
                out.append(row(doc_id, doc_type, "llm_alone", name, lex, negated=False,
                               model=config.LLM_MODEL))
    return out


def extract_llm_prov(doc_id, doc_type, text, sentences=None):
    """Returns (kept_rows, rejected_items)."""
    lex = lexicon()
    kept, rejected = [], []
    for batch in batches(sentences or split(text)):
        items = _ask(ONTOLOGY_PROMPT.format(doc_type=_kind(doc_type)), batch,
                     lambda b: "\n".join(f"[{s.sent_id}] {s.text}" for s in b))
        by_id = {s.sent_id: s for s in batch}
        for item in items:
            if not isinstance(item, dict):
                continue
            item["sent_id"] = int(re.sub(r"\D", "", str(item.get("sent_id"))) or -1)
            verdict, sent, span = check(item, by_id, lex)
            if verdict not in ("ok", "sent_id_corrected"):
                rejected.append({"doc_id": doc_id, "doc_type": doc_type, "sent_id": item["sent_id"],
                                 "sentence": by_id[item["sent_id"]].text if item["sent_id"] in by_id else None,
                                 "skill": item.get("skill"), "quote": item.get("quote"), "reason": verdict})
                continue
            kept.append(row(doc_id, doc_type, "llm_prov", str(item["skill"]).strip(), lex, sent, span,
                            skill_class=item.get("class"), negated=bool(item.get("negated")),
                            proficiency=item.get("proficiency"), requirement=item.get("requirement"),
                            check=verdict, model=config.LLM_MODEL))
    return kept, rejected


def extract_final(doc_id, doc_type, text, rules=None, sentences=None):
    """llm_prov plus learned rules, negated mentions removed, one row per skill per sentence,
    then the cleanup in clean.py (soft skills, degrees and employer names dropped)."""
    from .clean import clean
    sents = sentences or split(text)
    kept, _ = extract_llm_prov(doc_id, doc_type, text, sents)
    rows = [dict(r, extractor="final:llm_prov") for r in kept if not r["negated"]]
    if rules is not None:
        rows += [dict(r, extractor="final:rules") for r in rules.extract(doc_id, doc_type, sents)]
    seen, out = set(), []
    for r in rows:
        key = (r["sent_id"], r["canonical_id"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return clean(out)

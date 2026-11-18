"""Last cleanup of the final rows, following our labeling guide.

The LLM sometimes returns things the guide says are not skills. This step:
1. folds odd skill_class values into tool / technique / domain / cert,
2. drops soft skills, degrees and level words ("problem solving", "Master's degree", "Advanced"),
3. asks the LLM, for skills that look like an organisation name, whether the sentence uses the
   name as a tool or as an employer / client ("support for Google" vs "Google Apps"), and drops
   the employer / client ones.

Our extraction experiment scores the output before this step and after it, so the effect of
the cleanup is measured.
"""
import csv
import re
from functools import cache

from . import config, llm

SOFT = re.compile(
    r"^((complex|creative|strong|excellent|effective|good)\s+)?("
    r"problem[- ]solving|analytical (skills|thinking)|critical thinking|"
    r"(technical |team |project |thought )?leadership( skills| experience| development)?|"
    r"(written |verbal |oral |interpersonal |excellent )*communication( skills)?|"
    r"(written and verbal|verbal and written) communication( skills)?|"
    r"interpersonal( skills)?|teamwork|team ?player|collaboration|time management|"
    r"attention to detail|detail[- ]oriented|organi[sz]ational skills|multi-?tasking|"
    r"self[- ]motivated|work ethic|adaptability|flexibility|customer focus"
    r")$", re.I)

DEGREE = re.compile(r"\bdegree\b(?! of freedom)|^(bachelor|master|associate|doctor)('?s)?( of [a-z ]+)?$|^(MBA|B\.?S\.?|M\.?S\.?|Ph\.?D\.?)$",
                    re.I)
LEVEL = {"advanced", "intermediate", "basic", "beginner", "expert", "senior", "junior", "proficient"}

# Needs a name before the suffix: "New York Life Insurance" is checked, "insurance" (an industry) is not.
ORG_SUFFIX = re.compile(r"^\S.*\s(Inc|LLC|L\.L\.C|Corporation|Corp|Ltd|Company|Co\.|Insurance|Bank|Group|Holdings)\.?$",
                        re.I)


def skill_class(value, skill):
    """One of tool, technique, domain, cert; None for rows that are not skills."""
    v = (value or "").lower()
    if "cert" in v:
        return "cert"
    if v in ("degree", "proficiency", "soft skill", "soft"):
        return None
    if v in ("tool", "technique", "domain", "cert"):
        return v
    if any(w in v for w in ("tool", "platform", "framework", "software", "language", "library")):
        return "tool"
    if any(w in v for w in ("technique", "method", "standard", "practice")):
        return "technique"
    if "domain" in v or "knowledge" in v:
        return "domain"
    return "technique" if v else ("technique" if skill else None)


@cache
def company_names():
    path = config.SLICE / "companies.csv"
    if not path.exists():
        return frozenset()
    with open(path, encoding="utf-8") as f:
        return frozenset(r["name"].strip().lower() for r in csv.DictReader(f) if len(r["name"].strip()) > 2)


def looks_like_org(skill):
    s = (skill or "").strip()
    return s.lower() in company_names() or bool(ORG_SUFFIX.search(s))


ORG_PROMPT = """A skill extractor quoted some words from a sentence of a resume or job posting.
The quoted words name something that is also a company. Look ONLY at the quoted words and the
words right around them in the sentence, and decide what that exact occurrence refers to:
- "skill": a product, tool, platform or technology that is used, configured, programmed or asked for.
  The quote usually continues with a product word, or follows "using", "with", "in", "on".
  Examples: quote "Google Groups" in "access with Active Directory and Google Groups" -> skill.
  Quote "Cisco" in "configured Cisco routers" -> skill. Quote "Siemens" in "HMI (AB Panel View, Siemens)" -> skill.
- "organization": a company the person worked for, supported, sold to or partnered with.
  It usually follows "for", "at", "of", "to", or names the people or accounts of that company.
  Examples: quote "Google" in "set up email aliases for Google, Google X, and Nest" -> organization.
  Quote "New York Life Insurance" in "support for 600 New York Life Insurance agents" -> organization.
  Quote "IBM" in "worked at IBM as a consultant" -> organization.
Reply in JSON: {"use": "skill"} or {"use": "organization"}."""


def is_org_mention(skill, sentence, quote=""):
    user = f'Sentence: "{sentence}"\nQuoted words: "{quote or skill}"\nSkill name: "{skill}"'
    return llm.chat_json(ORG_PROMPT, user).get("use") == "organization"


def clean(rows):
    """Rows the guide counts as skills, with skill_class normalised."""
    out = []
    for r in rows:
        skill = (r.get("skill") or "").strip()
        cls = skill_class(r.get("skill_class"), skill)
        if r.get("skill_class") == "cert":
            cls = "cert"
        if cls is None or SOFT.match(skill) or DEGREE.search(skill) or skill.lower() in LEVEL:
            continue
        if cls != "cert" and looks_like_org(skill):
            try:
                if is_org_mention(skill, r.get("sentence") or "", r.get("span") or ""):
                    continue
            except llm.Truncated:
                pass
        out.append(dict(r, skill_class=cls))
    return out

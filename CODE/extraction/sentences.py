"""Numbered sentences from the team's shared splitter, plus the section each one is in.

The sentences and their numbers come from pipeline.split_sentences, so a
sent_id means the same thing everywhere in the project. Section headers
("Summary", "Skills") stay in the list as their own sentences; they only
set the section of the sentences after them.
"""
import re
from dataclasses import dataclass

import pipeline

SECTION_HEADERS = {
    "summary", "professional summary", "executive summary", "executive profile", "profile",
    "objective", "career overview", "highlights", "skills", "skill highlights",
    "technical skills", "core qualifications", "qualifications", "core competencies",
    "experience", "work experience", "professional experience", "work history",
    "employment history", "education", "education and training", "certifications",
    "certificates", "accomplishments", "achievements", "awards", "languages", "interests",
    "additional information", "activities", "activities and honors", "affiliations",
    "projects", "publications", "presentations", "volunteer experience",
    "responsibilities", "requirements", "qualifications and requirements",
    "about the role", "about us", "what you'll do", "what you will do",
    "what we're looking for", "preferred qualifications", "minimum qualifications",
    "required qualifications", "benefits", "job description", "job summary",
}


@dataclass
class Sentence:
    sent_id: int
    text: str
    section: str


def _header(text):
    key = re.sub(r"[^a-z' ]", "", text.lower()).strip()
    return key if key in SECTION_HEADERS else None


def split(text):
    out, section = [], "other"
    for i, s in enumerate(pipeline.split_sentences(text)):
        head = _header(s)
        if head:
            section = head
        out.append(Sentence(i, s, "header" if head else section))
    return out

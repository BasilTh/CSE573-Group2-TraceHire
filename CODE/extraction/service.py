"""The entry point the rest of the project calls: pipeline.extract_skills(text, extractor="llm").

Returns the same row fields as the lexicon extractor (sent_id, sentence, span,
skill, vocab, extractor, confidence) plus the extra columns described in
extractors.py. LLM answers are cached, so a document already seen costs nothing.

If the LLM can't be reached (or a sentence never gets a usable answer),
extract_rows raises ExtractorUnavailable instead of returning a quietly
incomplete list, so the app can say so and fall back to the word matcher.
"""
from . import llm
from .extractors import extract_final


class ExtractorUnavailable(RuntimeError):
    """Some of the text got no answer from the LLM."""


def extract_rows(text, doc_type="resume"):
    failures_before = len(llm.FAILURES)
    # The provenance-checked LLM output plus the cleanup, the way skills_llm.csv was made.
    rows = extract_final(None, doc_type, text, rules=None)
    if len(llm.FAILURES) > failures_before:
        raise ExtractorUnavailable(llm.FAILURES[-1][2])
    for r in rows:
        r.pop("doc_id")
        r.pop("doc_type")
    return rows

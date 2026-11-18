"""The provenance check (Idea A): a skill survives only if it quotes its sentence.

Two tests, both mechanical:
  1. The quote occurs in the stated sentence (case-insensitive, whole words,
     flexible whitespace). Its character span is recorded.
  2. The skill name is supported by the quote: every word of the skill appears
     in the quote, or the lexicon maps the quote to the same canonical skill.
Anything that fails is dropped and logged with the reason.
"""
import re

from .lexicon import skill_key


def find_quote(quote, sentence_text):
    """Return (start, end) of quote inside sentence_text, or None."""
    words = quote.strip().split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(w) for w in words)
    m = re.search(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])", sentence_text, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


# Words an LLM adds when naming a skill ("Python programming") that do not
# change which skill it is.
GENERIC = {"programming", "language", "languages", "software", "development",
           "skills", "skill", "experience", "tools", "tool", "framework",
           "platform", "the", "and", "of", "in"}


def _words(text):
    return set(re.split(r"[\s\-/]+", skill_key(text))) - {""}


def _stem_match(word, words):
    """Same word, or same first 5 letters ("analysis"/"analyses", "budget"/"budgets")."""
    return word in words or (len(word) >= 5 and any(len(w) >= 5 and w[:5] == word[:5] for w in words))


def skill_supported(skill, quote, lexicon):
    needed = _words(skill) - GENERIC
    quoted = _words(quote)
    if needed and all(_stem_match(w, quoted) for w in needed):
        return True
    cid = lexicon.resolve(skill)
    if cid.startswith("raw:"):
        return False
    return lexicon.resolve(quote) == cid or any(h[3] == cid for h in lexicon.find(quote))


def check(item, sentences, lexicon):
    """Validate one LLM item against the batch's sentences.

    item: dict with sent_id, skill, quote.  sentences: {sent_id: Sentence}.
    Returns (verdict, sentence, (start, end)) where verdict is "ok",
    "sent_id_corrected", or a rejection reason.
    """
    quote = str(item.get("quote") or "").strip()
    skill = str(item.get("skill") or "").strip()
    if not skill:
        return "no_skill", None, None
    if not quote:
        return "no_quote", None, None

    verdict = "ok"
    sent = sentences.get(item.get("sent_id"))
    span = find_quote(quote, sent.text) if sent else None
    if span is None:
        # The LLM sometimes misnumbers a sentence; accept the quote only if it
        # occurs in exactly one sentence of the batch.
        hits = [(s, find_quote(quote, s.text)) for s in sentences.values()]
        hits = [(s, sp) for s, sp in hits if sp]
        if len(hits) != 1:
            return "quote_not_in_sentence", sent, None
        sent, span = hits[0]
        verdict = "sent_id_corrected"

    if not skill_supported(skill, quote, lexicon):
        return "skill_not_in_quote", sent, span
    return verdict, sent, span

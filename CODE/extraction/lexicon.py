"""Skill lexicon: canonical skill IDs, aliases, and dictionary matching.

Uses the Lightcast Open Skills list when DATA/raw/lightcast/skills.json exists,
otherwise the team's O*NET list (DATA/slice/skills_vocab.csv) plus the short
forms in pipeline.SHORT_FORMS. Every skill gets a stable ID, so all extractors
and the gold labels are compared on the same canonical skills, and the
preferred name matches the `skill` column the lexicon baseline writes.
"""
import csv
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from . import config

VENDORS = {
    "microsoft", "adobe", "oracle", "apple", "google", "amazon", "ibm", "sap",
    "atlassian", "autodesk", "apache", "salesforce", "intuit", "cisco", "the",
    "amazon web services", "aws", "red hat", "vmware", "dassault systemes",
}

_TOKEN = re.compile(r"\.?[A-Za-z0-9][A-Za-z0-9+#.\-]*[A-Za-z0-9+#]|\.?[A-Za-z0-9][+#]*")
_MAX_NGRAM = 6


def skill_key(text):
    """Normalized comparison key: lowercase, punctuation-light, single spaces."""
    s = text.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9+#.\- ]", " ", s)
    s = re.sub(r"(?<![a-z0-9])[.\-]|[.\-](?![a-z0-9+#])", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=1)
def _english_words():
    p = Path("/usr/share/dict/words")
    if not p.exists():
        return frozenset()
    return frozenset(w.strip().lower() for w in p.read_text().splitlines() if w.strip())


# Vendor-stripped names that are everyday words ("Microsoft Project" -> "Project")
# would fire on ordinary text, so they are not used as aliases.
AMBIGUOUS = {"project", "word", "access", "director", "office", "teams", "publisher",
             "paint", "forms", "flow", "lens", "photos", "mail", "maps", "sheets", "connect"}

# Acronyms that are general terms, not one product: "AI" must not become
# "National Instruments NI Vision Builder for Automated Inspection AI".
GENERIC_ACRONYMS = {"ai", "bi", "ml", "it", "hr", "qa", "ui", "ux", "pm", "ci", "cd", "io", "os",
                    "db", "ip", "pc", "id", "ar", "vr", "lms", "plm", "mes", "ehr", "emr", "cnc",
                    "dcs", "ics", "erp", "crm", "cms", "hris", "ats", "pos", "lims", "alm", "gis",
                    "cad", "cam", "scada", "hmi", "plc", "mrp", "wms", "tms", "pms", "cis", "ads"}
_STOP = {"for", "of", "and", "the", "to", "in", "on", "with", "a", "&"}


def _trailing_acronym(name):
    """(acronym, initials of the words before it, same without small words), or None."""
    tokens = re.sub(r"\s+software$", "", name, flags=re.I).split()
    if len(tokens) >= 3 and tokens[-1].isupper() and 2 <= len(tokens[-1]) <= 6:
        return (tokens[-1].lower(), "".join(t[0].lower() for t in tokens[:-1]),
                "".join(t[0].lower() for t in tokens[:-1] if t.lower() not in _STOP))
    return None


def _aliases(name, shared_acronyms=frozenset()):
    out = {name}
    base = re.sub(r"\s+software$", "", name, flags=re.I)
    if base != name and base:
        out.add(base)
    tokens = base.split()
    low = base.lower()
    for v in sorted(VENDORS, key=len, reverse=True):
        if low.startswith(v + " ") and len(base) > len(v) + 1:
            stripped = base[len(v) + 1:]
            if stripped.lower() not in AMBIGUOUS:
                out.add(stripped)
            break
    # Trailing acronym that spells the words before it: "Amazon Web Services AWS".
    # Never for general terms (AI, LMS, POS). Otherwise kept when it spells the whole name,
    # or when no other product ends with the same acronym.
    t = _trailing_acronym(name)
    if t:
        acro, initials, initials_no_stop = t
        whole = acro in (initials, initials_no_stop)
        if acro not in GENERIC_ACRONYMS and initials.endswith(acro) and (whole or acro not in shared_acronyms):
            out.add(tokens[-1])
            out.add(" ".join(tokens[:-1]))
    # JavaScript spellings: "Node.js" also as "NodeJS", "React" also as "React.js" and "ReactJS".
    if re.fullmatch(r"[A-Za-z]+\.js", base, flags=re.I):
        out |= {base[:-3] + "JS", base[:-3] + " js"}
    elif re.fullmatch(r"[A-Z][a-z]+", base) and base.lower() in JS_LIBRARIES:
        out |= {base + ".js", base + "JS"}
    return out


JS_LIBRARIES = {"react", "angular", "vue", "express", "ember", "backbone", "next", "nuxt", "svelte",
                "knockout", "meteor", "three", "d3", "chart"}


class Lexicon:
    def __init__(self, entries, source, extra_aliases=()):
        """entries: iterable of (canonical_id, preferred_name).
        extra_aliases: (alias, canonical_id) pairs added before derived aliases."""
        self.source = source
        self.names = {}
        self.alias_to_id = {}
        self.case_sensitive = {}
        explicit, derived = [], []
        entries = list(entries)
        acronyms = Counter(t[0] for _, name in entries if (t := _trailing_acronym(name)))
        shared = frozenset(a for a, n in acronyms.items() if n > 1)
        for cid, name in entries:
            self.names[cid] = name
            explicit.append((cid, name))
            derived.extend((cid, a) for a in _aliases(name, shared) if a != name)
        words = _english_words()
        for cid, alias in explicit + [(c, a) for a, c in extra_aliases] + derived:
            key = skill_key(alias)
            if not key or key in self.alias_to_id:
                continue
            self.alias_to_id[key] = cid
            if len(key) <= 3 or alias.isupper() or (" " not in key and key in words):
                self.case_sensitive[key] = alias

    @classmethod
    def load(cls):
        if config.LIGHTCAST_SKILLS.exists():
            data = json.loads(config.LIGHTCAST_SKILLS.read_text())
            items = data.get("data", data) if isinstance(data, dict) else data
            return cls(((f"lc:{d['id']}", d["name"]) for d in items), "lightcast")
        import pipeline
        with open(config.SLICE / "skills_vocab.csv", newline="", encoding="utf-8") as f:
            names = [row["skill"] for row in csv.DictReader(f)]
        ids = {n: f"onet:{skill_key(n)}" for n in names}
        shorts = [(short, ids[full]) for short, full in pipeline.SHORT_FORMS.items() if full in ids]
        return cls(((ids[n], n) for n in names), "onet", shorts)

    def resolve(self, name):
        """Map any skill string to a canonical ID ('raw:<key>' when unknown)."""
        key = skill_key(name)
        return self.alias_to_id.get(key, f"raw:{key}")

    def label(self, cid):
        return self.names.get(cid, cid.split(":", 1)[-1])

    def find(self, text):
        """Greedy longest-match dictionary lookup. Yields (start, end, surface, canonical_id)."""
        toks = [(m.start(), m.end()) for m in _TOKEN.finditer(text)]
        i = 0
        while i < len(toks):
            hit = None
            for n in range(min(_MAX_NGRAM, len(toks) - i), 0, -1):
                s, e = toks[i][0], toks[i + n - 1][1]
                surface = text[s:e]
                key = skill_key(surface)
                cid = self.alias_to_id.get(key)
                if cid is None:
                    continue
                exact = self.case_sensitive.get(key)
                if exact is not None and surface != exact:
                    continue
                hit = (s, e, surface, cid, n)
                break
            if hit:
                yield hit[:4]
                i += hit[4]
            else:
                i += 1

"""Quick checks for the extraction code. Run from the repo root:

    python CODE/tests/test_extraction.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline  # noqa: E402
from extraction.extractors import extract_lexicon, lexicon  # noqa: E402
from extraction.lexicon import Lexicon  # noqa: E402
from extraction.provenance import check, find_quote  # noqa: E402
from extraction.sentences import split  # noqa: E402

LEX = Lexicon([("t:python", "Python"), ("t:spark", "Apache Spark"), ("t:go", "Go"),
               ("t:excel", "Microsoft Excel"), ("t:aws", "Amazon Web Services AWS software"),
               ("t:sql", "Structured query language SQL")], "test")


def test_sentence_ids_match_the_shared_splitter():
    text = "Summary   Built ETL pipelines in Python. Led a team.\nSkills\nPython, SQL"
    sents = split(text)
    assert [s.text for s in sents] == pipeline.split_sentences(text)
    assert [s.section for s in sents] == ["header", "summary", "summary", "header", "skills"]


def test_lexicon_aliases_and_ambiguity():
    assert LEX.resolve("Spark") == "t:spark"
    assert LEX.resolve("Excel") == "t:excel"
    assert LEX.resolve("AWS") == "t:aws"
    assert LEX.resolve("SQL") == "t:sql"
    assert [h[3] for h in LEX.find("Deployed on AWS with Python")] == ["t:aws", "t:python"]
    assert [h[3] for h in LEX.find("we go to market")] == []        # short names match case-sensitively


def test_team_lexicon_uses_short_forms():
    lex = lexicon()
    assert lex.label(lex.resolve("SQL")) == "Structured query language SQL"
    assert lex.label(lex.resolve("Spark")) == "Apache Spark"


def test_lexicon_rows_have_spans_inside_their_sentence():
    rows = extract_lexicon("d1", "resume", "Built data pipelines in Python and Spark on AWS.")
    assert {r["skill"] for r in rows} == {"Python", "Apache Spark", "Amazon Web Services AWS software"}
    for r in rows:
        assert r["sentence"][r["span_start"]:r["span_end"]] == r["span"]


def test_quote_matching():
    assert find_quote("built  ETL pipelines", "We Built ETL pipelines.") == (3, 22)
    assert find_quote("R", "React apps") is None                     # whole words only
    assert find_quote("C++", "Wrote C++ code") == (6, 9)


def test_provenance_check():
    sents = {0: split("Built ETL pipelines using Airflow and Python.")[0]}
    cases = {
        "ok": {"sent_id": 0, "skill": "Python", "quote": "Python"},
        "quote_not_in_sentence": {"sent_id": 0, "skill": "Spark", "quote": "Spark"},
        "skill_not_in_quote": {"sent_id": 0, "skill": "Spark", "quote": "ETL pipelines"},
        "sent_id_corrected": {"sent_id": 7, "skill": "Airflow", "quote": "Airflow"},
    }
    for verdict, item in cases.items():
        assert check(item, sents, LEX)[0] == verdict, (verdict, item)
    generic = {"sent_id": 0, "skill": "Python programming", "quote": "Python"}
    plural = {"sent_id": 0, "skill": "ETL pipeline", "quote": "ETL pipelines"}
    assert check(generic, sents, LEX)[0] == "ok"
    assert check(plural, sents, LEX)[0] == "ok"


def test_cut_off_reply_splits_the_batch(monkeypatch=None):
    from extraction import extractors, llm
    asked = []

    def fake_chat(system, user):
        asked.append(user.count("\n") + 1)
        if user.count("\n") + 1 > 2:                    # more than 2 sentences: "cut off"
            raise llm.Truncated("cut off")
        return {"skills": [{"sent_id": int(line[1:line.index("]")]), "skill": "Python", "quote": "Python"}
                           for line in user.splitlines()]}

    real = llm.chat_json
    llm.chat_json = fake_chat
    try:
        sents = split(" ".join(f"Used Python {i}." for i in range(5)))
        kept, rejected = extractors.extract_llm_prov("d", "resume", "", sentences=sents)
    finally:
        llm.chat_json = real
    assert asked[0] == 5 and len(asked) > 1                    # asked whole, then in smaller pieces
    assert sorted(r["sent_id"] for r in kept) == [0, 1, 2, 3, 4]  # no sentence lost


def test_old_cut_off_cache_entries_are_asked_again(tmp=None):
    import json
    import tempfile
    from pathlib import Path
    from extraction import llm
    path = Path(tempfile.mkdtemp()) / "x.json"
    path.write_text(json.dumps({"raw": '{"skills": [{"skill": "Py', "parsed": {}}))
    assert llm._cached(path) is None
    path.write_text(json.dumps({"raw": '{"skills": []}', "parsed": {"skills": []}}))
    assert llm._cached(path) == {"skills": []}


def test_certifications_are_tagged():
    from extraction.extractors import is_cert
    for name in ["CompTIA A+", "Cisco CCNA", "Certified Ethical Hacker", "PE license", "Security+",
                 "ITIL v3 Foundation", "Six Sigma Green Belt"]:
        assert is_cert(name), name
    for name in ["Python", "SSL certificates", "software licensing", "Certification and Accreditation",
                 "ITIL", "A/B testing", "C++"]:
        assert not is_cert(name), name


def test_cleanup_drops_what_the_guide_excludes():
    from extraction import clean as C
    rows = [{"skill": s, "skill_class": c, "sentence": "", "span": ""} for s, c in [
        ("Python", "tool"), ("problem solving", "technique"), ("Leadership", "technique"),
        ("Master's degree", "degree"), ("Advanced", "proficiency"), ("CompTIA A+", "certification"),
        ("Kubernetes", "platform/tool"), ("six degree of freedom physics model", "technique"),
        ("data communications", "domain")]]
    got = {r["skill"]: r["skill_class"] for r in C.clean(rows)}
    assert got == {"Python": "tool", "CompTIA A+": "cert", "Kubernetes": "tool",
                   "six degree of freedom physics model": "technique", "data communications": "domain"}, got


def test_no_wrong_product_matches():
    """Generic acronyms and partial quotes must not map to a product."""
    from extraction.extractors import canonical_id
    lex = lexicon()
    assert canonical_id("AI", lex, "AI") == "raw:ai"
    assert canonical_id("AI/ML", lex, "AI/ML") == "raw:ai ml"
    assert canonical_id("API Connect", lex, "API Connect") == "raw:api connect"
    assert canonical_id("Python programming", lex, "Python") == lex.resolve("Python")
    assert canonical_id("React.js", lex, "React.js") == canonical_id("React", lex, "React")
    assert canonical_id("NodeJS", lex, "NodeJS") == canonical_id("Node.js", lex, "Node.js")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

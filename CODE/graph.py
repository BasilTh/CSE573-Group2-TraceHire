"""Loads the data slice into the shared Neo4j database (login in .env).

Nodes: Candidate (resume), Job, Skill, Company, Location.
Edges: (Candidate)-[:HAS_SKILL]->(Skill), (Job)-[:REQUIRES_SKILL]->(Skill),
       (Job)-[:POSTED_BY]->(Company), (Job)-[:LOCATED_IN]->(Location).
Every skill edge is one mention and carries its provenance: doc_id, sent_id,
sentence, span, extractor, confidence. A row with no sentence or span is not loaded,
and certifications (skill_class = cert) are not loaded as skills.

Run from the repo root. It wipes the database and loads it again:

    python CODE/graph.py                    # skills from DATA/slice/skills.csv
    python CODE/graph.py skills_llm.csv     # skills from the LLM extractor
"""
import os
import sys

import certifi
from dotenv import load_dotenv
from neo4j import GraphDatabase, TrustCustomCAs

import pipeline

load_dotenv(pipeline.SLICE.parent.parent / ".env")
BATCH = 1000


def driver():
    # Aura's certificate chain is fine, but some Python builds do not trust its
    # root, so check it against certifi's roots instead of the system store.
    uri = os.environ["NEO4J_URI"].replace("neo4j+s://", "neo4j://")
    return GraphDatabase.driver(uri, auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
                                encrypted=True, trusted_certificates=TrustCustomCAs(certifi.where()))


def query(cypher, **params):
    with driver() as d:
        return d.execute_query(cypher, params, database_=os.getenv("NEO4J_DATABASE", "neo4j")).records


def skill_id(row):
    return row.get("canonical_id") or f'{row["vocab"]}:{row["skill"].lower()}'


def load(skills_file="skills.csv"):
    jobs, companies = pipeline.read("postings.csv"), pipeline.read("companies.csv")
    skills = [{**r, "skill_id": skill_id(r), "sent_id": int(r["sent_id"]), "confidence": float(r["confidence"] or 0)}
              for r in pipeline.skill_rows(skills_file) if r["sentence"] and r["span"]]

    with driver() as d, d.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as s:
        s.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS").consume()
        for label, key in (("Candidate", "id"), ("Job", "id"), ("Skill", "id"), ("Company", "id"), ("Location", "name")):
            s.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE").consume()

        def batches(cypher, rows):
            for i in range(0, len(rows), BATCH):
                s.run(cypher, rows=rows[i:i + BATCH]).consume()

        batches("UNWIND $rows AS r MERGE (c:Company {id: r.company_id}) "
                "SET c.name = r.name, c.size = r.company_size, c.city = r.city, c.state = r.state", companies)
        batches("UNWIND $rows AS r MERGE (j:Job {id: r.job_id}) "
                "SET j.title = r.title, j.salary = r.normalized_salary, j.views = r.views "
                "MERGE (l:Location {name: r.location}) MERGE (j)-[:LOCATED_IN]->(l) "
                "WITH j, r WHERE r.company_id <> '' "
                "MATCH (c:Company {id: r.company_id}) MERGE (j)-[:POSTED_BY]->(c)", jobs)
        batches("UNWIND $rows AS r MERGE (c:Candidate {id: r.resume_id}) SET c.category = r.category",
                pipeline.read("resumes.csv"))
        for doc_type, node, rel in (("resume", "Candidate", "HAS_SKILL"), ("job", "Job", "REQUIRES_SKILL")):
            batches("UNWIND $rows AS r MERGE (s:Skill {id: r.skill_id}) SET s.name = r.skill, s.source = r.vocab "
                    f"WITH s, r MATCH (d:{node} {{id: r.doc_id}}) "
                    f"CREATE (d)-[:{rel} {{doc_id: r.doc_id, sent_id: r.sent_id, sentence: r.sentence, "
                    "span: r.span, extractor: r.extractor, confidence: r.confidence}]->(s)",
                    [r for r in skills if r["doc_type"] == doc_type])
    return len(skills)


if __name__ == "__main__":
    n = load(*sys.argv[1:])
    counts = {r["label"]: r["n"] for r in query("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n")}
    edges = {r["type"]: r["n"] for r in query("MATCH ()-[e]->() RETURN type(e) AS type, count(*) AS n")}
    assert edges.get("HAS_SKILL", 0) + edges.get("REQUIRES_SKILL", 0) == n, (n, edges)
    assert not query("MATCH ()-[e:HAS_SKILL|REQUIRES_SKILL]->() WHERE e.span IS NULL OR e.sentence IS NULL RETURN e LIMIT 1")
    assert not query("MATCH (s:Skill) WHERE NOT (s)<--() RETURN s LIMIT 1")  # no orphan Skill
    print("loaded:", counts, edges)

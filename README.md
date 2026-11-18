# TraceHire

A LinkedIn Assistant built for CSE 573 Semantic Web Mining at Arizona State University, Fall 2026. We are Group 2, Project 3.

TraceHire is a career coach you can chat with. You give it your resume and it finds jobs that fit you, tells you which skills are worth learning next, looks up companies, and helps you draft a message to reach out.

We built it around one rule. Every skill the assistant lists has to point back to the exact sentence it came from. If it cannot show that sentence, the skill is dropped.

This repository holds the chat page, the agents behind it, the skill extractor, and the small data slice they run on.

## Why

Most career tools hand everyone the same tips. AI tools have a second problem, which is that they often list skills that are nowhere in your resume. We wanted advice you can check for yourself.

## How it works

1. A resume or job posting is split into sentences.
2. An LLM (`llama4-scout-17b` on the ASU Research Computing API) reads each sentence and pulls out the skills it mentions.
3. Each skill must quote the words it came from. A skill whose quote is not in the sentence, or does not name the skill, is thrown out.
4. Skill names are mapped to one skill list (O\*NET software names), so "Spark" and "Apache Spark" count as one skill.
5. When you ask something, a LangGraph orchestrator hands the question to the agent that fits it: job matching, skill advice, outreach, company research, or questions about your resume.
6. Each agent answers only from numbered sources (resume sentences, job postings, skill counts, company facts) and cites them. Citations that point nowhere are removed, and every cited source is listed in full under the answer.

Job matching is skill overlap between your resume and each posting, and skill advice is a count of the missing skills that the 100 closest postings ask for most.

## How to run

You need Python 3.10 or newer and a key for the ASU Research Computing API (or any OpenAI-compatible API).

```
pip install -r requirements.txt
cp .env.example .env          # then put your key in .env
streamlit run CODE/app.py --server.address localhost
```

Then:

* Upload a resume as PDF or text, or paste it. The LLM reads it in about 30 seconds.
* Or open `http://localhost:8501/?test=1` to pick one of the dataset's resumes, which are already read.
* Turn on "Compare with a plain LLM" to see the same question answered with no evidence, next to TraceHire's answer.

Without a key the page still opens and reads resumes with a simple word list, but it cannot answer questions.

The page has no login, so keep it on localhost. Uploaded resumes are sent to the LLM API, and the model's answers about their sentences are cached in `DATA/cache` on your machine.

## Folders

```
CODE/
  app.py            the Streamlit chat page
  agents.py         the LangGraph orchestrator and the agents
  pipeline.py       matching, skill advice, company profiles, the word-list extractor
  extraction/       the LLM skill extractor with the provenance check
  tests/            quick checks for the extraction code
  graph.py          loads the Neo4j knowledge graph from the skills file
  run_e1.py         E1, extraction against the hand labels
  run_e2.py         E2, job matching (TF-IDF, Sentence-BERT, skill overlap, hybrid)
  run_e3.py         E3, skill advice with one skill hidden
  judge_copy.py     makes and merges the E2 judging sheet
  make_slice.py     cuts the slice from the raw downloads
DATA/
  README.md         sources, licences, cleaning steps, file formats
  slice/            2,000 postings, 1,422 companies, 238 resumes, the skill list, the extracted skills
  ONTOLOGY.md       the knowledge graph schema
  gold/             50 hand labeled documents (the E1 answer key) and the labeling guide
  extracted/        the extractors' output on those 50 documents
EVALUATIONS/        E1, E2 and E3 results; see EVALUATIONS/README.md
.streamlit/         the look of the page
```

Checks, run from the repository root:

```
python CODE/pipeline.py
python CODE/tests/test_extraction.py
```

## Data

* LinkedIn Job Postings from Kaggle (A. Koneru), CC BY-SA 4.0.
* Resume Dataset from Kaggle (S. Bhawal), CC0.
* O\*NET 31.0 from the US Department of Labor, CC BY 4.0.

All three are public. We do not scrape LinkedIn, and the assistant never contacts real people. Email addresses and phone numbers in the data files are masked. See `DATA/README.md` for details and credit.

## Team

* Bharat Johnson: data and ontology
* Mathew Abraham Palliambil: extraction
* Rishikumar Senthilvel: graph and recommendation
* Basil Shibu Thomas: agents and interface

AI tools (Claude) were used for code and drafting.

# Data

The raw downloads are too big for GitHub, so this folder holds only the small slice the page runs on.

## Sources

* LinkedIn Job Postings by Arsh Koneru on Kaggle, version 13. About 124,000 postings, license CC BY-SA 4.0. https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
* Resume Dataset by S. Bhawal on Kaggle. 2,484 resumes in 24 job categories, license CC0. https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
* O\*NET 31.0 Database from the U.S. Department of Labor. We use the Software Skills file (31,821 rows), license CC BY 4.0. https://www.onetcenter.org/database.html

## What is in the slice folder

| File | Rows | What it is |
|---|---:|---|
| `postings.csv` | 2,000 | Job postings, picked at random from the 26,888 postings tagged Information Technology or Engineering that were left after cleaning |
| `companies.csv` | 1,422 | The companies behind those postings |
| `resumes.csv` | 238 | The resumes in the Information Technology (120) and Engineering (118) categories |
| `skills_vocab.csv` | 8,753 | Software names from O\*NET, each with its hot technology and in demand flags |
| `skills_llm.csv` | 50,305 | The skills the LLM extractor found in the resumes and postings. The page reads this file. |

Of the 2,000 postings, 594 (29.7%) have a `normalized_salary` value.

## How we cleaned it

1. Keep the postings that LinkedIn tagged IT or ENG.
2. Drop postings with a description shorter than 30 words.
3. Drop repeats, meaning postings with exactly the same title, company and description.
4. Pick 2,000 at random with a fixed seed (573).
5. Keep the resumes in the two categories and drop the ones with no text.
6. Mask email addresses and phone numbers in every file: letters and digits are replaced with `x` and `0`, so the text keeps its length and the character offsets below stay right.

## The skills format

Each row of `skills_llm.csv` is one skill found in one sentence.

* `doc_id` is the resume ID or the job ID, and `doc_type` is `resume` or `job`.
* `sent_id` is the number of the sentence inside the document, starting at 0.
* `sentence` is the full sentence the skill came from.
* `span` is the exact words in that sentence that named the skill. `span_start` and `span_end` say where they sit in the sentence.
* `skill` is the name of the skill, and `canonical_id` is its stable ID (`onet:...` when it is in the O\*NET list, `raw:...` when it is not).
* `vocab` says which skill list the name comes from (`onet` or `none`).
* `skill_class` is `tool`, `technique`, `domain` or `cert`. Certifications are shown apart and not counted as skills.
* `negated`, `proficiency` and `requirement` are filled when the sentence states them.
* `extractor`, `check`, `model` and `confidence` say how the row was produced.

A row with no sentence is not allowed. That is the rule the whole project is built on.

## Credit

`postings.csv` and `companies.csv` are a part of the LinkedIn Job Postings dataset by Arsh Koneru, and `skills_llm.csv` quotes sentences from it. These files stay under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). We changed them by selecting rows and columns and masking contact details.

`resumes.csv` comes from the Resume Dataset by S. Bhawal, released under CC0.

This folder includes information from the O\*NET 31.0 Database by the U.S. Department of Labor, Employment and Training Administration (USDOL/ETA). Used under the CC BY 4.0 license. O\*NET is a trademark of USDOL/ETA. We have modified some of this information. USDOL/ETA has not approved, endorsed, or tested these modifications.

# Evaluations

The results of our four experiments go in this folder.

* E1, extraction. We compare a word list, a plain LLM, and an LLM that must quote its source sentence. The code is ready (`python CODE/run_e1.py`, details in `CODE/extraction/README.md`). Results will appear in `e1/` once the gold documents are labeled.
* E2, matching. We compare TF-IDF, Sentence-BERT and a graph model. It runs on 20 resumes. `python CODE/run_e2.py` uses the word matcher's skills and writes `e2/results.md`. `python CODE/run_e2.py skills_llm.csv` uses the LLM's skills and writes `e2/results_skills_llm.md`. Both share one judging sheet, `e2/judging_sheet.csv`. A run adds the pairs it has not seen before, and `python CODE/judge_copy.py rishi` makes an easy copy of the pairs still to judge.
* E3, skill advice. We hide one real skill from a resume and check if the advice brings it back. `python CODE/run_e3.py` uses the word matcher's skills and writes `e3/results.md`. `python CODE/run_e3.py skills_llm.csv` uses the LLM's skills and writes `e3/results_skills_llm.md`.
* E4, end to end. We ask the same question to a plain LLM and to TraceHire and compare the answers.

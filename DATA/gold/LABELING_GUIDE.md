# Gold labeling guide (E1)

We label skills in 50 documents: 25 IT/Engineering resumes and 25 IT/engineering
job postings (listed in `selection.csv`). These labels are the answer key for E1,
so label what the text says, not what the systems suggest.

## Workflow

1. Open a sheet from `DATA/gold/sheets/<doc_id>.csv` (Excel, Numbers or Google Sheets).
   Each row is one sentence.
2. Fill the **skills** column. Separate skills with `;`.
   The **suggestions** column shows what our extractors found. It may be wrong or incomplete.
3. Save it as CSV (UTF-8) into `DATA/gold/labels/A/<doc_id>.csv` with the same file name.
   - Excel: File > Save As > "CSV UTF-8".
   - Numbers: File > Export To > CSV.
4. Second labeler for agreement (kappa): label the same document independently, without looking
   at the A file, and save to `DATA/gold/labels/B/<doc_id>.csv`. Aim for at least 10 documents.

Do not edit the `doc_id`, `sent_id` or `sentence` columns. Rows whose `section` is `header`
are section titles such as "Skills" or "Experience". Leave them empty.

Before you start, add your name next to the documents you take in `DATA/gold/assignments.md`,
so no one labels the same document twice by accident.

## What counts as a skill

Label it if the sentence says the person has it (resume) or the job asks for it (posting).

| Label | Examples |
|---|---|
| **Tools**: software, programming languages, libraries, platforms, systems | Python; Excel; AWS; SAP; AutoCAD; Kubernetes |
| **Techniques**: methods and practices | machine learning; ETL; regression analysis; budgeting; PLC programming |
| **Domain knowledge**: fields of knowledge | accounting; GAAP; network security; HVAC design; payroll |

Do **not** label:
- soft skills and traits: communication, teamwork, leadership, motivated, detail-oriented
- job titles, company names, school names, degrees ("B.S. in Civil Engineering")
- certifications and licenses (PMP, PE license). We may add these as a separate class later.
- vague phrases: "computer skills", "technical skills", "various software"
- skills that are negated: "no experience with Java", "Java not required"

Include skills with hedges such as "familiar with X", "exposure to X" and "X a plus".

## How to write each skill

- **Copy the words from the sentence** when possible: `Microsoft Excel`, not `Excel spreadsheets`.
  This lets the code find the exact span.
- One skill per entry. Split lists: `Python, SQL and Tableau` becomes `Python; SQL; Tableau`.
- When two skills share words, write each one in full:
  `hydraulic and hydrologic calculations` becomes `hydraulic calculations; hydrologic calculations`.
- If the same skill appears in several sentences, label each sentence where it appears.

## Hard cases

| Text | Label |
|---|---|
| "Go to market strategy" | nothing (not the Go language) |
| "Managed C-level relationships" | nothing (not the C language) |
| "Excel at problem solving" | nothing |
| "Built dashboards in Power BI" | Power BI |
| "SQL Server 2016" | SQL Server |

If you are unsure, leave a note in `DATA/gold/notes.md`. Don't guess, and we'll decide together.

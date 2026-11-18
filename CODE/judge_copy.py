"""Makes an easy-to-label copy of the E2 judging sheet for one judge, and merges
the filled copy back. The copy hides the other judge's column, so judging stays blind.

    python CODE/judge_copy.py rishi                  # writes judging_rishi.xlsx with the pairs rishi has not judged yet
    python CODE/judge_copy.py rishi --merge FILE     # FILE: the filled copy, as .xlsx or .csv

Open the .xlsx in Excel, Numbers or Google Sheets, put 1 (a reasonable fit) or 0
in the "fit" column, and save or download it. Then run the --merge line.
"""
import re
import sys

import pandas as pd
from openpyxl.styles import Font
from openpyxl.worksheet.datavalidation import DataValidation

import pipeline

ROOT = pipeline.SLICE.parent.parent
SHEET = ROOT / "EVALUATIONS" / "e2" / "judging_sheet.csv"
WIDTHS = {"resume": 28, "resume_skills": 30, "resume_start": 40, "job_title": 34, "company": 20,
          "location": 16, "fit": 6, "job_text": 60, "resume_id": 11, "job_id": 12}


def make(judge):
    sheet = pd.read_csv(SHEET, dtype=str).fillna("")
    sheet = sheet[sheet[f"judge_{judge}"] == ""].copy()  # a rerun of E2 can add pairs; only those are left to judge
    if sheet.empty:
        sys.exit(f"{judge} has judged every pair in the sheet.")
    texts = {r["resume_id"]: r["text"] for r in pipeline.read("resumes.csv")}
    sheet["resume"] = sheet.resume_title
    # The first lines after the title say what the person does, which helps when the title is vague.
    sheet["resume_start"] = sheet.resume_id.map(lambda r: " ".join(re.split(r"\s{2,}", texts[r].strip(), 1)[-1].split())[:300])
    sheet["job_text"] = sheet.job_text.str[:300]
    sheet["fit"] = sheet[f"judge_{judge}"]
    sheet["order"] = range(len(sheet))  # keeps the shuffled order inside each resume
    copy = sheet.sort_values(["resume_id", "order"])[list(WIDTHS)]
    path = ROOT / f"judging_{judge}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        copy.to_excel(xl, index=False, sheet_name="judge")
        ws = xl.sheets["judge"]
        ws.freeze_panes = "A2"
        for i, (col, width) in enumerate(WIDTHS.items(), 1):
            ws.column_dimensions[ws.cell(1, i).column_letter].width = width
            ws.cell(1, i).font = Font(bold=True)
        fit = ws.cell(1, list(WIDTHS).index("fit") + 1).column_letter
        dv = DataValidation(type="list", formula1='"1,0"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{fit}2:{fit}{len(copy) + 1}")
    print(f"wrote {path} with {len(copy)} rows for {judge}")


def merge(judge, filled):
    got = pd.read_excel(filled, dtype=str) if str(filled).endswith(".xlsx") else pd.read_csv(filled, dtype=str)
    got = got.fillna("")
    got["fit"] = got.fit.str.replace(r"\.0$", "", regex=True).str.strip()
    bad = got[~got.fit.isin(["", "0", "1"])]
    if len(bad):
        sys.exit(f"Only 1, 0 or empty is allowed in fit. Fix these rows:\n{bad[['resume_id', 'job_id', 'fit']]}")
    sheet = pd.read_csv(SHEET, dtype=str).fillna("")
    fits = dict(zip(zip(got.resume_id, got.job_id), got.fit))
    keys = list(zip(sheet.resume_id, sheet.job_id))
    unknown = set(fits) - set(keys)
    if unknown:
        sys.exit(f"{len(unknown)} pairs in {filled} are not in the sheet, for example {min(unknown)}. Nothing changed.")
    # Pairs that are not in the file keep the judgment they already have.
    sheet[f"judge_{judge}"] = [fits.get(k) or old for k, old in zip(keys, sheet[f"judge_{judge}"])]
    sheet.to_csv(SHEET, index=False)
    done = (sheet[f"judge_{judge}"] != "").sum()
    print(f"merged: {done} of {len(sheet)} pairs judged by {judge}, {sheet[f'judge_{judge}'].eq('1').sum()} marked 1")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("rishi", "basil"):
        sys.exit(__doc__)
    merge(sys.argv[1], sys.argv[3]) if "--merge" in sys.argv else make(sys.argv[1])


"""
Enrich the company dataset using raw LinkedIn company metadata.

Joins companies with their industries and specialities using company_id.
Multiple values are combined into semicolon-separated strings.

Input:  DATA/slice/companies.csv
Output: DATA/slice/companies_enriched.csv

The original companies.csv is not modified.
"""

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SLICE = ROOT / "DATA" / "slice"
RAW = ROOT / "DATA" / "raw" / "linkedin" / "companies"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def group_values(path, column):
    # Group unique values by company_id.
    grouped = defaultdict(set)

    for row in read_csv(path):
        company_id = row["company_id"]
        value = row[column].strip()

        if company_id and value:
            grouped[company_id].add(value)

    return grouped


def main():
    companies = read_csv(SLICE / "companies.csv")

    industries = group_values(
        RAW / "company_industries.csv", "industry"
    )
    specialities = group_values(
        RAW / "company_specialities.csv", "speciality"
    )

    enriched = []

    for company in companies:
        company_id = company["company_id"]

        enriched.append({
            **company,
            "industries": "; ".join(
                sorted(industries.get(company_id, set()))
            ),
            "specialities": "; ".join(
                sorted(specialities.get(company_id, set()))
            ),
        })

    output = SLICE / "companies_enriched.csv"
    fieldnames = list(companies[0].keys()) + [
        "industries",
        "specialities",
    ]

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)

    print(f"Original companies: {len(companies)}")
    print(f"Enriched companies: {len(enriched)}")
    print(
        "Companies with industries:",
        sum(bool(row["industries"]) for row in enriched),
    )
    print(
        "Companies with specialities:",
        sum(bool(row["specialities"]) for row in enriched),
    )
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
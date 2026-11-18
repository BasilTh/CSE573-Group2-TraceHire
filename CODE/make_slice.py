"""Cuts the small demo slice (DATA/slice) out of the raw downloads (DATA/raw).
DATA/README.md says where the raw files go. Run from the repo root:

    python CODE/make_slice.py
"""
import pandas as pd

RAW, OUT = "DATA/raw/", "DATA/slice/"

# TODO (Bharat): grow this to the 15,000 posting slice, drop the titles that the
# broad IT tag lets in by mistake (for example "Security Officer"), and take
# names and contact details out of the resumes.

tags = pd.read_csv(RAW + "linkedin/jobs/job_skills.csv").groupby("job_id").skill_abr.agg(set)
tags = tags[tags.map(lambda t: bool(t & {"IT", "ENG"}))]

posts = pd.read_csv(RAW + "linkedin/postings.csv", usecols=[
    "job_id", "title", "company_id", "company_name", "location", "description",
    "normalized_salary", "views", "applies", "formatted_experience_level"])
posts = posts[posts.job_id.isin(tags.index)]
posts = posts[posts.description.str.split().str.len() >= 30]
posts = posts.drop_duplicates(["title", "company_name", "description"])
print(len(posts), "IT or ENG postings after cleaning")
posts = posts.sample(2000, random_state=573)  # fixed seed, so everyone gets the same slice
posts["company_id"] = posts.company_id.astype("Int64")
posts["tags"] = posts.job_id.map(tags).map(lambda t: "|".join(sorted(t)))
posts.to_csv(OUT + "postings.csv", index=False)

companies = pd.read_csv(RAW + "linkedin/companies/companies.csv", usecols=[
    "company_id", "name", "description", "company_size", "city", "state", "country", "url"])
companies = companies[companies.company_id.isin(posts.company_id.dropna())]
companies.to_csv(OUT + "companies.csv", index=False)

resumes = pd.read_csv(RAW + "resumes/Resume/Resume.csv", usecols=["ID", "Resume_str", "Category"])
resumes.columns = ["resume_id", "text", "category"]
resumes = resumes[resumes.category.isin(["INFORMATION-TECHNOLOGY", "ENGINEERING"])]
resumes = resumes[resumes.text.str.strip().str.len() > 0]
resumes.to_csv(OUT + "resumes.csv", index=False)

# One row per software name. "max" of the Y/N flags gives Y if any occupation has Y.
vocab = pd.read_csv(RAW + "onet/software_skills.csv").groupby("Workplace Example").agg(
    category=("Element Name", "first"), hot=("Hot Technology", "max"),
    in_demand=("In Demand", "max"), occupations=("Title", "count"))
vocab.rename_axis("skill").to_csv(OUT + "skills_vocab.csv")

print(len(posts), "postings,", len(companies), "companies,", len(resumes), "resumes,", len(vocab), "skill names")

# TraceHire Knowledge Graph Ontology

## 1. Overview

TraceHire uses a Neo4j knowledge graph to represent candidates, job postings, skills, companies, and locations. The graph supports skill-based job matching and provides source evidence for extracted skills.

The graph schema and loading process are implemented in `CODE/graph.py`. Skill normalization and filtering are handled by `CODE/pipeline.py`, while the LLM-based extraction and provenance validation processes are documented in `CODE/extraction/README.md`.

The current ontology contains **five node types** and **four relationship types**.

## 2. Node Types

| Node Type | Unique Property | Additional Properties | Description |
|---|---|---|---|
| `Candidate` | `id` | `category` | Represents a candidate resume. |
| `Job` | `id` | `title`, `salary`, `views` | Represents a job posting. |
| `Skill` | `id` | `name`, `source` | Represents a normalized skill. |
| `Company` | `id` | `name`, `size`, `city`, `state` | Represents an employer. |
| `Location` | `name` | None currently loaded | Represents a job location. |

### 2.1 Candidate

A `Candidate` node represents one resume from the dataset.

- `id`: The resume's `resume_id`.
- `category`: The resume category, such as the category recorded in `resumes.csv`.

### 2.2 Job

A `Job` node represents one LinkedIn job posting.

- `id`: The posting's `job_id`.
- `title`: Job title.
- `salary`: Normalized salary from the posting.
- `views`: Number of views recorded in the dataset.

### 2.3 Skill

A `Skill` node represents a normalized skill extracted from a resume or job posting.

- `id`: Canonical skill identifier, when available; otherwise, a vocabulary-prefixed lowercase skill name.
- `name`: Normalized skill name.
- `source`: Vocabulary recorded for the skill.

The same `Skill` node can be connected to multiple candidates and jobs.

### 2.4 Company

A `Company` node represents an employer.

- `id`: The company's `company_id`.
- `name`: Company name.
- `size`: The `company_size` value from the source dataset.
- `city`: Company city.
- `state`: Company state.

The `size` field preserves the source value; the current loader does not convert it into an exact employee count.

### 2.5 Location

A `Location` node represents the location associated with a job posting.

- `name`: The posting's location string.

Locations are identified by their name strings rather than geographic coordinates.

## 3. Relationship Types

| Source Node | Relationship | Target Node | Description |
|---|---|---|---|
| `Candidate` | `HAS_SKILL` | `Skill` | Connects a resume to a skill extracted from it. |
| `Job` | `REQUIRES_SKILL` | `Skill` | Connects a job posting to a skill extracted from its description. |
| `Job` | `POSTED_BY` | `Company` | Associates a job posting with its company. |
| `Job` | `LOCATED_IN` | `Location` | Associates a job posting with its location. |

### 3.1 Relationship Directions

```text
(Candidate)-[:HAS_SKILL]->(Skill)

(Job)-[:REQUIRES_SKILL]->(Skill)

(Job)-[:POSTED_BY]->(Company)

(Job)-[:LOCATED_IN]->(Location)
```

### 3.2 Relationship Semantics

**HAS_SKILL**

Represents a skill mention extracted from a candidate's resume. It does not independently verify the candidate's proficiency.

**REQUIRES_SKILL**

Represents a skill mention extracted from a job posting. Although the relationship is named `REQUIRES_SKILL`, the current graph loader does not distinguish mandatory qualifications from preferred skills.

**POSTED_BY**

Connects a job to a company when the job contains a nonempty `company_id` that matches a loaded company.

**LOCATED_IN**

Connects a job to a location identified by the posting's location string.

### 3.3 Multiple Skill Mentions

Skill relationships are created using Cypher `CREATE`, rather than `MERGE`.

Consequently, multiple mentions of the same skill in one document can produce multiple relationships between the same candidate or job and the same skill node.

This preserves individual mentions and their evidence.

## 4. Skill Provenance

Every `HAS_SKILL` and `REQUIRES_SKILL` relationship contains provenance information identifying the source of the extracted skill.

| Property | Description |
|---|---|
| `doc_id` | Identifier of the source resume or job posting. |
| `sent_id` | Sentence index in the source document. |
| `sentence` | Source sentence containing the extracted skill. |
| `span` | Text span identifying the skill mention. |
| `extractor` | Extraction method used to identify the skill. |
| `confidence` | Confidence value supplied by the extraction process. |

### 4.1 Why Provenance Is Stored on Relationships

A skill such as Python can appear in many different resumes and job postings.

Storing provenance on the relationship allows each mention to retain its own sentence, text span, and extraction information while referring to the same normalized `Skill` node.

### 4.2 Graph Loader Requirements

Before loading a skill relationship, `CODE/graph.py` requires:

- A nonempty `sentence`.
- A nonempty `span`.

The loader converts `sent_id` to an integer and `confidence` to a floating-point value, using zero when the confidence field is empty.

Records missing a sentence or span are not loaded as skill relationships.

## 5. Skill Classification

The LLM extraction pipeline categorizes skills using three classes:

| Skill Class | Description |
|---|---|
| `tool` | Tools and technologies. |
| `technique` | Methods or technical practices. |
| `domain` | Areas of knowledge or expertise. |

The shared pipeline also recognizes `skill_class = cert` as an exclusion category for certifications.

### 5.1 Certification Handling

The `same_names()` function in `CODE/pipeline.py` excludes rows whose `skill_class` equals `cert`.

Because `graph.py` loads skills through `pipeline.skill_rows()`, these records are excluded from the graph's skill relationships.

Certifications are not represented as a separate Neo4j node type in the current ontology.

### 5.2 Skill Classes in Neo4j

The current graph uses a single `Skill` node type for qualifying extracted skills.

Although the extraction output includes `skill_class`, `CODE/graph.py` does not currently store it as a property of the `Skill` node or skill relationship.

Therefore, skill classes belong to the extraction schema rather than the persisted graph schema.

## 6. Skill Normalization and Identity

The pipeline uses canonical skill identifiers to group different names referring to the same skill.

### 6.1 Skill Identifier

The `skill_id()` function in `CODE/graph.py` uses:

1. The record's `canonical_id`, if present and nonempty.
2. Otherwise, the vocabulary and lowercase skill name in the format `vocab:skill`.

The LLM extraction documentation describes canonical identifiers such as:

- `onet:...` for mapped vocabulary skills.
- `raw:...` for skills not yet included in the vocabulary.

### 6.2 Canonical Names

The `same_names()` function in `CODE/pipeline.py` uses canonical identifiers to assign a common skill name to equivalent records.

For example, the implementation documentation describes `Agile` and `agile methodology` as names that can count as one skill when they share a `canonical_id`.

This normalization allows matching to compare canonical skill names instead of treating every surface form as a separate skill.

### 6.3 Vocabulary Source

The `Skill.source` property is populated from the extracted record's `vocab` field.

The LLM extraction documentation states that recognized vocabulary skills use their O*NET names, while unmatched skills retain the LLM's name with `vocab = none`.

## 7. LLM Extraction and Provenance Validation

The LLM extraction process is documented in `CODE/extraction/README.md`.

It operates on numbered sentences produced by `pipeline.split_sentences`.

The LLM identifies skill mentions and fills extraction fields, including:

- Skill name and supporting quote.
- Skill class.
- Negation status.
- Proficiency, when stated.
- Required or preferred status for job postings.

### 7.1 Provenance Validation Rules

An LLM extraction is accepted only when both conditions are satisfied:

1. **Sentence evidence:** The quoted text occurs in the identified source sentence, using whole-word matching that ignores capitalization and extra whitespace.
2. **Skill evidence:** The quote identifies the extracted skill, either through matching skill words or through a shared canonical vocabulary mapping.

If the quoted evidence appears in exactly one other sentence within the batch, the pipeline can correct the sentence ID and mark the extraction `sent_id_corrected`.

Extractions failing these checks are rejected and logged.

### 7.2 Negation Handling

The LLM extraction pipeline removes negated mentions, such as "no Java experience," before producing its final skill output.

However, `pipeline.same_names()` only explicitly excludes certifications, and `graph.py` does not independently check the `negated` field.

Therefore, negation filtering depends on the extraction process used to produce the input file.

### 7.3 Additional Extraction Fields

The shared extraction format includes the following fields beyond the core provenance columns:

| Field | Meaning |
|---|---|
| `canonical_id` | Stable identifier for the extracted skill. |
| `span_start` | Start position of the span in the sentence. |
| `span_end` | End position of the span in the sentence. |
| `skill_class` | Skill classification. |
| `negated` | Indicates a negated mention. |
| `proficiency` | `basic`, `intermediate`, or `advanced`, when stated. |
| `requirement` | `required` or `preferred` for job postings. |
| `check` | Provenance validation or extraction-rule status. |
| `model` | Model used for extraction. |

These fields are available in the extraction output, but the current Neo4j loader persists only the canonical identifier through `Skill.id` and the six relationship provenance properties described in Section 4.

## 8. Constraints and Validation

### 8.1 Uniqueness Constraints

The graph loader creates Neo4j uniqueness constraints for the following properties:

| Node Label | Unique Property |
|---|---|
| `Candidate` | `id` |
| `Job` | `id` |
| `Skill` | `id` |
| `Company` | `id` |
| `Location` | `name` |

Nodes are created or matched using these identifiers.

### 8.2 Post-Load Validation

After loading, `CODE/graph.py` performs three checks:

**Skill relationship count**

The combined number of `HAS_SKILL` and `REQUIRES_SKILL` relationships must equal the number of eligible skill records processed by the loader.

**Provenance completeness**

No skill relationship may have a null `sentence` or `span`.

**Orphan skill detection**

Every `Skill` node must have at least one incoming relationship.

These checks validate the loaded graph structure. They do not independently establish that every extracted skill is semantically correct.

## 9. Graph Loading and Data Sources

The graph loader reads the demo dataset from `DATA/slice`.

| Input File | Graph Content |
|---|---|
| `postings.csv` | Job nodes, location nodes, and job-company relationships. |
| `companies.csv` | Company nodes. |
| `resumes.csv` | Candidate nodes. |
| `skills.csv` | Default skill nodes and relationships. |
| `skills_llm.csv` | Alternative LLM-extracted skill input. |

The default graph-loading command uses `skills.csv`.

The loader also accepts `skills_llm.csv` as an alternative input.

## 10. Current Scope and Limitations

The current ontology supports:

- Five node types and four relationship types.
- Canonical skill identification and normalization.
- Multiple evidence-bearing skill relationships.
- Per-mention provenance.
- Certification exclusion.
- Node uniqueness constraints.
- Structural validation after loading.

Current limitations include:

- Skill classes are not persisted in Neo4j.
- Proficiency and required/preferred status are not persisted in the graph.
- The graph loader does not independently filter negated mentions.
- Job skill relationships do not distinguish required from preferred qualifications.
- Company industry and speciality enrichment fields are not loaded.
- O*NET occupation mappings are not represented in the current graph schema.
- Locations are stored as strings rather than coordinates.

These limitations describe the current implementation, not necessarily the intended final capabilities of the project.

## 11. Implementation References

| File | Responsibility |
|---|---|
| `CODE/graph.py` | Neo4j node creation, relationships, constraints, and validation. |
| `CODE/pipeline.py` | Skill normalization, certification filtering, and matching support. |
| `CODE/extraction/README.md` | LLM extraction, skill classes, provenance validation, and output format. |
| `DATA/README.md` | Dataset descriptions and shared data format. |
| `DATA/slice/skills.csv` | Default extracted-skill dataset. |
| `DATA/slice/skills_llm.csv` | LLM-extracted skill dataset. |
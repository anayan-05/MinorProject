# PathPilot Review NLP — Dataset Provenance & Licence Register

Owner: Anayan (`feature/nlp-model`)  
Last updated: 2026-11-25

## 1. Registered Datasets
All raw datasets are stored outside Git under `data/raw/nlp/` (ignored via `.gitignore`). Exact checksums, URLs, and versions are recorded in [`docs/nlp/dataset_manifest.csv`](dataset_manifest.csv).

| Dataset ID | Source & Citation | Terms / Licence Caveat | Domain & Intended Use |
|---|---|---|---|
| `semeval2014_rest_train` | SemEval-2014 Task 4 Restaurants Train v2 (`alexcadillon/SemEval2014Task4`, Pontiki et al. 2014) | Research shared-task terms; assess restrictions before commercial deployment | Benchmark & transfer ABSA training/validation only. **NOT** Kolkata travel-safety ground truth. |
| `semeval2014_rest_test_gold` | SemEval-2014 Task 4 Restaurants Test Gold | Research shared-task terms | Held-out benchmark test evaluation (kept untouched during training). |
| `semeval2014_rest_trial` | SemEval-2014 Task 4 Restaurants Trial | Research shared-task terms | Quick parser/schema smoke inspection. |
| `semeval2014_laptop_train` / `test_gold` | SemEval-2014 Task 4 Laptops Train v2 & Test Gold | Research shared-task terms | Aspect-term benchmark reference only (contains `aspectTerms`, no `aspectCategories`). |

## 2. Optional Future Datasets (Pending Verification / Collection)
- **SemEval-2016 Task 5 ABSA** (`alt.qcri.org/semeval2016/task5/`): Not yet downloaded; requires checking per-language README licence constraints before use.
- **Local Kolkata Travel-Safety Pilot Snippets (100–200 snippets)**: Must come only from permitted public sources or consented manual contributor snippets (never from unauthorized Google Maps scraping), anonymised and double-annotated per [`docs/nlp/annotation_guide.md`](annotation_guide.md). No local Kolkata snippets are fabricated or claimed in this repository.

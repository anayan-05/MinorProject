# PathPilot Review ABSA — Model Card

Version: `pathpilot-distilroberta-base-absa-v1.0.0`  
Owner: Anayan (`feature/nlp-model`)

## 1. Intended Use
- **Task:** Aspect-conditioned 3-class sentiment polarity classification (`negative`, `neutral`, `positive`) on anonymised review snippets.
- **Role in PathPilot:** Supplies structured perception evidence (`aspect`, `sentiment`, `confidence`, `evidence_text`, `source_date`, `model_version`, `limitations`) to Amisa's geospatial layer and Deep's evidence UI.
- **Out-of-Scope / Non-Goals:** Must **never** compute a route safety score, claim guaranteed safety, or treat subjective reviews as verified crime/incident ground truth.

## 2. Architecture & Training Summary
- **Backbone:** `distilroberta-base` (`RobertaForSequenceClassification`, RoBERTa BPE tokeniser, `max_length=96`)
- **Input Format:** Sequence pair `("Aspect: <aspect>", "<anonymised review text>")`
- **Training Regime:** 10 epochs, AdamW (`lr=5e-5`, linear warmup 10% + decay), inverse-sqrt class weights, grouped split by sentence ID with near-duplicate leakage assertions.
- **Benchmark Performance (SemEval-2014 Task 4 Restaurants Gold Test, n=973):**
  - Accuracy: `0.8746` | Macro-F1: `0.7985` | Weighted-F1: `0.8766`
  - CPU Smoke Latency: `155.43 ms`

## 3. Critical Limitations & Domain Gap
1. **Domain Transfer Gap:** Current weights are trained on **SemEval-2014 Task 4 Restaurants** (English restaurant reviews), NOT annotated Kolkata travel-safety reviews. Before relying on predictions for Kolkata corridors, fine-tune/evaluate on the double-annotated Kolkata pilot set (`templates/review_annotation_template.csv`).
2. **Mixed-Aspect Sentences:** Multi-clause reviews expressing opposite polarity for different aspects can cause polarity bleed across aspects.
3. **Neutral Class Sensitivity:** `neutral` test F1 (`0.6432`) trails `positive` (`0.9275`) and `negative` (`0.8248`) due to smaller support (`94/973`) and subtle phrasing.
4. **Privacy & Ethics:** All inputs are passed through PII anonymisation (`ml/nlp/preprocess.py`). Do not ingest raw un-anonymised reviews or scrape prohibited platforms.

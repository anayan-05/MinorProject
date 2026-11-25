# ANAYAN — MASTER PROMPT: PROJECT LEAD + REVIEW NLP
Work on `feature/nlp-model`, paths `ml/nlp/`, `docs/nlp/`, NLP tests. Read shared context and handbook.

History/context: PathPilot (formerly SafeRoam concept) is a Kolkata-first women-traveller safety app integrating explainable geospatial route evidence, review NLP, optional user-authorised audio, alternative routes, emergency resources and confirmed SOS. The target is a fully working Android prototype installed over USB. Five Antigravity agents work separately; GitHub is shared source of truth. Deep waits for design references only for final UI styling.

Your ownership: review aspect/sentiment intelligence. Candidate aspects: lighting, crowd_activity, police_presence, transport_access, road_condition, isolation, commercial_activity, general_environment. You do not calculate final route safety score.

Data: obtain official SemEval-2014 Task 4 ABSA; optionally SemEval-2016 Task 5 if official release and terms are verified. These are benchmark/transfer datasets, not Kolkata travel-safety truth. For local data, use permitted sources or consented/manual snippets; do not indiscriminately scrape Google Maps or bypass terms. Anonymise identifiers; record source, terms, date and location ID.

Annotation: create guidelines; pilot 100–200 snippets; double-annotate a subset; calculate agreement and adjudicate. Polarity positive/negative/neutral. Use `templates/review_annotation_template.csv`. Split by location/source where feasible and prevent near-duplicate leakage.

Mandatory notebook: `ml/nlp/training/pathpilot_review_absa.ipynb`, fresh-Colab runnable. Include setup/versions/seeds; configurable Drive path; schema validation; privacy cleaning; duplicates; label distribution; grouped split/leakage assertions; RoBERTa tokenisation; model training; held-out macro-F1, per-class precision/recall/F1, confusion matrix; error analysis; save/reload artefacts; inference demo; CPU smoke test. Start with aspect-category/polarity classifier; BIO extraction only if span labels/data justify it. Never invent metrics; keep test untouched.

Deliver scripts `preprocess.py`, `train.py`, `inference.py`, labels map, annotation guide, provenance/licence register, model card, API contract, mock fixture and evaluation report. Inference returns aspect/sentiment/confidence/evidence/source date/model version/limitations, not safety score. Coordinate schema with Amisa and fixture with Deep. No datasets/weights/reviews/secrets in Git. Test, diff, commit and PR on your branch.

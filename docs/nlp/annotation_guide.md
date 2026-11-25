# PathPilot Review NLP — Annotation Guide

Version: 1.0.0  
Owner: Anayan (`feature/nlp-model`)

## 1. Purpose & Epistemic Rules
PathPilot extracts **aspect-based perceptions** from permitted or consented local review snippets.  
- Reviews represent **subjective perceptions**, never verified crime/safety truth.
- NLP outputs (`aspect`, `sentiment`, `confidence`, `evidence_text`, `source_date`, `model_version`, `limitations`) feed Amisa's geospatial evidence layer and Deep's evidence UI. NLP **never** computes a final route safety score.
- Do **not** indiscriminately scrape Google Maps or bypass platform terms of service. Use only permitted sources or consented manual contributions.

## 2. Canonical Schema (`templates/review_annotation_template.csv`)
Every annotated row must match:
`review_id,location_id,text,aspect,sentiment,evidence_span_start,evidence_span_end,source_id,collection_date,anonymised,adjudication_status`

- `review_id`: Unique ID (e.g., `kol_rev_0001`).
- `location_id`: Stable geospatial area/segment/POI identifier coordinated with Amisa (e.g., `kmc_ward_063_park_street`).
- `text`: Anonymised snippet text (PII removed before saving).
- `aspect`: One of the 8 canonical PathPilot aspects below.
- `sentiment`: One of `positive`, `negative`, `neutral`.
- `evidence_span_start`, `evidence_span_end`: Character offsets (0-indexed, half-open `[start, end)`) in `text` for the supporting phrase, or `-1` if sentence-level without span.
- `source_id`: Registered source identifier matching `docs/nlp/dataset_manifest.csv`.
- `collection_date`: ISO 8601 date (`YYYY-MM-DD`) when collected/contributed.
- `anonymised`: `true` / `false` (must be `true` prior to training/export).
- `adjudication_status`: `single`, `double_agreed`, `adjudicated`, `rejected`.

## 3. Target Aspects (8 Categories)
| Aspect | Definition | Example Cues (Perception) |
|---|---|---|
| `lighting` | Streetlights, footpath visibility, dark corners | "well lit near the metro gate", "pitch dark stretch after 9 pm" |
| `crowd_activity` | Pedestrian/commuter presence, bustle vs emptiness | "busy with office commuters", "hardly anyone around" |
| `police_presence` | Visible police booth, traffic constable, patrol van | "traffic police booth at the crossing", "never see patrol here" |
| `transport_access` | Metro, bus stand, auto/cab availability | "easy autos right outside exit", "no buses after 8:30" |
| `road_condition` | Footpath width, potholes, waterlogging, encroachments | "broken pavement forces walking on road", "wide paved sidewalk" |
| `isolation` | Deserted lanes, dead-ends, lack of open exits | "long isolated wall stretch with no exits" |
| `commercial_activity` | Open shops, pharmacies, tea stalls, markets | "shops stay open late around Gariahat", "everything shutters early" |
| `general_environment` | Overall comfort/unease perception when no single aspect fits | "felt comfortable walking here in the evening" |

## 4. Polarity Definitions
- `positive`: Indicates favourable, reassuring, well-lit, accessible, or active perception for that aspect.
- `negative`: Indicates unfavourable, dark, isolated, obstructed, or concerning perception for that aspect.
- `neutral`: Descriptive mention of the aspect without clear positive or negative safety-relevant polarity, or balanced mixed cue.

Note on `isolation`: "Very isolated / deserted" is `negative` (risk-related isolation perception), while "not isolated / well-connected" is `positive`.

## 5. Privacy & PII Anonymisation Rules
Before storing or annotating snippets:
1. Replace personal names (including `Mr./Ms./Dr.` patterns), phone numbers, emails, `@usernames`, URLs, and personal IDs with `[REDACTED_NAME]`, `[REDACTED_PHONE]`, `[REDACTED_EMAIL]`, `[REDACTED_HANDLE]`, `[REDACTED_URL]`.
2. Remove exact home/apartment numbers or private personal trajectories.
3. Verify `anonymised=true` via `ml/nlp/preprocess.py` privacy checks.

## 6. Pilot Protocol, Double Annotation & Split Hygiene
1. **Pilot batch**: Annotate 100–200 permitted/consented snippets across diverse Kolkata wards/corridors.
2. **Double annotation**: At least 20–30% of the pilot subset is independently annotated by two annotators.
3. **Agreement & adjudication**: Compute Cohen's kappa ($\kappa$) for `aspect` and `sentiment`; resolve disagreements through adjudication (`adjudication_status=adjudicated`) and freeze guideline clarifications.
4. **Leakage-free splitting**: Group splits by `location_id` (and `source_id` where feasible) so corridors/locations in test never leak into train/val, and enforce exact + near-duplicate text checks across splits.

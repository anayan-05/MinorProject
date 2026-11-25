# PathPilot Review NLP — API Contract (`/api/v1/reviews/analyse`)

Version: `1.0.0`  
Owner: Anayan (`feature/nlp-model`)  
Consumers: Amisa (`geospatial/`), Deep (`mobile/`), Ayushman (`navigation/`)

## 1. Design Rules
- Base path: `/api/v1`
- Format: JSON, UTF-8
- Coordinates (if `location` is supplied): WGS84 `[longitude, latitude]`
- Distances: metres; Timestamps: ISO 8601 with timezone (`Z` or `+05:30`)
- Explicit unknowns: `null` or `"unknown"`, never silently `0`
- **No safety score calculation**: This endpoint returns perceived aspect/sentiment evidence only.
- Mock flag: `is_mock` must be `true` for fixture/fallback responses and `false` for genuine model inference.

## 2. Endpoint: `POST /api/v1/reviews/analyse`

### Request Schema
```json
{
  "schema_version": "1.0.0",
  "location_id": "kmc_ward_063_park_street",
  "coordinates": [88.3516, 22.5535],
  "reviews": [
    {
      "review_id": "rev_001",
      "text": "The main avenue near Park Street metro is brightly lit with open shops until 10 PM.",
      "source_id": "consented_pilot",
      "source_date": "2026-11-10T18:30:00+05:30",
      "candidate_aspects": ["lighting", "commercial_activity", "transport_access"]
    }
  ]
}
```
- `candidate_aspects` is optional. If omitted, the service evaluates the canonical PathPilot aspects (or configured model aspect set) and returns detected/evaluated aspects.

### Response Schema
```json
{
  "schema_version": "1.0.0",
  "generated_at": "2026-11-25T16:30:00+00:00",
  "location_id": "kmc_ward_063_park_street",
  "model_version": "pathpilot-roberta-absa-v1.0.0",
  "model_domain": "semeval2014_transfer_benchmark",
  "is_mock": false,
  "results": [
    {
      "review_id": "rev_001",
      "location_id": "kmc_ward_063_park_street",
      "aspect": "lighting",
      "sentiment": "positive",
      "confidence": 0.9124,
      "evidence_text": "The main avenue near Park Street metro is brightly lit with open shops until 10 PM.",
      "source_id": "consented_pilot",
      "source_date": "2026-11-10T18:30:00+05:30",
      "anonymised": true,
      "model_version": "pathpilot-roberta-absa-v1.0.0",
      "limitations": [
        "Review-derived aspect/sentiment reflects subjective perceptions, not verified safety facts.",
        "Does not compute or guarantee route safety."
      ]
    }
  ],
  "warnings": [
    "Reviews are subjective perceptions and may be sparse or dated; verify source_date before use."
  ]
}
```

## 3. Field Guarantees
- `sentiment`: `"positive" | "negative" | "neutral"`
- `confidence`: `float` in `[0.0, 1.0]` (softmax probability of predicted sentiment)
- `source_date`: ISO 8601 string or `null` if unknown
- `limitations`: Non-empty list of explicit epistemic caveats
- `is_mock`: Boolean (`true` for mock fixture / heuristic fallback, `false` for trained RoBERTa inference)

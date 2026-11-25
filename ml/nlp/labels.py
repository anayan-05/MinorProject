"""Canonical aspect and polarity label definitions for PathPilot Review NLP.

Coordinates aspect names with Amisa (geospatial) and Deep (mobile UI).
Explicitly separates PathPilot Kolkata travel-safety aspects from
SemEval-2014 Task 4 / SemEval-2016 Task 5 benchmark/transfer categories.
"""

from __future__ import annotations

from typing import Dict, List

SCHEMA_VERSION = "1.0.0"

# Canonical PathPilot travel-safety aspects (Handbook §5 & §11)
PATHPILOT_ASPECTS: List[str] = [
    "lighting",
    "crowd_activity",
    "police_presence",
    "transport_access",
    "road_condition",
    "isolation",
    "commercial_activity",
    "general_environment",
]

PATHPILOT_ASPECT_DESCRIPTIONS: Dict[str, str] = {
    "lighting": "Perceived street/footpath/station lighting visibility at the relevant time.",
    "crowd_activity": "Perceived presence of pedestrians, commuters, or bystanders.",
    "police_presence": "Perceived visibility or responsiveness of police/traffic/patrol personnel.",
    "transport_access": "Perceived availability/accessibility of metro, bus, auto, taxi, or transit stops.",
    "road_condition": "Perceived footpath/road surface state, drainage, construction, or obstacles.",
    "isolation": "Perceived desertedness, dead-end stretches, or lack of active exits.",
    "commercial_activity": "Perceived activity of shops, stalls, markets, or eateries along the stretch.",
    "general_environment": "Overall environmental atmosphere/perception not tied to a narrower aspect.",
}

# Canonical 3-class sentiment polarities
POLARITIES: List[str] = ["negative", "neutral", "positive"]
POLARITY_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(POLARITIES)}
ID_TO_POLARITY: Dict[int, str] = {idx: label for idx, label in enumerate(POLARITIES)}

# SemEval-2014 Task 4 Restaurant aspect categories (benchmark/transfer only)
SEMEVAL_2014_RESTAURANT_ASPECTS: List[str] = [
    "food",
    "service",
    "price",
    "ambience",
    "anecdotes/miscellaneous",
]

SEMEVAL_BENCHMARK_DISCLAIMER = (
    "SemEval-2014 Task 4 / SemEval-2016 Task 5 datasets are restaurant/laptop ABSA "
    "benchmarks used for transfer verification only; they are NOT Kolkata "
    "travel-safety ground truth. Reviews reflect subjective perceptions, not verified facts."
)

DEFAULT_LIMITATIONS: List[str] = [
    "Review-derived aspect/sentiment reflects subjective perceptions, not verified safety facts.",
    "Does not compute or guarantee route safety.",
    "Sparse local Kolkata travel-safety coverage reduces out-of-domain reliability until local annotated data is added.",
]

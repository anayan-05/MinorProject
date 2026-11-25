"""Preprocessing, privacy anonymisation, SemEval-2014 parsing, and leakage-safe splitting.

Owner: Anayan (feature/nlp-model)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import defusedxml.ElementTree as ET
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from ml.nlp.labels import (
    PATHPILOT_ASPECTS,
    POLARITIES,
    SEMEVAL_2014_RESTAURANT_ASPECTS,
)

REQUIRED_ANNOTATION_COLUMNS: List[str] = [
    "review_id",
    "location_id",
    "text",
    "aspect",
    "sentiment",
    "evidence_span_start",
    "evidence_span_end",
    "source_id",
    "collection_date",
    "anonymised",
    "adjudication_status",
]

VALID_ADJUDICATION_STATUSES: Set[str] = {
    "single",
    "double_agreed",
    "adjudicated",
    "rejected",
    "benchmark_gold",
}

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,30}\b")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,5}\)?[\s.-]?)?\d{5}[\s.-]?\d{5}\b"
    r"|(?:\+91[\s-]?)?[6-9]\d{9}\b"
)
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_AADHAAR_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
_TITLED_NAME_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr|Shri|Smt)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
)


def anonymise_text(text: str) -> Tuple[str, bool]:
    """Redact common PII patterns from review text.

    Returns:
        (cleaned_text, was_modified)
    """
    if not isinstance(text, str):
        return "", False
    cleaned = text
    cleaned = _EMAIL_RE.sub("[REDACTED_EMAIL]", cleaned)
    cleaned = _URL_RE.sub("[REDACTED_URL]", cleaned)
    cleaned = _HANDLE_RE.sub("[REDACTED_HANDLE]", cleaned)
    cleaned = _PHONE_RE.sub("[REDACTED_PHONE]", cleaned)
    cleaned = _PAN_RE.sub("[REDACTED_ID]", cleaned)
    cleaned = _AADHAAR_RE.sub("[REDACTED_ID]", cleaned)
    cleaned = _TITLED_NAME_RE.sub("[REDACTED_NAME]", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, (cleaned != text.strip())


def normalize_text(text: str) -> str:
    """Normalize text for exact/near-duplicate leakage checks."""
    lowered = text.lower().strip()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def char_ngrams(text: str, n: int = 4) -> Set[str]:
    """Return character n-grams of normalized text for Jaccard similarity."""
    norm = normalize_text(text)
    if len(norm) < n:
        return {norm} if norm else set()
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    """Compute Jaccard similarity between two n-gram sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def validate_annotation_df(
    df: pd.DataFrame,
    allowed_aspects: Sequence[str] = PATHPILOT_ASPECTS,
    require_anonymised: bool = True,
) -> List[str]:
    """Validate DataFrame schema and field constraints against the annotation template.

    Returns a list of validation error strings (empty list if valid).
    """
    errors: List[str] = []
    missing_cols = [c for c in REQUIRED_ANNOTATION_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return errors

    if df.empty:
        errors.append("Dataset is empty.")
        return errors

    if df["review_id"].isna().any() or (df["review_id"].astype(str).str.strip() == "").any():
        errors.append("Found empty review_id values.")

    if df["location_id"].isna().any() or (df["location_id"].astype(str).str.strip() == "").any():
        errors.append("Found empty location_id values.")

    if df["text"].isna().any() or (df["text"].astype(str).str.strip() == "").any():
        errors.append("Found empty text values.")

    bad_aspects = set(df["aspect"].dropna().astype(str)) - set(allowed_aspects)
    if bad_aspects:
        errors.append(f"Invalid aspect values: {sorted(bad_aspects)}")

    bad_sentiments = set(df["sentiment"].dropna().astype(str)) - set(POLARITIES)
    if bad_sentiments:
        errors.append(f"Invalid sentiment values: {sorted(bad_sentiments)}")

    bad_status = set(df["adjudication_status"].dropna().astype(str)) - VALID_ADJUDICATION_STATUSES
    if bad_status:
        errors.append(f"Invalid adjudication_status values: {sorted(bad_status)}")

    if require_anonymised:
        anon_vals = df["anonymised"].astype(str).str.lower().isin({"true", "1", "yes"})
        if not anon_vals.all():
            errors.append(f"{(~anon_vals).sum()} rows are not marked anonymised=true.")

    for idx, row in df.iterrows():
        text = str(row["text"])
        start = int(row["evidence_span_start"])
        end = int(row["evidence_span_end"])
        if start == -1 and end == -1:
            continue
        if start < 0 or end <= start or end > len(text):
            errors.append(
                f"Invalid evidence span [{start}, {end}) at row index {idx} (len={len(text)})."
            )
            if len(errors) >= 10:
                break

    return errors


def deduplicate_records(
    df: pd.DataFrame,
    text_col: str = "text",
    aspect_col: str = "aspect",
) -> Tuple[pd.DataFrame, int]:
    """Drop exact normalized (text, aspect) duplicates, preserving first occurrence."""
    work = df.copy()
    work["_norm_text"] = work[text_col].astype(str).map(normalize_text)
    before = len(work)
    work = work.drop_duplicates(subset=["_norm_text", aspect_col], keep="first").reset_index(
        drop=True
    )
    removed = before - len(work)
    work = work.drop(columns=["_norm_text"])
    return work, removed


def find_cross_split_near_duplicates(
    split_a: Sequence[str],
    split_b: Sequence[str],
    threshold: float = 0.90,
) -> List[Tuple[int, int, float]]:
    """Find near-duplicate pairs across two text lists using char 4-gram Jaccard."""
    grams_a = [char_ngrams(t) for t in split_a]
    grams_b = [char_ngrams(t) for t in split_b]
    matches: List[Tuple[int, int, float]] = []
    for i, ga in enumerate(grams_a):
        if not ga:
            continue
        for j, gb in enumerate(grams_b):
            if not gb:
                continue
            # Fast length-ratio bound before computing Jaccard
            if min(len(ga), len(gb)) / max(len(ga), len(gb)) < threshold:
                continue
            sim = jaccard_similarity(ga, gb)
            if sim >= threshold:
                matches.append((i, j, sim))
    return matches


def remove_cross_split_text_leakage(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    text_col: str = "text",
    threshold: float = 0.90,
) -> Tuple[pd.DataFrame, int]:
    """Remove rows from train_df whose text is an exact/near-duplicate of other_df."""
    other_norms = set(other_df[text_col].astype(str).map(normalize_text))
    train_norms = train_df[text_col].astype(str).map(normalize_text)
    exact_mask = train_norms.isin(other_norms)

    remaining_indices = train_df.index[~exact_mask].tolist()
    remaining_texts = train_df.loc[remaining_indices, text_col].astype(str).tolist()
    other_texts = other_df[text_col].astype(str).unique().tolist()
    near_pairs = find_cross_split_near_duplicates(remaining_texts, other_texts, threshold=threshold)
    near_drop_indices = {remaining_indices[i] for i, _, _ in near_pairs}

    drop_indices = set(train_df.index[exact_mask].tolist()) | near_drop_indices
    cleaned = train_df.drop(index=list(drop_indices)).reset_index(drop=True)
    return cleaned, len(drop_indices)


def grouped_train_val_split(
    df: pd.DataFrame,
    group_col: str = "group_id",
    label_col: str = "sentiment",
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train/val keeping all rows of each group together."""
    if df[group_col].nunique() < 2:
        raise ValueError("Need at least 2 distinct groups for grouped split.")

    n_splits = max(2, round(1.0 / val_size))
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = next(sgkf.split(df, df[label_col], df[group_col]))
    except ValueError:
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_idx, val_idx = next(gss.split(df, df[label_col], df[group_col]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def assert_no_split_leakage(
    splits: Dict[str, pd.DataFrame],
    group_col: str = "group_id",
    text_col: str = "text",
    near_dup_threshold: float = 0.90,
) -> None:
    """Assert zero group overlap, zero exact normalized text overlap, and zero near-dups."""
    names = list(splits.keys())
    for idx_a in range(len(names)):
        for idx_b in range(idx_a + 1, len(names)):
            name_a, name_b = names[idx_a], names[idx_b]
            df_a, df_b = splits[name_a], splits[name_b]

            groups_a = set(df_a[group_col].astype(str))
            groups_b = set(df_b[group_col].astype(str))
            group_overlap = groups_a & groups_b
            assert not group_overlap, (
                f"Group leakage between {name_a} and {name_b}: {len(group_overlap)} groups"
            )

            texts_a = set(df_a[text_col].astype(str).map(normalize_text))
            texts_b = set(df_b[text_col].astype(str).map(normalize_text))
            text_overlap = texts_a & texts_b
            assert not text_overlap, (
                f"Exact normalized text leakage between {name_a} and {name_b}: {len(text_overlap)}"
            )

            near_dups = find_cross_split_near_duplicates(
                df_a[text_col].astype(str).unique().tolist(),
                df_b[text_col].astype(str).unique().tolist(),
                threshold=near_dup_threshold,
            )
            assert not near_dups, (
                f"Near-duplicate text leakage between {name_a} and {name_b}: {len(near_dups)} pairs"
            )


@dataclass
class ParseStats:
    total_sentences: int
    total_aspect_entries: int
    kept_entries: int
    excluded_conflict: int
    excluded_other_polarity: int
    pii_redacted_rows: int


def parse_semeval2014_restaurant_xml(
    xml_path: Path,
    source_id: str,
    collection_date: str = "2014-03-01",
    drop_conflict: bool = True,
) -> Tuple[pd.DataFrame, ParseStats]:
    """Parse SemEval-2014 Task 4 Restaurant XML (aspectCategories) into PathPilot schema."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    rows: List[Dict[str, object]] = []

    total_sentences = 0
    total_entries = 0
    excluded_conflict = 0
    excluded_other = 0
    pii_redacted = 0

    for sent in root.findall(".//sentence"):
        total_sentences += 1
        sent_id = str(sent.attrib.get("id", f"s_{total_sentences}"))
        raw_text_el = sent.find("text")
        raw_text = raw_text_el.text if (raw_text_el is not None and raw_text_el.text) else ""
        cleaned_text, was_redacted = anonymise_text(raw_text)
        if was_redacted:
            pii_redacted += 1

        ac_parent = sent.find("aspectCategories")
        if ac_parent is None:
            continue

        for ac_idx, ac in enumerate(ac_parent.findall("aspectCategory")):
            total_entries += 1
            category = str(ac.attrib.get("category", "")).strip()
            polarity = str(ac.attrib.get("polarity", "")).strip().lower()

            if polarity == "conflict" and drop_conflict:
                excluded_conflict += 1
                continue
            if polarity not in POLARITIES:
                excluded_other += 1
                continue
            if category not in SEMEVAL_2014_RESTAURANT_ASPECTS:
                continue

            rows.append(
                {
                    "review_id": f"{source_id}_{sent_id}_{ac_idx}",
                    "location_id": f"semeval_rest_{sent_id}",
                    "group_id": f"{source_id}:{sent_id}",
                    "text": cleaned_text,
                    "aspect": category,
                    "sentiment": polarity,
                    "evidence_span_start": -1,
                    "evidence_span_end": -1,
                    "source_id": source_id,
                    "collection_date": collection_date,
                    "anonymised": True,
                    "adjudication_status": "benchmark_gold",
                }
            )

    df = pd.DataFrame(rows)
    stats = ParseStats(
        total_sentences=total_sentences,
        total_aspect_entries=total_entries,
        kept_entries=len(df),
        excluded_conflict=excluded_conflict,
        excluded_other_polarity=excluded_other,
        pii_redacted_rows=pii_redacted,
    )
    return df, stats


def download_semeval2014_dataset(output_dir: Path) -> Dict[str, Tuple[Path, str]]:
    """Download official SemEval-2014 Task 4 XML files via HuggingFace Hub if missing.

    Returns mapping of logical key -> (local_path, sha256_hex).
    """
    from huggingface_hub import hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "restaurants_train": "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train_v2.xml",
        "restaurants_test_gold": "ABSA_Gold_TestData/Restaurants_Test_Gold.xml",
        "laptops_train": "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Train_v2.xml",
        "laptops_test_gold": "ABSA_Gold_TestData/Laptops_Test_Gold.xml",
        "restaurants_trial": "restaurants-trial.xml",
    }
    resolved: Dict[str, Tuple[Path, str]] = {}
    for key, rel_path in targets.items():
        local_candidate = output_dir / rel_path
        if not local_candidate.exists():
            downloaded = hf_hub_download(
                repo_id="alexcadillon/SemEval2014Task4",
                filename=rel_path,
                repo_type="dataset",
                local_dir=str(output_dir),
            )
            local_candidate = Path(downloaded)
        sha256 = hashlib.sha256(local_candidate.read_bytes()).hexdigest()
        resolved[key] = (local_candidate, sha256)
    return resolved


def prepare_semeval2014_splits(
    raw_dir: Path,
    processed_dir: Path,
    val_size: float = 0.15,
    seed: int = 42,
) -> Dict[str, object]:
    """Prepare leakage-checked train/val/test CSVs from SemEval-2014 Restaurants XML."""
    resolved = download_semeval2014_dataset(raw_dir)
    train_xml, _ = resolved["restaurants_train"]
    test_xml, _ = resolved["restaurants_test_gold"]

    full_train_df, train_stats = parse_semeval2014_restaurant_xml(
        train_xml, source_id="semeval2014_rest_train"
    )
    test_df, test_stats = parse_semeval2014_restaurant_xml(
        test_xml, source_id="semeval2014_rest_test"
    )

    full_train_df, train_dedup = deduplicate_records(full_train_df)
    test_df, test_dedup = deduplicate_records(test_df)

    # Remove any train rows whose sentence text matches or near-duplicates the untouched test set
    full_train_df, cross_test_removed = remove_cross_split_text_leakage(
        full_train_df, test_df, threshold=0.90
    )

    train_df, val_df = grouped_train_val_split(
        full_train_df, group_col="group_id", label_col="sentiment", val_size=val_size, seed=seed
    )
    # Remove any rare cross-sentence near-duplicates between train and val
    train_df, cross_val_removed = remove_cross_split_text_leakage(
        train_df, val_df, threshold=0.90
    )

    assert_no_split_leakage(
        {"train": train_df, "val": val_df, "test": test_df},
        group_col="group_id",
        text_col="text",
        near_dup_threshold=0.90,
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "semeval2014_rest_train.csv"
    val_path = processed_dir / "semeval2014_rest_val.csv"
    test_path = processed_dir / "semeval2014_rest_test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_parse_stats": train_stats.__dict__,
        "test_parse_stats": test_stats.__dict__,
        "train_exact_dedup_removed": train_dedup,
        "test_exact_dedup_removed": test_dedup,
        "cross_test_leakage_removed": cross_test_removed,
        "cross_val_leakage_removed": cross_val_removed,
    }

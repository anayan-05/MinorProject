"""Training and evaluation script for PathPilot RoBERTa ABSA.

Owner: Anayan (feature/nlp-model)
Supports SemEval-2014 Task 4 benchmark/transfer data and PathPilot annotated CSVs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from ml.nlp.labels import ID_TO_POLARITY, POLARITIES, POLARITY_TO_ID
from ml.nlp.preprocess import prepare_semeval2014_splits

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_hf_tokenizer_and_model(model_name: str, num_labels: int) -> Tuple[Any, Any]:
    """Load tokenizer and model, preferring local HF cache if offline."""
    try:
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        mod = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=ID_TO_POLARITY,
            label2id=POLARITY_TO_ID,
            local_files_only=True,
        )
        return tok, mod
    except OSError:
        tok = AutoTokenizer.from_pretrained(model_name)
        mod = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=ID_TO_POLARITY,
            label2id=POLARITY_TO_ID,
        )
        return tok, mod


class BucketBatchSampler(Sampler[List[int]]):
    """Group similarly sized sequences into batches and shuffle batch order per epoch."""

    def __init__(self, lengths: Sequence[int], batch_size: int, seed: int = 42) -> None:
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        noisy_indices = sorted(
            range(len(self.lengths)),
            key=lambda i: self.lengths[i] + rng.uniform(-2.0, 2.0),
        )
        batches = [
            noisy_indices[i : i + self.batch_size]
            for i in range(0, len(noisy_indices), self.batch_size)
        ]
        rng.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


class ABSADataset(Dataset):
    """Sequence-pair dataset pairing review text with an aspect label (dynamic padding)."""

    def __init__(
        self,
        texts: Sequence[str],
        aspects: Sequence[str],
        labels: Sequence[int],
        tokenizer: Any,
        max_length: int = 96,
    ) -> None:
        self.encodings = tokenizer(
            [f"Aspect: {a}" for a in aspects],
            list(texts),
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.lengths = [len(ids) for ids in self.encodings["input_ids"]]
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics_bundle(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    aspects: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute overall accuracy, macro/weighted F1, per-class P/R/F1, and per-aspect stats."""
    labels_idx = list(range(len(POLARITIES)))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels_idx, average="macro", zero_division=0))
    weighted_f1 = float(
        f1_score(y_true, y_pred, labels=labels_idx, average="weighted", zero_division=0)
    )
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, zero_division=0
    )

    per_class: Dict[str, Dict[str, Any]] = {}
    for i, name in enumerate(POLARITIES):
        per_class[name] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(sup[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels_idx).tolist()

    per_aspect: Dict[str, Dict[str, Any]] = {}
    if aspects is not None:
        df_tmp = pd.DataFrame(
            {"aspect": list(aspects), "y_true": list(y_true), "y_pred": list(y_pred)}
        )
        for asp, sub in df_tmp.groupby("aspect"):
            asp_acc = float(accuracy_score(sub["y_true"], sub["y_pred"]))
            present = sorted(set(sub["y_true"]) | set(sub["y_pred"]))
            asp_macro_f1 = float(
                f1_score(
                    sub["y_true"], sub["y_pred"], labels=present, average="macro", zero_division=0
                )
            )
            per_aspect[str(asp)] = {
                "accuracy": round(asp_acc, 4),
                "macro_f1": round(asp_macro_f1, 4),
                "support": int(len(sub)),
            }

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "per_aspect": per_aspect,
        "confusion_matrix": cm,
        "confusion_matrix_labels": list(POLARITIES),
    }


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module,
    aspects: Optional[Sequence[str]] = None,
) -> Tuple[float, Dict[str, Any], List[int], List[float]]:
    """Run deterministic evaluation over a DataLoader."""
    model.eval()
    total_loss = 0.0
    total_items = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    pred_confs: List[float] = []

    for batch in loader:
        labels = batch["labels"].to(device)
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total_items += bs

        probs = torch.softmax(logits, dim=-1)
        confs, preds = torch.max(probs, dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        pred_confs.extend(confs.cpu().tolist())

    avg_loss = total_loss / max(1, total_items)
    metrics = compute_metrics_bundle(y_true, y_pred, aspects=aspects)
    return round(avg_loss, 4), metrics, y_pred, pred_confs


def build_error_analysis(
    df: pd.DataFrame,
    y_pred: Sequence[int],
    pred_confs: Sequence[float],
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """Collect highest-confidence misclassifications for qualitative error analysis."""
    errors: List[Dict[str, Any]] = []
    for idx, row in df.reset_index(drop=True).iterrows():
        true_label = str(row["sentiment"])
        pred_label = ID_TO_POLARITY[int(y_pred[idx])]
        if true_label != pred_label:
            errors.append(
                {
                    "review_id": str(row.get("review_id", idx)),
                    "aspect": str(row["aspect"]),
                    "text": str(row["text"]),
                    "true_sentiment": true_label,
                    "pred_sentiment": pred_label,
                    "confidence": round(float(pred_confs[idx]), 4),
                }
            )
    errors.sort(key=lambda r: r["confidence"], reverse=True)
    return errors[:top_n]


def _apply_layer_freezing(model: torch.nn.Module, freeze_bottom_layers: int) -> Tuple[int, int]:
    """Freeze embeddings and bottom N transformer blocks; keep top blocks and classifier trainable."""
    if freeze_bottom_layers <= 0:
        total = sum(p.numel() for p in model.parameters())
        return total, total

    for name, param in model.named_parameters():
        if name.startswith("classifier"):
            param.requires_grad = True
            continue
        if "encoder.layer." in name:
            try:
                layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
                param.requires_grad = layer_idx >= freeze_bottom_layers
            except (IndexError, ValueError):
                param.requires_grad = False
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    model_name: str = "distilroberta-base",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 5e-5,
    max_length: int = 96,
    seed: int = 42,
    use_class_weights: bool = True,
    freeze_bottom_layers: Optional[int] = None,
) -> Dict[str, Any]:
    """Train RoBERTa ABSA model for N epochs, log per-epoch stats, and evaluate test once."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    if freeze_bottom_layers is None:
        # On CPU, freeze bottom 5 of 6 DistilRoBERTa layers (train layer.5 + classifier) for feasible 10-epoch training;
        # on GPU (e.g. Colab T4), default to full fine-tuning (0 frozen layers).
        freeze_bottom_layers = 5 if device.type == "cpu" else 0

    tokenizer, model = _load_hf_tokenizer_and_model(model_name, num_labels=len(POLARITIES))
    model = model.to(device)
    trainable_params, total_params = _apply_layer_freezing(model, freeze_bottom_layers)
    print(
        f"Device={device} | model={model_name} | freeze_bottom_layers={freeze_bottom_layers} | "
        f"trainable_params={trainable_params:,}/{total_params:,}",
        flush=True,
    )

    train_labels = [POLARITY_TO_ID[s] for s in train_df["sentiment"]]
    val_labels = [POLARITY_TO_ID[s] for s in val_df["sentiment"]]
    test_labels = [POLARITY_TO_ID[s] for s in test_df["sentiment"]]

    train_ds = ABSADataset(
        train_df["text"].tolist(), train_df["aspect"].tolist(), train_labels, tokenizer, max_length
    )
    val_ds = ABSADataset(
        val_df["text"].tolist(), val_df["aspect"].tolist(), val_labels, tokenizer, max_length
    )
    test_ds = ABSADataset(
        test_df["text"].tolist(), test_df["aspect"].tolist(), test_labels, tokenizer, max_length
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    train_sampler = BucketBatchSampler(train_ds.lengths, batch_size=batch_size, seed=seed)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    if use_class_weights:
        counts = np.bincount(train_labels, minlength=len(POLARITIES)).astype(np.float32)
        weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        weights = weights / weights.mean()
        class_weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)
    else:
        class_weight_tensor = None
        loss_fn = torch.nn.CrossEntropyLoss()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(0.1 * total_steps)), num_training_steps=total_steps
    )

    epoch_history: List[Dict[str, Any]] = []
    epoch_log_path = output_dir / "epoch_history.jsonl"
    if epoch_log_path.exists():
        epoch_log_path.unlink()

    best_val_macro_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        seen = 0
        train_preds: List[int] = []
        train_trues: List[int] = []

        for step_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            bs = labels.size(0)
            running_loss += float(loss.item()) * bs
            seen += bs
            preds = torch.argmax(outputs.logits.detach(), dim=-1)
            train_preds.extend(preds.cpu().tolist())
            train_trues.extend(labels.cpu().tolist())
            if step_idx % 30 == 0 or step_idx == len(train_loader):
                print(
                    f"  [Epoch {epoch:02d} step {step_idx:02d}/{len(train_loader)}] "
                    f"avg_loss={running_loss / max(1, seen):.4f} elapsed={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )

        train_loss = round(running_loss / max(1, seen), 4)
        train_metrics = compute_metrics_bundle(train_trues, train_preds)
        val_loss, val_metrics, _, _ = evaluate_loader(
            model, val_loader, device, loss_fn, aspects=val_df["aspect"].tolist()
        )
        epoch_seconds = round(time.perf_counter() - t0, 2)
        current_lr = float(scheduler.get_last_lr()[0])

        is_best = val_metrics["macro_f1"] > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "train_per_class": train_metrics["per_class"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_per_class": val_metrics["per_class"],
            "val_per_aspect": val_metrics["per_aspect"],
            "val_confusion_matrix": val_metrics["confusion_matrix"],
            "learning_rate": round(current_lr, 8),
            "epoch_seconds": epoch_seconds,
            "is_best": is_best,
        }
        epoch_history.append(epoch_record)
        with epoch_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(epoch_record) + "\n")

        print(
            f"[Epoch {epoch:02d}/{epochs:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_metrics['accuracy']:.4f} train_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"lr={current_lr:.2e} ({epoch_seconds:.1f}s){' *BEST*' if is_best else ''}",
            flush=True,
        )
        for cls_name, cls_stats in val_metrics["per_class"].items():
            print(
                f"   val[{cls_name}]: P={cls_stats['precision']:.4f} R={cls_stats['recall']:.4f} "
                f"F1={cls_stats['f1']:.4f} (n={cls_stats['support']})",
                flush=True,
            )

    # Reload best saved model for single evaluation on untouched held-out test set
    best_model = AutoModelForSequenceClassification.from_pretrained(str(output_dir)).to(device)
    test_loss, test_metrics, test_preds, test_confs = evaluate_loader(
        best_model, test_loader, device, loss_fn, aspects=test_df["aspect"].tolist()
    )
    error_examples = build_error_analysis(test_df, test_preds, test_confs, top_n=15)

    # CPU smoke test timing on reloaded model
    best_cpu_model = best_model.to(torch.device("cpu"))
    best_cpu_model.eval()
    smoke_t0 = time.perf_counter()
    smoke_inputs = tokenizer(
        "Aspect: food",
        "The pasta was fresh and delicious though the waiter took twenty minutes.",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        smoke_logits = best_cpu_model(**smoke_inputs).logits
        smoke_probs = torch.softmax(smoke_logits, dim=-1).squeeze(0)
    smoke_ms = round((time.perf_counter() - smoke_t0) * 1000, 2)
    smoke_pred_id = int(torch.argmax(smoke_probs).item())

    weight_bytes = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file())
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_version": f"pathpilot-{model_name}-absa-v1.0.0",
        "device": str(device),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "max_length": max_length,
        "freeze_bottom_layers": freeze_bottom_layers,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "use_class_weights": use_class_weights,
        "class_weights": (
            [round(float(w), 4) for w in class_weight_tensor.cpu().tolist()]
            if class_weight_tensor is not None
            else None
        ),
        "dataset_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "epoch_history": epoch_history,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "error_analysis_top15": error_examples,
        "cpu_smoke_test": {
            "latency_ms": smoke_ms,
            "pred_sentiment": ID_TO_POLARITY[smoke_pred_id],
            "confidence": round(float(smoke_probs[smoke_pred_id].item()), 4),
        },
        "artifact_size_mb": round(weight_bytes / (1024 * 1024), 2),
    }

    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PathPilot RoBERTa ABSA model.")
    parser.add_argument("--model-name", default="distilroberta-base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-bottom-layers", type=int, default=None)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/nlp/semeval2014"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/nlp"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/models/pathpilot-roberta-absa")
    )
    args = parser.parse_args()

    prep_info = prepare_semeval2014_splits(args.raw_dir, args.processed_dir, seed=args.seed)
    print("Prepared dataset splits:", json.dumps(prep_info, indent=2), flush=True)

    train_df = pd.read_csv(prep_info["train_path"])
    val_df = pd.read_csv(prep_info["val_path"])
    test_df = pd.read_csv(prep_info["test_path"])

    summary = train_and_evaluate(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
        freeze_bottom_layers=args.freeze_bottom_layers,
    )
    print("Final held-out test metrics:", json.dumps(summary["test_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()

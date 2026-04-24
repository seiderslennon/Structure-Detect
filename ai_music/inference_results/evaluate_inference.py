#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

GEN_MODELS = ["heartmula", "songgeneration2", "diffrhythm2", "songbloom"]


def infer_source_model(song_dir: str) -> str:
    """
    Map each row to source model by basename.
    - generated samples look like: '<model>_NNN'
    - real samples are YouTube-like IDs (no known model prefix)
    """
    base = os.path.basename(str(song_dir)).lower()
    for model in GEN_MODELS:
        if base.startswith(f"{model}_"):
            return model
    return "real"


def _equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """EER is the operating point where FPR == FNR, reported as a single rate."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_metrics(df: pd.DataFrame, split_name: str) -> Dict[str, float]:
    y_true = df["y_true_fake"].astype(int).values  # 1=fake, 0=real
    y_pred = df["y_pred_fake"].astype(int).values  # 1=fake, 0=real
    y_score = df["fake_prob"].astype(float).values

    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    if len(set(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, y_score))
        eer = _equal_error_rate(y_true, y_score)
    else:
        auc = eer = float("nan")

    return {
        "split": split_name,
        "n_samples": len(df),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
        "accuracy": accuracy,
        "f1_fake": f1,
        "auc_fake": auc,
        "eer": eer,
    }


def _normalize_test_predictions_format(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Normalize a `test_predictions.csv`-style frame to the expected evalset schema.

    Source columns: filepath, target, y_true, y_pred
      - target / y_true: 1 = real, 0 = fake
      - y_pred: sigmoid output == probability of fake (threshold `threshold` for "fake")
    """
    out = pd.DataFrame()
    out["song_dir"] = df["filepath"].astype(str)
    fake_prob = df["y_pred"].astype(float)
    out["fake_prob"] = fake_prob
    out["real_prob"] = 1.0 - fake_prob
    is_fake_pred = fake_prob > threshold
    out["prediction_label"] = is_fake_pred.map({True: "fake", False: "real"})
    out["prediction"] = (~is_fake_pred).astype(int)  # 1 = real, 0 = fake
    out["confidence"] = out[["real_prob", "fake_prob"]].max(axis=1)
    return out


def build_eval_frame(df: pd.DataFrame, fake_threshold: float = 0.5) -> pd.DataFrame:
    evalset_cols = {"song_dir", "prediction", "prediction_label", "real_prob", "fake_prob"}
    test_pred_cols = {"filepath", "y_true", "y_pred"}

    if evalset_cols.issubset(df.columns):
        normalized = df.copy()
    elif test_pred_cols.issubset(df.columns):
        normalized = _normalize_test_predictions_format(df, threshold=fake_threshold)
    else:
        missing = sorted(evalset_cols - set(df.columns))
        raise ValueError(
            f"Unrecognized CSV schema. Missing evalset columns: {missing}. "
            f"Expected either {sorted(evalset_cols)} or {sorted(test_pred_cols)}."
        )

    out = normalized.copy()
    out["source_model"] = out["song_dir"].apply(infer_source_model)
    out["y_true_fake"] = (out["source_model"] != "real").astype(int)
    out["y_pred_fake"] = out["prediction_label"].astype(str).str.lower().eq("fake").astype(int)
    return out


def summarize_counts(df: pd.DataFrame) -> None:
    counts = df["source_model"].value_counts()
    print("\n=== Source counts inferred from song_dir ===")
    for split in ["real"] + GEN_MODELS:
        print(f"{split:>15}: {int(counts.get(split, 0))}")


def evaluate(
    df: pd.DataFrame, n_real_samples: int = 100, seed: int = 42
) -> pd.DataFrame:
    """Produce the report table: `overall` + one balanced row per generator.

    - `overall`: all songs in the eval frame (real + every generator's fakes).
    - per-generator rows: `n_real_samples` real songs vs all that generator's fakes.
    """
    rows: List[Dict[str, float]] = [compute_metrics(df, "overall")]

    real_pool = df[df["source_model"] == "real"].copy()
    if len(real_pool) < n_real_samples:
        raise ValueError(
            f"Requested {n_real_samples} real samples, but only found {len(real_pool)}."
        )

    for model in GEN_MODELS:
        fake_pool = df[df["source_model"] == model].copy()
        if fake_pool.empty:
            continue
        sampled_real = real_pool.sample(n=n_real_samples, random_state=seed)
        subset = pd.concat([sampled_real, fake_pool], ignore_index=True)
        rows.append(compute_metrics(subset, model))

    return pd.DataFrame(rows)


def _print_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("(empty)")
        return
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _load_and_build(
    csv_path: str, fake_threshold: float
) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    return build_eval_frame(raw, fake_threshold=fake_threshold)


def _intersect_song_dirs(eval_dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Restrict every frame to the basename (sans audio extension) intersection.

    This is what makes cross-model comparison apples-to-apples when different
    runs include slightly different song sets.
    """
    import re

    def key(path: str) -> str:
        b = os.path.basename(str(path)).lower()
        return re.sub(r"\.(mp3|wav|flac|ogg)$", "", b)

    key_sets = []
    for df in eval_dfs:
        key_sets.append(set(df["song_dir"].astype(str).map(key)))
    common = set.intersection(*key_sets) if key_sets else set()

    out = []
    for df in eval_dfs:
        keys = df["song_dir"].astype(str).map(key)
        out.append(df[keys.isin(common)].copy())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more inference CSVs. Reports accuracy, F1 (positive "
            "class = fake), AUC, and EER for the overall set and for each "
            "generation model on a balanced (N real + all fake) subset. "
            "Accepts both the `song_dir/prediction_label/real_prob/fake_prob` "
            "schema and the sonics `filepath/target/y_true/y_pred` schema."
        )
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        default=[
            "/home/lennon/AI_music/ai_music/inference_results/structture_evalset.csv"
        ],
        help="Path(s) to inference CSV. Pass multiple for a cross-model comparison table.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help=(
            "Labels for each --csv (same order). Defaults to the CSV file basename "
            "without extension."
        ),
    )
    parser.add_argument(
        "--intersect",
        action="store_true",
        help=(
            "When multiple CSVs are provided, restrict evaluation to the intersection "
            "of their song sets (by basename). Strongly recommended for cross-model "
            "comparison."
        ),
    )
    parser.add_argument(
        "--save",
        default="",
        help="Optional path to save metrics as CSV.",
    )
    parser.add_argument(
        "--balanced-real-samples",
        type=int,
        default=100,
        help="Number of real songs to sample for each per-generator row.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for balanced real-song sampling.",
    )
    parser.add_argument(
        "--fake-threshold",
        type=float,
        default=0.5,
        help=(
            "Decision threshold on y_pred (fake probability) used when the input CSV "
            "is in test_predictions format (columns: filepath,target,y_true,y_pred). "
            "Ignored for CSVs that already contain prediction_label."
        ),
    )
    args = parser.parse_args()

    csv_paths: List[str] = list(args.csv)
    if args.labels:
        if len(args.labels) != len(csv_paths):
            raise ValueError(
                f"--labels count ({len(args.labels)}) must match --csv count ({len(csv_paths)})."
            )
        labels = list(args.labels)
    else:
        labels = [Path(p).stem for p in csv_paths]

    eval_dfs = [
        _load_and_build(p, fake_threshold=args.fake_threshold) for p in csv_paths
    ]

    if len(eval_dfs) > 1 and args.intersect:
        eval_dfs = _intersect_song_dirs(eval_dfs)

    for label, p, ed in zip(labels, csv_paths, eval_dfs):
        print(f"\n### [{label}] {p}")
        summarize_counts(ed)

    tables: List[pd.DataFrame] = []
    for label, ed in zip(labels, eval_dfs):
        m = evaluate(
            ed,
            n_real_samples=args.balanced_real_samples,
            seed=args.seed,
        )
        m.insert(0, "model", label)
        tables.append(m)

    metrics = pd.concat(tables, ignore_index=True)

    if len(eval_dfs) == 1:
        display = metrics.drop(columns=["model"])
    else:
        display = metrics.sort_values(["split", "model"]).reset_index(drop=True)

    print(
        f"\n=== Metrics (positive class = fake; per-generator rows use "
        f"{args.balanced_real_samples} real + all fake, seed={args.seed}) ==="
    )
    _print_table(display)

    if args.save:
        metrics.to_csv(args.save, index=False)
        print(f"\nSaved metrics table to: {args.save}")


if __name__ == "__main__":
    main()

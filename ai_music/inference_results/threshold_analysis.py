#!/usr/bin/env python3
"""
Threshold-sweep & calibration diagnostics for inference CSVs.

Complements evaluate_inference.py: while evaluate_inference reports a single
operating point (threshold=0.5 by default), this script characterizes how the
classifier's score distribution behaves across thresholds, which is what you
actually want when AUC is high but accuracy at 0.5 is low (saturation /
mis-calibration).

Reports per CSV:
  - threshold sweep table (acc / F1 / TP / FP / FN / TN at user-chosen thresholds)
  - the EER operating point and its threshold
  - the threshold that maximizes accuracy and the one that maximizes F1(fake)
  - P(fake) percentiles split by true label (saturation check)

Accepts the same three CSV schemas as evaluate_inference.py:
  1. predict_evalset.py / predict_cached.py: song_id, label, prediction_label, real_prob, fake_prob
  2. infer.py:                              song_dir, prediction_label, real_prob, fake_prob
  3. test_predictions:                      filepath, target, y_true, y_pred

Usage:
    # single CSV
    python -m ai_music.inference_results.threshold_analysis \\
        --csv ai_music/inference_results/v306_ep0_ssep_evalset.csv

    # cross-run comparison restricted to common songs
    python -m ai_music.inference_results.threshold_analysis \\
        --csv  v306_ep0_ssep_evalset.csv  structtture_ep4_evalset.csv \\
        --labels v306_ep0 ep4 \\
        --intersect

    # custom threshold grid + save the sweep
    python -m ai_music.inference_results.threshold_analysis \\
        --csv v306_ep0_ssep_evalset.csv \\
        --thresholds 0.5 0.1 0.01 1e-3 1e-4 1e-5 \\
        --save sweep.csv
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from evaluate_inference import (  # noqa: E402
    GEN_MODELS,
    build_eval_frame,
    evaluate,
)


DEFAULT_THRESHOLDS = (0.5, 0.1, 0.05, 0.01, 1e-3, 1e-4, 1e-5, 1e-6)


def _load_eval_frame(csv_path: str, fake_threshold: float = 0.5) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    return build_eval_frame(raw, fake_threshold=fake_threshold)


def _intersect_eval_frames(eval_dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Restrict every frame to the basename intersection (matching evaluate_inference)."""
    import re

    def key(path: str) -> str:
        b = os.path.basename(str(path)).lower()
        return re.sub(r"\.(mp3|wav|flac|ogg)$", "", b)

    if not eval_dfs:
        return eval_dfs
    key_sets = [set(df["song_dir"].astype(str).map(key)) for df in eval_dfs]
    common = set.intersection(*key_sets)
    out = []
    for df in eval_dfs:
        keys = df["song_dir"].astype(str).map(key)
        out.append(df[keys.isin(common)].copy())
    return out


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def threshold_sweep(
    df: pd.DataFrame, thresholds: Sequence[float]
) -> pd.DataFrame:
    """For each threshold, compute acc / F1 / confusion entries.
    Decision rule: predict fake if fake_prob >= threshold.
    """
    y_true = df["y_true_fake"].astype(int).values
    s = df["fake_prob"].astype(float).values

    rows = []
    for thr in thresholds:
        y_pred = (s >= thr).astype(int)
        cm = _confusion(y_true, y_pred)
        acc = float((y_pred == y_true).mean())
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        rows.append({
            "threshold": float(thr),
            "accuracy": acc,
            "f1_fake": f1,
            **cm,
        })
    return pd.DataFrame(rows)


def best_thresholds(df: pd.DataFrame, n_grid: int = 1024) -> Dict[str, Dict[str, float]]:
    """Find threshold grid points that maximize accuracy / F1, plus the EER threshold.
    The grid is taken from unique fake_prob values (capped at n_grid evenly-spaced
    samples) so we evaluate at every meaningful operating point without wasting
    work on plateaus.
    """
    y_true = df["y_true_fake"].astype(int).values
    s = df["fake_prob"].astype(float).values

    if len(s) == 0:
        return {}

    cand = np.unique(s)
    if len(cand) > n_grid:
        idx = np.linspace(0, len(cand) - 1, n_grid).astype(int)
        cand = cand[idx]
    # also include 0.5 so we can compare to the default
    cand = np.unique(np.concatenate([cand, [0.5]]))

    best_acc = {"threshold": 0.5, "accuracy": -1.0, "f1_fake": float("nan")}
    best_f1 = {"threshold": 0.5, "accuracy": float("nan"), "f1_fake": -1.0}
    for thr in cand:
        y_pred = (s >= thr).astype(int)
        acc = float((y_pred == y_true).mean())
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if acc > best_acc["accuracy"]:
            best_acc = {"threshold": float(thr), "accuracy": acc, "f1_fake": f1}
        if f1 > best_f1["f1_fake"]:
            best_f1 = {"threshold": float(thr), "accuracy": acc, "f1_fake": f1}

    out = {"max_accuracy": best_acc, "max_f1": best_f1}

    # EER threshold (where FPR == FNR)
    if len(set(y_true)) >= 2:
        fpr, tpr, thrs = roc_curve(y_true, s)
        fnr = 1.0 - tpr
        i = int(np.nanargmin(np.abs(fpr - fnr)))
        eer_thr = float(thrs[i]) if np.isfinite(thrs[i]) else float("nan")
        eer = float((fpr[i] + fnr[i]) / 2.0)
        auc = float(roc_auc_score(y_true, s))
        out["eer"] = {
            "threshold": eer_thr,
            "fpr": float(fpr[i]),
            "fnr": float(fnr[i]),
            "eer": eer,
            "auc": auc,
        }
    return out


def score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class percentiles of fake_prob — saturation diagnostic."""
    rows = []
    for cls, mask in [("real", df["y_true_fake"] == 0), ("fake", df["y_true_fake"] == 1)]:
        vals = df.loc[mask, "fake_prob"].astype(float).values
        if len(vals) == 0:
            continue
        pcts = np.percentile(vals, [0, 5, 25, 50, 75, 95, 100])
        rows.append({
            "true_label": cls,
            "n": int(len(vals)),
            "min": float(pcts[0]),
            "p05": float(pcts[1]),
            "p25": float(pcts[2]),
            "median": float(pcts[3]),
            "p75": float(pcts[4]),
            "p95": float(pcts[5]),
            "max": float(pcts[6]),
            "mean_log10": float(np.mean(np.log10(np.clip(vals, 1e-30, 1.0)))),
        })
    return pd.DataFrame(rows)


def _print_table(df: pd.DataFrame, float_fmt=None) -> None:
    if df is None or df.empty:
        print("(empty)")
        return
    if float_fmt is None:
        float_fmt = lambda x: f"{x:.4f}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"
    print(df.to_string(index=False, float_format=float_fmt))


def _format_best(best: Dict[str, Dict[str, float]]) -> str:
    lines = []
    if "max_accuracy" in best:
        b = best["max_accuracy"]
        lines.append(
            f"  best-accuracy threshold = {b['threshold']:.3e}   "
            f"acc={b['accuracy']:.4f}  f1={b['f1_fake']:.4f}"
        )
    if "max_f1" in best:
        b = best["max_f1"]
        lines.append(
            f"  best-f1 threshold       = {b['threshold']:.3e}   "
            f"acc={b['accuracy']:.4f}  f1={b['f1_fake']:.4f}"
        )
    if "eer" in best:
        e = best["eer"]
        lines.append(
            f"  EER threshold           = {e['threshold']:.3e}   "
            f"eer={e['eer']:.4f}  auc={e['auc']:.4f}"
        )
    return "\n".join(lines)


def _apply_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return a copy of ``df`` with prediction columns recomputed using ``threshold``.
    Decision rule: predict fake iff ``fake_prob >= threshold``.
    """
    out = df.copy()
    is_fake = out["fake_prob"].astype(float).values >= float(threshold)
    out["y_pred_fake"] = is_fake.astype(int)
    out["prediction_label"] = np.where(is_fake, "fake", "real")
    # evalset schema: prediction == 1 means real, 0 means fake
    out["prediction"] = (~is_fake).astype(int)
    return out


def report_at_threshold(
    df: pd.DataFrame,
    threshold: float,
    n_real_samples: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the overall + per-generator metrics table (matching
    ``evaluate_inference.evaluate``) after thresholding ``fake_prob`` at
    ``threshold``. AUC / EER are threshold-independent and unchanged.
    """
    rethr = _apply_threshold(df, threshold)
    return evaluate(rethr, n_real_samples=n_real_samples, seed=seed)


def analyse_one(
    label: str,
    df: pd.DataFrame,
    thresholds: Sequence[float],
    per_generator: bool,
    n_grid: int,
    n_real_samples: int = 100,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Run the full diagnostic on one eval frame and pretty-print it."""
    print(f"\n{'='*70}\n[{label}]  n={len(df)}  "
          f"real={int((df['y_true_fake'] == 0).sum())}  "
          f"fake={int((df['y_true_fake'] == 1).sum())}\n{'='*70}")

    sweep = threshold_sweep(df, thresholds)
    sweep.insert(0, "label", label)

    print("\n--- threshold sweep (predict fake if fake_prob >= threshold) ---")
    _print_table(
        sweep.drop(columns=["label"]),
        float_fmt=lambda x: f"{x:.4f}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}",
    )

    best = best_thresholds(df, n_grid=n_grid)
    if best:
        print("\n--- optimal operating points ---")
        print(_format_best(best))

    best_f1_table: Optional[pd.DataFrame] = None
    if best and "max_f1" in best:
        thr_f1 = float(best["max_f1"]["threshold"])
        best_f1_table = report_at_threshold(
            df, thr_f1, n_real_samples=n_real_samples, seed=seed
        )
        print(
            f"\n=== Metrics @ best-F1 threshold = {thr_f1:.6e} "
            f"(positive class = fake; per-generator rows use "
            f"{n_real_samples} real + all fake, seed={seed}) ==="
        )
        print(
            best_f1_table.to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    dist = score_distribution(df)
    dist.insert(0, "label", label)
    print("\n--- P(fake) percentiles by true label (saturation check) ---")
    _print_table(dist.drop(columns=["label"]),
                 float_fmt=lambda x: f"{x:.2e}" if abs(x) < 1 else f"{x:.4f}")

    per_gen_table: Optional[pd.DataFrame] = None
    if per_generator:
        rows = []
        for model in GEN_MODELS:
            sub = df[df["source_model"].isin(["real", model])].copy()
            n_fake = int((sub["y_true_fake"] == 1).sum())
            if n_fake == 0:
                continue
            best_sub = best_thresholds(sub, n_grid=n_grid)
            row = {
                "generator": model,
                "n_real": int((sub["y_true_fake"] == 0).sum()),
                "n_fake": n_fake,
                "auc": best_sub.get("eer", {}).get("auc", float("nan")),
                "eer": best_sub.get("eer", {}).get("eer", float("nan")),
                "eer_threshold": best_sub.get("eer", {}).get("threshold", float("nan")),
                "best_acc": best_sub.get("max_accuracy", {}).get("accuracy", float("nan")),
                "best_acc_threshold": best_sub.get("max_accuracy", {}).get("threshold", float("nan")),
            }
            rows.append(row)
        if rows:
            per_gen_table = pd.DataFrame(rows)
            per_gen_table.insert(0, "label", label)
            print("\n--- per-generator (real + that generator's fakes) ---")
            _print_table(per_gen_table.drop(columns=["label"]),
                         float_fmt=lambda x: f"{x:.4f}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}")

    return {
        "sweep": sweep,
        "distribution": dist,
        "per_generator": per_gen_table if per_gen_table is not None else pd.DataFrame(),
        "best_f1_metrics": (
            best_f1_table if best_f1_table is not None else pd.DataFrame()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--csv", nargs="+", required=True,
        help="Path(s) to inference CSV. Pass multiple for a cross-run comparison.",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Display labels for each --csv (defaults to filename stem).",
    )
    parser.add_argument(
        "--intersect", action="store_true",
        help="When multiple CSVs are provided, restrict to the intersection of "
             "their song basenames before computing metrics.",
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS),
        help=f"Threshold grid for the sweep table (default: {DEFAULT_THRESHOLDS}).",
    )
    parser.add_argument(
        "--per-generator", action="store_true",
        help="Also report AUC/EER/best-accuracy per generator (real + that "
             "generator's fakes only).",
    )
    parser.add_argument(
        "--n-grid", type=int, default=1024,
        help="Max number of unique fake_prob values to scan when searching for "
             "best-accuracy / best-f1 thresholds (default: 1024).",
    )
    parser.add_argument(
        "--save", default="",
        help="Optional path prefix. Writes <prefix>_sweep.csv, <prefix>_distribution.csv, "
             "and (if --per-generator) <prefix>_per_generator.csv.",
    )
    parser.add_argument(
        "--fake-threshold", type=float, default=0.5,
        help="Threshold to use when normalizing the test_predictions CSV schema "
             "(filepath/y_pred). Ignored for the other schemas.",
    )
    parser.add_argument(
        "--balanced-real-samples", type=int, default=100,
        help="Number of real songs to sample for each per-generator row in the "
             "best-F1 metrics table (default: 100).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for balanced real-song sampling in the best-F1 "
             "metrics table (default: 42).",
    )
    args = parser.parse_args()

    csv_paths: List[str] = list(args.csv)
    labels = (
        list(args.labels) if args.labels
        else [Path(p).stem for p in csv_paths]
    )
    if len(labels) != len(csv_paths):
        raise ValueError(
            f"--labels count ({len(labels)}) must match --csv count ({len(csv_paths)})."
        )

    eval_dfs = [_load_eval_frame(p, fake_threshold=args.fake_threshold) for p in csv_paths]

    if len(eval_dfs) > 1 and args.intersect:
        eval_dfs = _intersect_eval_frames(eval_dfs)

    sweep_tables: List[pd.DataFrame] = []
    dist_tables: List[pd.DataFrame] = []
    per_gen_tables: List[pd.DataFrame] = []
    best_f1_tables: List[pd.DataFrame] = []

    thrs_sorted = sorted(set(args.thresholds), reverse=True)

    for label, df in zip(labels, eval_dfs):
        out = analyse_one(
            label, df,
            thresholds=thrs_sorted,
            per_generator=args.per_generator,
            n_grid=args.n_grid,
            n_real_samples=args.balanced_real_samples,
            seed=args.seed,
        )
        sweep_tables.append(out["sweep"])
        dist_tables.append(out["distribution"])
        if not out["per_generator"].empty:
            per_gen_tables.append(out["per_generator"])
        if not out["best_f1_metrics"].empty:
            tbl = out["best_f1_metrics"].copy()
            tbl.insert(0, "label", label)
            best_f1_tables.append(tbl)

    if args.save:
        prefix = Path(args.save)
        if prefix.suffix.lower() == ".csv":
            prefix = prefix.with_suffix("")
        prefix.parent.mkdir(parents=True, exist_ok=True)

        sweep_all = pd.concat(sweep_tables, ignore_index=True)
        sweep_all.to_csv(f"{prefix}_sweep.csv", index=False)
        print(f"\nSaved threshold sweep to: {prefix}_sweep.csv")

        dist_all = pd.concat(dist_tables, ignore_index=True)
        dist_all.to_csv(f"{prefix}_distribution.csv", index=False)
        print(f"Saved distribution to:    {prefix}_distribution.csv")

        if per_gen_tables:
            pg_all = pd.concat(per_gen_tables, ignore_index=True)
            pg_all.to_csv(f"{prefix}_per_generator.csv", index=False)
            print(f"Saved per-generator to:   {prefix}_per_generator.csv")

        if best_f1_tables:
            bf1_all = pd.concat(best_f1_tables, ignore_index=True)
            bf1_all.to_csv(f"{prefix}_best_f1_metrics.csv", index=False)
            print(f"Saved best-F1 metrics to: {prefix}_best_f1_metrics.csv")


if __name__ == "__main__":
    main()

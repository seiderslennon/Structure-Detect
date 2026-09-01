"""
Score the SONICS SpecTTTra baseline on *our* val split.

The SONICS model was trained on full (non-source-separated) audio living under
/data/SONICS/sonics/{fake,real}_songs/<filename>.mp3. The filenames match
combined_songs.csv exactly, so we can rebuild the same seed-42 90/10 val split
that cached_dataset.py produces and feed those mp3s through SONICS' own
AudioClassifier + AudioDataset (so resampling / cropping / normalization match
their training pipeline bit-for-bit).

Output CSV mirrors predict_cached.py's schema, so you can diff the two files
directly:
    song_id, label, prediction, prediction_label, confidence, real_prob, fake_prob

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m ai_music.predict_sonics \
        --config /data/SONICS/sonics/configs/spectttra_f5t7-120s.yaml \
        --checkpoint /data/SONICS/sonics/output/spectttra_gamma-t=120/best_checkpoint.pth \
        --csv_out ai_music/inference_results/sonics_spectttra_gamma_t120_val.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Make the vendored SONICS repo importable.
_SONICS_REPO = Path(__file__).resolve().parent.parent / "tmp" / "sonics"
if str(_SONICS_REPO) not in sys.path:
    sys.path.insert(0, str(_SONICS_REPO))

from sonics.models.model import AudioClassifier  # noqa: E402
from sonics.utils.config import dict2cfg  # noqa: E402
from sonics.utils.dataset import AudioDataset  # noqa: E402

DATASET_ROOT = Path("/data/SONICS/dataset_50k")
SONICS_AUDIO_ROOT = Path("/data/SONICS/sonics")


def _file_pair_exists(row, root: Path) -> bool:
    """Mirror cached_dataset.file_pair_exists so we get the same population
    that training/predict_cached.py see (i.e. only songs whose v+a were
    actually source-separated)."""
    d = root / row["source"] / str(row["filename"])
    return (d / "vocals.wav").exists() and (d / "accompaniment.wav").exists()


def build_val_split(csv_path: Path, seed: int = 42, val_frac: float = 0.1) -> pd.DataFrame:
    """Reproduce cached_dataset.CachedDataset.get_tracks(split='val').
    Same filter, same shuffle seeds, same per-class 90/10 cut."""
    df = pd.read_csv(csv_path)
    df = df[df.apply(lambda r: _file_pair_exists(r, DATASET_ROOT), axis=1)].reset_index(drop=True)
    real = df[df.source == "real"].sample(frac=1, random_state=seed).reset_index(drop=True)
    fake = df[df.source == "fake"].sample(frac=1, random_state=seed).reset_index(drop=True)
    ri = int((1.0 - val_frac) * len(real))
    fi = int((1.0 - val_frac) * len(fake))
    val = pd.concat([real[ri:], fake[fi:]], ignore_index=True)
    return val


def map_to_sonics_paths(val: pd.DataFrame) -> pd.DataFrame:
    """For each val row, resolve its full-mix mp3 under /data/SONICS/sonics/.
    Drops rows whose mp3 is missing (rare, but keeps inference robust)."""
    rows = []
    skipped = 0
    for _, r in val.iterrows():
        is_fake = r["source"] == "fake"
        rel = f"{'fake_songs' if is_fake else 'real_songs'}/{r['filename']}.mp3"
        path = SONICS_AUDIO_ROOT / rel
        if not path.exists():
            skipped += 1
            continue
        rows.append({
            "song_id": f"{r['source']}_{r['filename']}",  # matches predict_cached.py
            "label": r["source"],                          # 'real' / 'fake'
            "filepath": str(path),
            "target": 1 if is_fake else 0,                 # SONICS convention: 1 = fake
        })
    if skipped:
        print(f"[warn] skipped {skipped} val rows missing mp3 under {SONICS_AUDIO_ROOT}")
    return pd.DataFrame(rows)


@torch.no_grad()
def collect_predictions(model, loader, device, use_amp: bool):
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16) if use_amp
        else torch.amp.autocast("cuda", enabled=False)
    )
    model.eval()
    all_probs = []
    all_targets = []
    pbar = tqdm(loader, desc="infer", ncols=100, dynamic_ncols=False)
    for batch in pbar:
        x = batch["audio"].to(device, non_blocking=True)
        y = batch["target"]
        with autocast_ctx:
            logits = model(x)
        logits = logits.squeeze(-1) if logits.dim() > 1 else logits
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.cpu().numpy().astype(int))
    return np.concatenate(all_targets), np.concatenate(all_probs)


def write_csv(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "song_id", "label", "prediction", "prediction_label",
        "confidence", "real_prob", "fake_prob",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def print_metrics(rows):
    """Same metrics block as predict_cached.py."""
    labelled = [r for r in rows if r["label"] in ("real", "fake")]
    if not labelled:
        print("No ground-truth labels; skipping metrics.")
        return

    y_true_fake = np.array([1 if r["label"] == "fake" else 0 for r in labelled])
    y_pred_fake = np.array([1 if r["prediction_label"] == "fake" else 0 for r in labelled])
    fake_score = np.array([r["fake_prob"] for r in labelled])

    n = len(labelled)
    n_real = int((y_true_fake == 0).sum())
    n_fake = int((y_true_fake == 1).sum())
    acc = float((y_true_fake == y_pred_fake).mean())

    print()
    print(f"{'='*60}")
    print(f"  n={n}  real={n_real}  fake={n_fake}")
    print(f"  accuracy = {acc:.4f}")

    try:
        from sklearn.metrics import f1_score, roc_auc_score, roc_curve
        f1 = float(f1_score(y_true_fake, y_pred_fake, zero_division=0))
        print(f"  f1 (fake)= {f1:.4f}")
        if n_real > 0 and n_fake > 0:
            auc = float(roc_auc_score(y_true_fake, fake_score))
            fpr, tpr, _ = roc_curve(y_true_fake, fake_score)
            fnr = 1.0 - tpr
            idx = int(np.nanargmin(np.abs(fpr - fnr)))
            eer = float((fpr[idx] + fnr[idx]) / 2.0)
            print(f"  auc      = {auc:.4f}")
            print(f"  eer      = {eer:.4f}")
    except ImportError:
        print("  (install scikit-learn for f1 / auc / eer)")
    print(f"{'='*60}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True,
                   help="SONICS yaml (e.g. /data/SONICS/sonics/configs/spectttra_f5t7-120s.yaml).")
    p.add_argument("--checkpoint", required=True, help="SONICS .pth checkpoint.")
    p.add_argument("--combined_csv", default="/data/SONICS/dataset_50k/combined_songs.csv")
    p.add_argument("--csv_out", required=True, help="Output predictions CSV (predict_cached schema).")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.1)
    args = p.parse_args()

    cfg = dict2cfg(yaml.safe_load(open(args.config)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(getattr(cfg.environment, "mixed_precision", False)) and device.type == "cuda"

    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}  (amp={use_amp})")
    print(f"max_len:    {cfg.audio.max_len} samples"
          f"  ({cfg.audio.max_time}s @ {cfg.audio.sample_rate}Hz)")

    print("\n[1/4] Building val split (matches cached_dataset's seed-42 90/10)...")
    val = build_val_split(Path(args.combined_csv), seed=args.seed, val_frac=args.val_frac)
    sonics_df = map_to_sonics_paths(val)
    print(f"  -> {len(sonics_df)} songs "
          f"({(sonics_df.target == 1).sum()} fake / {(sonics_df.target == 0).sum()} real)")
    if len(sonics_df) == 0:
        raise RuntimeError("Empty val set after path mapping.")

    print("\n[2/4] Loading SONICS AudioClassifier + checkpoint...")
    model = AudioClassifier(cfg).to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    print(f"  -> loaded {args.checkpoint}")

    # Eval-time AudioDataset settings: deterministic crop, no augment, std-norm
    # (matches what train.py/test.py pass to get_dataloader for valid/test).
    ds = AudioDataset(
        filepaths=sonics_df.filepath.tolist(),
        labels=sonics_df.target.tolist(),
        skip_times=None,
        num_classes=cfg.num_classes,
        max_len=cfg.audio.max_len,
        random_sampling=False,
        normalize="std",
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    print(f"\n[3/4] Running inference on {len(ds)} clips (batch={args.batch_size})...")
    y_true_fake, fake_prob = collect_predictions(model, loader, device, use_amp)

    # Translate SONICS' P(fake) into our schema (prediction=1 means real).
    pred_label_fake = (fake_prob > 0.5).astype(int)
    rows = []
    for sid, lab, ptarget, pfake in zip(
        sonics_df.song_id.values,
        sonics_df.label.values,
        pred_label_fake,
        fake_prob,
    ):
        prediction = int(0 if ptarget == 1 else 1)  # 1 = real, 0 = fake
        rows.append({
            "song_id": sid,
            "label": lab,
            "prediction": prediction,
            "prediction_label": "fake" if ptarget == 1 else "real",
            "confidence": float(max(pfake, 1.0 - pfake)),
            "real_prob": float(1.0 - pfake),
            "fake_prob": float(pfake),
        })

    write_csv(rows, args.csv_out)
    print(f"\n[4/4] Wrote {len(rows)} predictions to {args.csv_out}")
    print_metrics(rows)


if __name__ == "__main__":
    main()

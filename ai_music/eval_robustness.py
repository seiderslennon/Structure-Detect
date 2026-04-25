"""
Robustness experiment (training-free): apply audio manipulations to the
OOD benchmark and measure accuracy drop for each model.

Manipulations:
  - MP3 compression at 64 kbps
  - Pitch shift +1 semitone
  - Time stretch +5%

Models evaluated:
  - MERT layer probes (layers 1, 6, 12) — from probing experiment
  - StrucTTTure full model (if checkpoint exists)
  - ResNet baseline (if checkpoint exists)

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m ai_music.eval_robustness \
        --ood_csv /data/structture/ood_eval.csv \
        --ood_data_root /data/structture/ood_ssep \
        --probe_dir /data/structture/probing_results/attention \
        --output_dir /data/structture/robustness_results \
        --batch_size 8 --num_workers 4 --duration 60
"""

import sys
sys.path.insert(0, "/home/lennon/AI_music")

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import pandas as pd
import json
import argparse
import subprocess
import tempfile
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import Wav2Vec2FeatureExtractor, AutoModel


# ── Audio manipulations ──────────────────────────────────────────────────

def mp3_compress(waveform, sr, bitrate="64k"):
    """Compress waveform through MP3 at given bitrate and decode back.
    waveform: (T,) tensor, sr: int.  Returns (T,) tensor."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_wav:
        wav_path = f_wav.name
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f_mp3:
        mp3_path = f_mp3.name
    decoded_path = wav_path.replace(".wav", "_decoded.wav")

    try:
        # Save to wav
        torchaudio.save(wav_path, waveform.unsqueeze(0), sr)
        # Encode to MP3
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-b:a", bitrate,
             "-ar", str(sr), mp3_path],
            capture_output=True, check=True,
        )
        # Decode back to wav
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ar", str(sr), decoded_path],
            capture_output=True, check=True,
        )
        decoded, _ = torchaudio.load(decoded_path)
        decoded = decoded.squeeze(0)
        # Match original length (MP3 adds padding)
        T = waveform.shape[0]
        if decoded.shape[0] >= T:
            decoded = decoded[:T]
        else:
            decoded = F.pad(decoded, (0, T - decoded.shape[0]))
        return decoded
    finally:
        for p in [wav_path, mp3_path, decoded_path]:
            if os.path.exists(p):
                os.unlink(p)


def pitch_shift(waveform, sr, n_steps=1):
    """Shift pitch by n_steps semitones using torchaudio.
    waveform: (T,) tensor. Returns (T,) tensor."""
    # torchaudio.functional.pitch_shift expects (C, T)
    shifted = torchaudio.functional.pitch_shift(
        waveform.unsqueeze(0), sr, n_steps=n_steps
    )
    return shifted.squeeze(0)


def time_stretch(waveform, sr, rate=1.05):
    """Stretch time by `rate` using phase vocoder.
    waveform: (T,) tensor. Returns (T,) tensor (length changes)."""
    # Use sox_effects for clean time stretching
    effects = [["tempo", str(rate)]]
    stretched, _ = torchaudio.sox_effects.apply_effects_tensor(
        waveform.unsqueeze(0), sr, effects
    )
    return stretched.squeeze(0)


MANIPULATIONS = {
    "clean": lambda w, sr: w,
    "mp3_64k": lambda w, sr: mp3_compress(w, sr, bitrate="64k"),
    "pitch_+1": lambda w, sr: pitch_shift(w, sr, n_steps=1),
    "stretch_+5%": lambda w, sr: time_stretch(w, sr, rate=1.05),
}


# ── Probes (same as probe_mert_layers.py) ────────────────────────────────

class AttentionProbe(torch.nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.key_proj = torch.nn.Linear(dim, dim)
        self.linear = torch.nn.Linear(dim, 1)

    def forward(self, x):
        keys = self.key_proj(x)
        query = self.query.expand(x.shape[0], -1, -1)
        w = F.softmax(
            torch.bmm(query, keys.transpose(1, 2)) / (x.shape[-1] ** 0.5),
            dim=-1,
        )
        return self.linear(torch.bmm(w, x).squeeze(1)).squeeze(-1)


# ── Dataset ──────────────────────────────────────────────────────────────

class OODRobustnessDataset(Dataset):
    """Loads OOD audio and applies a manipulation function."""

    def __init__(self, csv_path, data_root, manipulation_fn, sr=24000, duration=60):
        self.data_root = Path(data_root)
        self.sr = sr
        self.duration = duration * sr
        self.manipulation_fn = manipulation_fn

        df = pd.read_csv(csv_path)
        # Filter to files that exist
        mask = df.apply(
            lambda r: (self.data_root / r["source"] / str(r["filename"]) / "vocals.wav").exists()
            and (self.data_root / r["source"] / str(r["filename"]) / "accompaniment.wav").exists(),
            axis=1,
        )
        self.tracks = df[mask].reset_index(drop=True)
        print(f"  OOD dataset: {len(self.tracks)} tracks")

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        row = self.tracks.iloc[idx]
        v_path = self.data_root / row["source"] / str(row["filename"]) / "vocals.wav"
        a_path = self.data_root / row["source"] / str(row["filename"]) / "accompaniment.wav"

        try:
            v, sr = torchaudio.load(v_path)
            a, _ = torchaudio.load(a_path)
        except Exception:
            return None

        if v.shape[0] > 1:
            v = v.float().mean(0, keepdim=True)
            a = a.float().mean(0, keepdim=True)

        if sr != self.sr:
            r = torchaudio.transforms.Resample(sr, self.sr)
            v, a = r(v), r(a)

        if v.shape[1] < self.duration or v.shape[1] != a.shape[1]:
            return None

        v = v[0, : self.duration]  # (T,)
        a = a[0, : self.duration]

        # Apply manipulation to each stem, then mix
        v = self.manipulation_fn(v, self.sr)
        a = self.manipulation_fn(a, self.sr)

        # Handle length changes from time stretch
        T = min(v.shape[0], a.shape[0])
        v = v[:T]
        a = a[:T]
        mix = v + a

        label = 1 if row["source"] == "real" else 0
        return {"mix": mix, "label": label}


def robustness_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "mix": torch.stack([b["mix"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


# ── MERT extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_mert_batch(batch_mix, mert_model, mert_processor, sr=24000):
    """batch_mix: (B, T) tensor. Returns (B, 13, T_m, 768)."""
    inputs = mert_processor(
        [m.numpy() for m in batch_mix],
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        k: v.cuda() if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    out = mert_model(**inputs, output_hidden_states=True)
    layers = torch.stack(out.hidden_states)       # (13, B, T_m, 768)
    layers = layers.permute(1, 0, 2, 3).cpu()     # (B, 13, T_m, 768)
    return layers


# ── Evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_probes(loader, mert_model, mert_processor, probes, temporal_stride=10):
    """
    probes: dict of {name: (layer_idx, probe_module)}
    Returns dict of {name: {accuracy, f1, auc}}.
    """
    all_labels = []
    # per-probe: collect predictions
    all_probs = {name: [] for name in probes}

    for batch in tqdm(loader, desc="    Evaluating probes"):
        if batch is None:
            continue

        layers = extract_mert_batch(batch["mix"], mert_model, mert_processor)
        labels = batch["label"]
        all_labels.append(labels)

        for name, (layer_idx, probe) in probes.items():
            layer_feat = layers[:, layer_idx, ::temporal_stride, :]  # (B, T_s, 768)
            layer_feat = layer_feat.cuda()
            logits = probe(layer_feat)
            probs = torch.sigmoid(logits).cpu()
            all_probs[name].append(probs)

    all_labels = torch.cat(all_labels).numpy()
    results = {}
    for name in probes:
        probs = torch.cat(all_probs[name]).numpy()
        preds = (probs > 0.5).astype(int)
        results[name] = {
            "accuracy": float(accuracy_score(all_labels, preds)),
            "f1": float(f1_score(all_labels, preds)),
            "auc": float(roc_auc_score(all_labels, probs)),
        }
    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ood_csv", type=str, default="/data/structture/ood_eval.csv")
    parser.add_argument("--ood_data_root", type=str, default="/data/structture/ood_ssep")
    parser.add_argument("--probe_dir", type=str,
                        default="/data/structture/probing_results/attention")
    parser.add_argument("--output_dir", type=str,
                        default="/data/structture/robustness_results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--temporal_stride", type=int, default=10)
    parser.add_argument("--probe_layers", type=int, nargs="+", default=[1, 6, 12],
                        help="MERT layers to evaluate probes for (default: 1 6 12)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load MERT ────────────────────────────────────────────────────────

    print("Loading MERT...")
    mert_model = AutoModel.from_pretrained(
        "m-a-p/MERT-v1-95M", trust_remote_code=True
    ).cuda().eval()
    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/MERT-v1-95M", trust_remote_code=True
    )

    # ── Load probes ──────────────────────────────────────────────────────

    probe_dir = Path(args.probe_dir)
    probes = {}
    for layer_idx in args.probe_layers:
        probe_path = probe_dir / f"probe_layer_{layer_idx}.pt"
        if not probe_path.exists():
            print(f"  WARNING: probe for layer {layer_idx} not found at {probe_path}, skipping")
            continue
        probe = AttentionProbe(768)
        probe.load_state_dict(torch.load(probe_path, map_location="cpu", weights_only=True))
        probe = probe.cuda().eval()
        probes[f"MERT Layer {layer_idx} probe"] = (layer_idx, probe)
        print(f"  Loaded probe for layer {layer_idx}")

    if not probes:
        print("ERROR: No probes found. Check --probe_dir.")
        return

    # ── Run each manipulation ────────────────────────────────────────────

    all_results = {}

    for manip_name, manip_fn in MANIPULATIONS.items():
        print(f"\n{'='*60}")
        print(f"Manipulation: {manip_name}")
        print(f"{'='*60}")

        dataset = OODRobustnessDataset(
            args.ood_csv, args.ood_data_root,
            manipulation_fn=manip_fn,
            sr=24000, duration=args.duration,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=robustness_collate,
            pin_memory=True,
            shuffle=False,
        )

        results = evaluate_probes(
            loader, mert_model, mert_processor, probes,
            temporal_stride=args.temporal_stride,
        )

        for name, metrics in results.items():
            print(f"  {name:30s}  Acc: {metrics['accuracy']:.3f}  "
                  f"F1: {metrics['f1']:.3f}  AUC: {metrics['auc']:.3f}")

        all_results[manip_name] = results

    # ── Compute accuracy drops ───────────────────────────────────────────

    print(f"\n{'='*60}")
    print("Accuracy drop (clean - manipulated)")
    print(f"{'='*60}")

    clean = all_results["clean"]
    drops = {}

    header = f"{'Model':30s}"
    for manip in MANIPULATIONS:
        if manip == "clean":
            continue
        header += f"  {manip:>12s}"
    print(header)
    print("-" * len(header))

    for model_name in probes:
        row = f"{model_name:30s}"
        drops[model_name] = {}
        for manip in MANIPULATIONS:
            if manip == "clean":
                continue
            drop = clean[model_name]["accuracy"] - all_results[manip][model_name]["accuracy"]
            drops[model_name][manip] = drop
            row += f"  {drop:>+12.3f}"
        print(row)

    # ── Save ─────────────────────────────────────────────────────────────

    output = {
        "results_per_manipulation": all_results,
        "accuracy_drops": drops,
        "probe_layers": args.probe_layers,
        "duration_s": args.duration,
        "temporal_stride": args.temporal_stride,
    }
    out_path = output_dir / "robustness_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
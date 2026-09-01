"""
EvalSetCachedDataset: flat-layout sibling of ``CachedDataset`` for the SONICS
EvalSet.

Differences from CachedDataset:
- Source-separated stems are not used at score time; per-modality features are
  loaded from <cache_dir>/<song_id>.pt (produced by precompute_features_evalset.py).
- The MERT input is selectable via ``mix_source``:
    * ``"stems_sum"`` (default): load
      ``<ssep_root>/<song_id>/{vocals,accompaniment}.wav`` and use
      ``vocals + accompaniment`` as the MERT input. This matches the
      pre-b66ca5e training pipeline (e.g. lightning_logs/ablate-beat,
      fusion-concat, mert-only) and the current default training pipeline.
    * ``"mp3"``: load the *original* mp3 from
      ``<mix_root>/<song_id>.mp3`` — a flat layout with no fake_songs /
      real_songs split. ``mix_root`` defaults to /data/SONICS/EvalSet.
      This matches checkpoints trained after commit b66ca5e ("fixing mix
      pipeline").
- Labels are optional. If a labels CSV is provided (e.g.
  /data/SONICS/evalset_complete.csv with columns ``filepath,target``,
  target=1 -> real, target=0 -> fake), they're attached to each sample so
  the predict script can compute metrics. Songs with no label are returned
  with label=None (predict script will skip metrics for those).
"""

import os
import sys

sys.path.insert(0, str("/home/lennon/AI_music/ISMIR2019-Large-Vocabulary-Chord-Recognition"))
sys.path.insert(0, str("/home/lennon/AI_music/beat_this"))
sys.path.insert(0, "/home/lennon/AI_music")

from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")


def _load_label_map(labels_csv):
    """Parse a labels CSV. Accepts the schema in /data/SONICS/evalset_complete.csv:
        filepath,target          (target: 1=real, 0=fake)
    Returns {song_id (mp3 stem): 'real' | 'fake'}.
    """
    if labels_csv is None:
        return {}
    df = pd.read_csv(labels_csv)
    if "filepath" not in df.columns or "target" not in df.columns:
        raise ValueError(
            f"{labels_csv} must have 'filepath' and 'target' columns "
            f"(got {list(df.columns)})"
        )
    out = {}
    for _, row in df.iterrows():
        sid = Path(row["filepath"]).stem
        out[sid] = "real" if int(row["target"]) == 1 else "fake"
    return out


class EvalSetCachedDataset(torch.utils.data.Dataset):
    """Flat-layout cached dataset for the SONICS EvalSet."""

    def __init__(
        self,
        data_configs,
        cache_dir="/data/structture/cached_features/ssep_evalset",
        mix_root="/data/SONICS/EvalSet",
        labels_csv=None,
        mix_source="stems_sum",
        ssep_root="/data/SONICS/ssep_EvalSet",
    ):
        if mix_source not in ("mp3", "stems_sum"):
            raise ValueError(
                f"mix_source must be 'mp3' or 'stems_sum'; got {mix_source!r}"
            )
        self.sr = data_configs["sample_rate"]
        self.duration = data_configs["duration"]
        self.cache_dir = Path(cache_dir)
        self.mix_root = Path(mix_root)
        self.ssep_root = Path(ssep_root)
        self.mix_source = mix_source
        self.label_map = _load_label_map(labels_csv)

        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"cache_dir does not exist: {self.cache_dir}. "
                f"Run precompute_features_evalset.py first."
            )

        cached_ids = sorted(p.stem for p in self.cache_dir.glob("*.pt"))
        if not cached_ids:
            raise RuntimeError(f"No .pt files in {self.cache_dir}")

        # Build per-song mix path(s) based on the requested mix_source. We
        # resolve to either a single mp3 file (mix_source="mp3") or the pair
        # of stem files (mix_source="stems_sum"); songs missing those inputs
        # are dropped here so MERT never sees a missing file at score time.
        rows = []
        missing_mix = 0
        for sid in cached_ids:
            if self.mix_source == "mp3":
                mix_path = self.mix_root / f"{sid}.mp3"
                if not mix_path.exists():
                    missing_mix += 1
                    continue
                rows.append({"song_id": sid, "mix_path": str(mix_path),
                              "vocals_path": None, "accomp_path": None,
                              "label": self.label_map.get(sid)})
            else:  # stems_sum
                v_path = self.ssep_root / sid / "vocals.wav"
                a_path = self.ssep_root / sid / "accompaniment.wav"
                if not (v_path.exists() and a_path.exists()):
                    missing_mix += 1
                    continue
                rows.append({"song_id": sid, "mix_path": None,
                              "vocals_path": str(v_path),
                              "accomp_path": str(a_path),
                              "label": self.label_map.get(sid)})
        if not rows:
            where = (
                f"mp3 mixes under {self.mix_root}" if self.mix_source == "mp3"
                else f"stems under {self.ssep_root}"
            )
            raise RuntimeError(f"No songs left after filtering for {where}")

        self.tracks = pd.DataFrame(rows)
        self._resamplers = {}

        n_labelled = int(self.tracks["label"].notna().sum())
        src_str = (
            f"mix_root={self.mix_root}" if self.mix_source == "mp3"
            else f"ssep_root={self.ssep_root} (vocals+accomp)"
        )
        print(
            f"EvalSetCachedDataset: {len(self.tracks)} songs "
            f"({n_labelled} labelled, {len(self.tracks) - n_labelled} unlabelled, "
            f"{missing_mix} dropped no-mix). "
            f"cache={self.cache_dir}, mix_source={self.mix_source}, {src_str}"
        )

    def _get_resampler(self, orig_sr, target_sr):
        key = (orig_sr, target_sr)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(orig_sr, target_sr)
        return self._resamplers[key]

    def __len__(self):
        return len(self.tracks)

    def _load_mono(self, path):
        audio, sr = torchaudio.load(path)
        if audio.shape[0] > 1:
            audio = audio.float().mean(dim=0, keepdim=True)
        else:
            audio = audio.float()
        if self.sr and sr != self.sr:
            audio = self._get_resampler(sr, self.sr)(audio)
            sr = self.sr
        return audio, sr

    @torch.no_grad()
    def __getitem__(self, idx):
        row = self.tracks.iloc[idx]
        song_id = row["song_id"]

        cached = torch.load(self.cache_dir / f"{song_id}.pt", weights_only=True)
        whisper_emb = cached["whisper"].detach().unsqueeze(0)
        crepe_emb = cached["crepe"].detach().unsqueeze(0)
        chord_emb = cached["chord"].detach().unsqueeze(0)
        beat_emb = cached["beat"].detach().unsqueeze(0)

        if self.mix_source == "mp3":
            mix_audio, sr = self._load_mono(row["mix_path"])
        else:  # stems_sum: align lengths, then add vocals + accompaniment
            v_audio, sr = self._load_mono(row["vocals_path"])
            a_audio, _ = self._load_mono(row["accomp_path"])
            n = min(v_audio.shape[1], a_audio.shape[1])
            mix_audio = v_audio[:, :n] + a_audio[:, :n]
        duration = self.duration * sr

        # Pad short mixes with zeros so we always score the song. Models trained
        # on 60s clips handle silence padding gracefully.
        if mix_audio.shape[1] < duration:
            mix_audio = torch.nn.functional.pad(
                mix_audio, (0, duration - mix_audio.shape[1])
            )
        mix_clip = mix_audio[:, :duration].squeeze(0).contiguous()

        return {
            "emb": (whisper_emb, crepe_emb, chord_emb, beat_emb),
            "mix": mix_clip,
            "label": row["label"],   # 'real' / 'fake' / None
            "song_id": song_id,
        }


def evalset_collate(batch):
    """Mirrors cached_dataset.cached_collate but tolerates label=None."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    embeddings = [item["emb"] for item in batch]
    labels = [item["label"] for item in batch]
    song_ids = [item["song_id"] for item in batch]

    whisper_batch = torch.stack([emb[0] for emb in embeddings])
    crepe_batch = torch.stack([emb[1] for emb in embeddings])
    chord_batch = torch.stack([emb[2] for emb in embeddings])
    beat_this_batch = torch.stack([emb[3] for emb in embeddings])
    mix_batch = torch.stack([item["mix"] for item in batch])

    return {
        "emb": (whisper_batch, crepe_batch, chord_batch, beat_this_batch),
        "mix": mix_batch,
        "label": labels,
        "song_id": song_ids,
    }


def get_evalset_dataloader(
    data_configs,
    train_configs,
    cache_dir="/data/structture/cached_features/ssep_evalset",
    mix_root="/data/SONICS/EvalSet",
    labels_csv=None,
    batch_size=None,
    num_workers=None,
    mix_source="stems_sum",
    ssep_root="/data/SONICS/ssep_EvalSet",
):
    dataset = EvalSetCachedDataset(
        data_configs,
        cache_dir=cache_dir,
        mix_root=mix_root,
        labels_csv=labels_csv,
        mix_source=mix_source,
        ssep_root=ssep_root,
    )

    bs = batch_size if batch_size is not None else int(train_configs.get("batch_size", 8))
    nw = num_workers if num_workers is not None else int(train_configs.get("num_workers", 0))
    pin_memory = bool(train_configs.get("pin_memory", False))

    loader_kwargs = dict(
        batch_size=bs,
        num_workers=nw,
        pin_memory=pin_memory,
        collate_fn=evalset_collate,
        shuffle=False,
        drop_last=False,
    )
    if nw > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(train_configs.get("prefetch_factor", 4))

    return DataLoader(dataset, **loader_kwargs)

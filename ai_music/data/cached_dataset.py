"""
CachedDataset: loads precomputed whisper/crepe/chord/beat from .pt files
and returns the raw mix audio for the LightningModule to run MERT on
in a single batched forward pass during training_step.

The .pt cache format is unchanged from precompute_features.py.

Usage in train.py:
    from ai_music.data.cached_dataset import get_cached_dataloader as get_dataloader
"""

import sys
sys.path.insert(0, str("/home/lennon/AI_music/ISMIR2019-Large-Vocabulary-Chord-Recognition"))
sys.path.insert(0, str("/home/lennon/AI_music/beat_this"))
sys.path.insert(0, "/home/lennon/AI_music")

import torch
from torch.utils.data import DataLoader
import torchaudio
import pandas as pd
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")


class CachedDataset():
    """
    Loads precomputed whisper/crepe/chord/beat from .pt files and returns
    the raw mix audio waveform. MERT is run on the batched audio inside the
    LightningModule's training_step so it benefits from batched inference,
    multi-process dataloading, and mixed-precision autocast.
    """
    def __init__(self, data_configs, split, cache_dir="/data/structture/cached_features"):
        self.paths_csv = Path(data_configs["data_root"])
        self.sr = data_configs["sample_rate"]
        self.duration = data_configs["duration"]
        self.random_sample = data_configs["random_sample"]

        self.pathext = Path(os.path.split(self.paths_csv)[0])
        self.cache_dir = Path(cache_dir) / split

        df = pd.read_csv(self.paths_csv)
        mask = df.apply(lambda row: file_pair_exists(row, self.pathext), axis=1)
        self.df = df[mask].reset_index(drop=True)
        self.get_tracks(split)

        # Filter to songs that have cached features
        before = len(self.tracks)
        self.tracks = self.tracks[self.tracks.apply(
            lambda row: (self.cache_dir / f"{row['source']}_{row['filename']}.pt").exists(),
            axis=1)].reset_index(drop=True)
        after = len(self.tracks)

        # Resamplers built once; workers inherit them via fork.
        self._resamplers = {}

        print(f"{split} size: {after} (cached) ({before - after} skipped, no cache file)")

    def _get_resampler(self, orig_sr, target_sr):
        key = (orig_sr, target_sr)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(orig_sr, target_sr)
        return self._resamplers[key]

    def __len__(self):
        return len(self.tracks)

    @torch.no_grad()
    def __getitem__(self, idx):
        idx_row = self.tracks.iloc[idx]
        song_id = f"{idx_row['source']}_{idx_row['filename']}"

        # Load cached features (kept fp16 to halve H2D bytes; autocast handles dtype).
        # .detach() because some entries in the cache files were saved with
        # requires_grad=True (precompute_features.py didn't always detach), which
        # breaks DataLoader multiprocessing serialization.
        cached = torch.load(self.cache_dir / f"{song_id}.pt", weights_only=True)
        whisper_emb = cached['whisper'].detach().unsqueeze(0)   # (1, T_w, 384)
        crepe_emb   = cached['crepe'].detach().unsqueeze(0)     # (1, T_c, 256)
        chord_emb   = cached['chord'].detach().unsqueeze(0)     # (1, T_ch, 240)
        beat_emb    = cached['beat'].detach().unsqueeze(0)      # (1, T_b, 512)

        # Load audio for MERT (full mix); kept on CPU so DataLoader workers can do this.
        v_path = self.pathext / idx_row['source'] / idx_row['filename'] / 'vocals.wav'
        a_path = self.pathext / idx_row['source'] / idx_row['filename'] / 'accompaniment.wav'

        v_audio, sr = torchaudio.load(v_path)
        a_audio, _  = torchaudio.load(a_path)
        if v_audio.shape[0] > 1:
            v_audio = v_audio.float().mean(dim=0, keepdim=True)
            a_audio = a_audio.float().mean(dim=0, keepdim=True)

        if self.sr and (sr != self.sr):
            resampler = self._get_resampler(sr, self.sr)
            v_audio = resampler(v_audio)
            a_audio = resampler(a_audio)
            sr = self.sr
        duration = self.duration * sr

        if len(v_audio[0]) < duration or len(v_audio[0]) != len(a_audio[0]):
            return None

        # Deterministic crop from start (matches caching).
        v_clip = v_audio[:, :duration]
        a_clip = a_audio[:, :duration]
        mix_clip = (v_clip + a_clip).squeeze(0).contiguous()  # (T,) fp32, leaf

        sample = {
            "emb": (whisper_emb, crepe_emb, chord_emb, beat_emb),
            "mix": mix_clip,
            "label": idx_row['source'],
        }
        return sample

    def get_tracks(self, split):
        split_ratio = 0.9

        real_df = self.df[self.df['source'] == 'real'].sample(frac=1, random_state=42).reset_index(drop=True)
        fake_df = self.df[self.df['source'] == 'fake'].sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"\n{'='*60}")
        print(f"Total dataset - Real: {len(real_df)}, Fake: {len(fake_df)}")

        real_split_index = int(split_ratio * len(real_df))
        fake_split_index = int(split_ratio * len(fake_df))

        if split == "train":
            self.tracks = pd.concat([real_df[:real_split_index], fake_df[:fake_split_index]], ignore_index=True)
            self.tracks = self.tracks.sample(frac=1, random_state=42).reset_index(drop=True)
        elif split == "val":
            self.tracks = pd.concat([real_df[real_split_index:], fake_df[fake_split_index:]], ignore_index=True)
            self.tracks = self.tracks.sample(frac=1, random_state=42).reset_index(drop=True)


def file_pair_exists(row, basepath):
    song_dir = basepath / row["source"] / str(row["filename"])
    vocals = song_dir / "vocals.wav"
    accomp = song_dir / "accompaniment.wav"
    return vocals.exists() and accomp.exists()


def cached_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    embeddings = [item['emb'] for item in batch]
    labels = [item['label'] for item in batch]

    whisper_batch    = torch.stack([emb[0] for emb in embeddings])
    crepe_batch      = torch.stack([emb[1] for emb in embeddings])
    chord_batch      = torch.stack([emb[2] for emb in embeddings])
    beat_this_batch  = torch.stack([emb[3] for emb in embeddings])
    mix_batch        = torch.stack([item['mix'] for item in batch])  # (B, T) fp32

    return {
        'emb': (whisper_batch, crepe_batch, chord_batch, beat_this_batch),
        'mix': mix_batch,
        'label': labels,
    }


def get_cached_dataloader(split, data_configs, train_configs, shuffle=True):
    dataset = CachedDataset(data_configs, split)

    num_workers = int(train_configs.get('num_workers', 0))
    pin_memory = bool(train_configs.get('pin_memory', False))

    # Drop the last partial batch only on the training loader so BatchNorm
    # never sees a batch of size 1 (the ResNet classifier head uses BatchNorm1d).
    # Validation runs in eval mode (running stats), so a batch of 1 is harmless there.
    loader_kwargs = dict(
        batch_size=train_configs['batch_size'],
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=cached_collate,
        shuffle=shuffle,
        drop_last=shuffle,
    )
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = int(train_configs.get('prefetch_factor', 4))

    return DataLoader(dataset, **loader_kwargs)

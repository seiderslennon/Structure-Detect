"""
CachedDataset: loads precomputed whisper/crepe/chord/beat from .pt files,
runs only MERT on-the-fly on the full mix (vocal + accompaniment).

Drop-in replacement for AudioDataset. Same batch format, same collate function.
~5x faster training since only 1 model runs per sample instead of 5.

Usage in train.py:
    # Replace:
    from ai_music.data.dataset import get_dataloader
    # With:
    from ai_music.data.cached_dataset import get_cached_dataloader as get_dataloader
"""

import sys
sys.path.insert(0, str("/home/lennon/AI_music/ISMIR2019-Large-Vocabulary-Chord-Recognition"))
sys.path.insert(0, str("/home/lennon/AI_music/beat_this"))
sys.path.insert(0, "/home/lennon/AI_music")

from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch

from torch.utils.data import DataLoader
import numpy as np
import torchaudio
import pandas as pd
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")


class CachedDataset():
    """
    Loads precomputed whisper/crepe/chord/beat from .pt files.
    Runs only MERT on-the-fly on the full mix (vocal + accompaniment).
    """
    def __init__(self, data_configs, split, cache_dir="/data/structture/cached_features"):
        self.paths_csv = Path(data_configs["data_root"])
        self.sr = data_configs["sample_rate"]
        self.duration = data_configs["duration"]
        self.random_sample = data_configs["random_sample"]

        self.mert_model = None
        self.mert_processor = None
        self._models_initialized = False

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

        self._init_models()
        print(f"{split} size: {after} (cached) ({before - after} skipped, no cache file)")

    def _init_models(self):
        if self._models_initialized:
            return
        if self.mert_model is None:
            self.mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
            self.mert_model = self.mert_model.to('cuda')
            self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        self._models_initialized = True

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        if not self._models_initialized:
            self._init_models()

        idx_row = self.tracks.iloc[idx]
        song_id = f"{idx_row['source']}_{idx_row['filename']}"

        # Load cached features (float16 -> float32)
        cached = torch.load(self.cache_dir / f"{song_id}.pt", weights_only=True)
        whisper_emb = cached['whisper'].float().unsqueeze(0)   # (1, T_w, 384)
        crepe_emb = cached['crepe'].float().unsqueeze(0)       # (1, T_c, 256)
        chord_emb = cached['chord'].float().unsqueeze(0)       # (1, T_ch, 240)
        beat_emb = cached['beat'].float().unsqueeze(0)          # (1, T_b, 512)

        # Load audio for MERT (full mix)
        v_path = self.pathext / idx_row['source'] / idx_row['filename'] / 'vocals.wav'
        a_path = self.pathext / idx_row['source'] / idx_row['filename'] / 'accompaniment.wav'

        v_audio, sr = torchaudio.load(v_path)
        a_audio, sr = torchaudio.load(a_path)
        if v_audio.shape[0] > 1:
            v_audio = v_audio.float().mean(dim=0, keepdim=True)
            a_audio = a_audio.float().mean(dim=0, keepdim=True)

        if self.sr and (sr != self.sr):
            transform = torchaudio.transforms.Resample(sr, self.sr)
            v_audio = transform(v_audio)
            a_audio = transform(a_audio)
            sr = self.sr
        duration = self.duration * sr

        if len(v_audio[0]) < duration or len(v_audio[0]) != len(a_audio[0]):
            return None

        # Deterministic crop from start (matches caching)
        v_clip = v_audio[:, :duration]
        a_clip = a_audio[:, :duration]

        # Full mix for MERT
        mix_clip = v_clip + a_clip
        mert_emb = self._mert_emb(mix_clip)

        embeddings = (whisper_emb.detach().cpu(), crepe_emb.detach().cpu(),
                      chord_emb.detach().cpu(), beat_emb.detach().cpu(),
                      mert_emb.detach().cpu())
        sample = {"emb": embeddings,
                  "label": idx_row['source']}
        return sample

    def _mert_emb(self, clip):
        mert_inputs = self.mert_processor(clip.squeeze(0), sampling_rate=24000, return_tensors="pt")
        mert_inputs = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in mert_inputs.items()}
        with torch.no_grad():
            mert_output = self.mert_model(**mert_inputs, output_hidden_states=True)
        mert_all_layer_hidden_states = torch.stack(mert_output.hidden_states).squeeze()
        return mert_all_layer_hidden_states

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

    whisper_batch = torch.stack([emb[0] for emb in embeddings])
    crepe_batch = torch.stack([emb[1] for emb in embeddings])
    chord_batch = torch.stack([emb[2] for emb in embeddings])
    beat_this_batch = torch.stack([emb[3] for emb in embeddings])
    mert_batch = torch.stack([emb[4] for emb in embeddings])

    return {
        'emb': (whisper_batch, crepe_batch, chord_batch, beat_this_batch, mert_batch),
        'label': labels
    }


def get_cached_dataloader(split, data_configs, train_configs, shuffle=True):
    dataset = CachedDataset(data_configs, split)

    dataloader = DataLoader(
        dataset,
        batch_size=train_configs['batch_size'],
        num_workers=0,  # must be 0: MERT runs on GPU inside __getitem__
        pin_memory=train_configs['pin_memory'],
        collate_fn=cached_collate,
        shuffle=shuffle
    )
    return dataloader
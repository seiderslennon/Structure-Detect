"""
CachedDataset: loads precomputed whisper/crepe/chord/beat from .pt files
and returns the raw mix audio for the LightningModule to run MERT on
in a single batched forward pass during training_step.

The .pt cache format is unchanged from precompute_features.py.

The MERT input is selectable via ``data.mix_source`` in the YAML config:

- ``"stems_sum"`` (default): load the source-separated stems from
  ``<paths_csv parent>/<source>/<filename>/{vocals,accompaniment}.wav``
  and use ``vocals + accompaniment`` as the MERT input. This matches the
  pre-b66ca5e training pipeline (and the current default, since most
  in-flight experiments train on summed stems).

- ``"mp3"``: load the *original*, non-source-separated mp3 from
  ``data.mix_root`` (default ``/data/SONICS/sonics``). The folder naming
  convention is ``<mix_root>/<source>_songs/<filename>.mp3`` where
  ``source`` is ``real`` or ``fake`` (so e.g. ``real`` -> ``real_songs``)
  and ``filename`` matches the source-separated song-folder name 1:1.
  This matches the post-b66ca5e ("fixing mix pipeline") training pipeline.

The cached per-modality features (whisper/crepe/chord/beat) are unchanged
between modes — only MERT's input differs.

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
    def __init__(self, data_configs, split, cache_dir="/data/structture/cached_features",
                 mix_source=None):
        self.paths_csv = Path(data_configs["data_root"])
        self.sr = data_configs["sample_rate"]
        self.duration = data_configs["duration"]
        self.random_sample = data_configs["random_sample"]
        # Root that holds the original (non-source-separated) mp3s used as the
        # MERT input when mix_source="mp3". See the module docstring.
        self.mix_root = Path(data_configs.get("mix_root", "/data/SONICS/sonics"))

        # MERT input selection. Order of precedence: explicit constructor arg,
        # then data_configs["mix_source"], then "stems_sum" (the default —
        # matches the pre-b66ca5e training pipeline that the current set of
        # checkpoints was trained on). Set data.mix_source: mp3 in the YAML to
        # opt into the post-b66ca5e original-mp3-mix pipeline.
        resolved = mix_source if mix_source is not None else data_configs.get("mix_source", "stems_sum")
        if resolved not in ("mp3", "stems_sum"):
            raise ValueError(
                f"data.mix_source must be 'mp3' or 'stems_sum'; got {resolved!r}"
            )
        self.mix_source = resolved

        self.pathext = Path(os.path.split(self.paths_csv)[0])
        self.cache_dir = Path(cache_dir) / split

        df = pd.read_csv(self.paths_csv)
        # file_pair_exists already requires both stems to be present, so it
        # covers the precondition for mix_source="stems_sum" too.
        mask = df.apply(lambda row: file_pair_exists(row, self.pathext), axis=1)
        self.df = df[mask].reset_index(drop=True)
        self.get_tracks(split)

        # Filter to songs that have cached per-modality features
        before = len(self.tracks)
        self.tracks = self.tracks[self.tracks.apply(
            lambda row: (self.cache_dir / f"{row['source']}_{row['filename']}.pt").exists(),
            axis=1)].reset_index(drop=True)
        after_cache = len(self.tracks)

        # In "mp3" mode, additionally filter to songs whose original-mix mp3
        # exists, so MERT never sees a missing file at training time. In
        # "stems_sum" mode the stems were already verified by file_pair_exists.
        if self.mix_source == "mp3":
            self.tracks = self.tracks[self.tracks.apply(
                lambda row: self._mix_path(row).exists(),
                axis=1)].reset_index(drop=True)
        after_mix = len(self.tracks)

        # Resamplers built once; workers inherit them via fork.
        self._resamplers = {}

        if self.mix_source == "mp3":
            tail = (
                f"({before - after_cache} skipped no-cache, "
                f"{after_cache - after_mix} skipped no-mp3)"
            )
        else:
            tail = f"({before - after_cache} skipped no-cache)"
        print(f"{split} size: {after_mix} mix_source={self.mix_source} {tail}")

    def _get_resampler(self, orig_sr, target_sr):
        key = (orig_sr, target_sr)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(orig_sr, target_sr)
        return self._resamplers[key]

    def _mix_path(self, row):
        """Resolve the original-mix mp3 path for a given CSV row.
        ``source == 'fake'`` -> ``<mix_root>/fake_songs/<filename>.mp3``.
        ``source == 'real'`` -> ``<mix_root>/real_songs/<filename>.mp3``.
        """
        sub = "fake_songs" if row["source"] == "fake" else "real_songs"
        return self.mix_root / sub / f"{row['filename']}.mp3"

    def __len__(self):
        return len(self.tracks)

    def _load_mono(self, path):
        """Load `path`, downmix to mono, and resample to self.sr. Returns
        (audio[1, T], sr) — kept on CPU so DataLoader workers can do this in
        parallel."""
        audio, sr = torchaudio.load(path)
        if audio.shape[0] > 1:
            audio = audio.float().mean(dim=0, keepdim=True)
        else:
            audio = audio.float()
        if self.sr and sr != self.sr:
            audio = self._get_resampler(sr, self.sr)(audio)
            sr = self.sr
        return audio, sr

    def _stems_paths(self, idx_row):
        """Source-separated stem paths under <pathext>/<source>/<filename>/."""
        song_dir = self.pathext / idx_row['source'] / str(idx_row['filename'])
        return song_dir / 'vocals.wav', song_dir / 'accompaniment.wav'

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

        # Build the MERT input. mp3 -> original full-mix mp3; stems_sum ->
        # vocals + accompaniment from the source-separated stems.
        if self.mix_source == "mp3":
            mix_audio, sr = self._load_mono(self._mix_path(idx_row))
        else:  # stems_sum
            v_path, a_path = self._stems_paths(idx_row)
            v_audio, sr = self._load_mono(v_path)
            a_audio, _ = self._load_mono(a_path)
            n = min(v_audio.shape[1], a_audio.shape[1])
            mix_audio = v_audio[:, :n] + a_audio[:, :n]

        duration = self.duration * sr
        if mix_audio.shape[1] < duration:
            return None

        # Deterministic crop from start (matches caching window).
        mix_clip = mix_audio[:, :duration].squeeze(0).contiguous()  # (T,) fp32, leaf

        sample = {
            "emb": (whisper_emb, crepe_emb, chord_emb, beat_emb),
            "mix": mix_clip,
            "label": idx_row['source'],
            "song_id": song_id,
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
    song_ids = [item.get('song_id') for item in batch]

    whisper_batch    = torch.stack([emb[0] for emb in embeddings])
    crepe_batch      = torch.stack([emb[1] for emb in embeddings])
    chord_batch      = torch.stack([emb[2] for emb in embeddings])
    beat_this_batch  = torch.stack([emb[3] for emb in embeddings])
    mix_batch        = torch.stack([item['mix'] for item in batch])  # (B, T) fp32

    return {
        'emb': (whisper_batch, crepe_batch, chord_batch, beat_this_batch),
        'mix': mix_batch,
        'label': labels,
        'song_id': song_ids,
    }


def get_cached_dataloader(split, data_configs, train_configs, shuffle=True,
                          mix_source=None):
    """Build a DataLoader over the cached split.

    `mix_source` overrides whatever's in data_configs["mix_source"]. If both are
    None, CachedDataset falls back to "stems_sum" (default — vocals+accompaniment
    fed to MERT). Pass "mp3" here (or set data.mix_source: mp3 in the YAML) to
    feed MERT the original non-source-separated mp3 mix instead — required for
    training/scoring post-b66ca5e checkpoints.
    """
    dataset = CachedDataset(data_configs, split, mix_source=mix_source)

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

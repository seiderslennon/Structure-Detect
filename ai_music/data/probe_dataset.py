import torch
import torchaudio
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class ProbeDataset(Dataset):
    """Loads audio clips for MERT probing. CPU only, safe for DataLoader workers."""
    
    def __init__(self, tracks, data_root, sr=24000, duration=10):
        self.tracks = tracks
        self.data_root = Path(data_root)
        self.sr = sr
        self.duration = duration * sr

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        row = self.tracks.iloc[idx]
        v_path = self.data_root / row['source'] / str(row['filename']) / 'vocals.wav'
        a_path = self.data_root / row['source'] / str(row['filename']) / 'accompaniment.wav'

        try:
            v_audio, sr = torchaudio.load(v_path)
            a_audio, sr = torchaudio.load(a_path)
        except Exception:
            return None

        if v_audio.shape[0] > 1:
            v_audio = v_audio.float().mean(dim=0, keepdim=True)
            a_audio = a_audio.float().mean(dim=0, keepdim=True)

        if sr != self.sr:
            transform = torchaudio.transforms.Resample(sr, self.sr)
            v_audio = transform(v_audio)
            a_audio = transform(a_audio)

        if v_audio.shape[1] < self.duration or v_audio.shape[1] != a_audio.shape[1]:
            return None

        mix = v_audio[:, :self.duration] + a_audio[:, :self.duration]
        label = 1 if row['source'] == 'real' else 0
        return mix.squeeze(0), label


def probe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    mixes = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    return {'mix': mixes, 'label': labels}


def get_tracks(csv_path, data_root, split, split_ratio=0.9):
    df = pd.read_csv(csv_path)
    data_root = Path(data_root)
    df = df[df.apply(lambda r:
        (data_root / r['source'] / str(r['filename']) / 'vocals.wav').exists() and
        (data_root / r['source'] / str(r['filename']) / 'accompaniment.wav').exists(),
        axis=1)].reset_index(drop=True)

    real = df[df['source'] == 'real'].sample(frac=1, random_state=42).reset_index(drop=True)
    fake = df[df['source'] == 'fake'].sample(frac=1, random_state=42).reset_index(drop=True)
    ri, fi = int(split_ratio * len(real)), int(split_ratio * len(fake))

    if split == 'train':
        return pd.concat([real[:ri], fake[:fi]]).sample(frac=1, random_state=42).reset_index(drop=True)
    elif split == 'val':
        return pd.concat([real[ri:], fake[fi:]]).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        return df


def get_probe_dataloader(tracks, data_root, sr=24000, duration=10,
                         batch_size=16, num_workers=4):
    dataset = ProbeDataset(tracks, data_root, sr=sr, duration=duration)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=probe_collate, pin_memory=True, shuffle=False)
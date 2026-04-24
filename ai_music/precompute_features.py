"""
Cache the four specialized features (whisper, crepe, chord, beat) per song.
MERT is NOT cached — run on-the-fly during training (one fast forward pass).

Saves ~5MB per song in float16 as individual .pt files. Resume-safe: skips existing files.

Usage:
    CUDA_VISIBLE_DEVICES=1 python precompute_features.py \
        --config ai_music/configs/SpecTTTra.yaml --split train --num_workers 4
"""

import sys
sys.path.insert(0, "/home/lennon/AI_music")
sys.path.insert(0, "/home/lennon/AI_music/beat_this")
sys.path.insert(0, "/home/lennon/AI_music/ISMIR2019-Large-Vocabulary-Chord-Recognition")

import torch
import torchaudio
import torchcrepe
import whisper
import numpy as np
import pandas as pd
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
from feature_extractor import FeatureExtractor
from beat_this.inference import load_model as load_beat_model
from beat_this.preprocessing import LogMelSpect


class AudioOnlyDataset(torch.utils.data.Dataset):
    """Loads audio on CPU. No GPU models."""
    
    def __init__(self, tracks, data_root, sr=24000, duration=60):
        self.tracks = tracks
        self.data_root = Path(data_root)
        self.sr = sr
        self.duration_samples = duration * sr
    
    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, idx):
        row = self.tracks.iloc[idx]
        song_id = str(row['filename'])
        source = row['source']
        v_path = self.data_root / source / song_id / 'vocals.wav'
        a_path = self.data_root / source / song_id / 'accompaniment.wav'
        
        try:
            v_audio, sr = torchaudio.load(v_path)
            a_audio, _ = torchaudio.load(a_path)
        except Exception:
            return None

        if v_audio.shape[0] > 1:
            v_audio = v_audio.float().mean(dim=0, keepdim=True)
            a_audio = a_audio.float().mean(dim=0, keepdim=True)

        if sr != self.sr:
            resample = torchaudio.transforms.Resample(sr, self.sr)
            v_audio = resample(v_audio)
            a_audio = resample(a_audio)

        if v_audio.shape[1] < self.duration_samples or v_audio.shape[1] != a_audio.shape[1]:
            return None

        v_clip = v_audio[:, :self.duration_samples]
        a_clip = a_audio[:, :self.duration_samples]
        label = 1 if source == 'real' else 0

        return {
            'vocal': v_clip.squeeze(0),
            'accomp': a_clip.squeeze(0),
            'label': label,
            'song_id': f"{source}_{song_id}"
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='/home/lennon/AI_music/ai_music/configs/SpecTTTra.yaml')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val'])
    parser.add_argument('--output_dir', type=str, default='/data/structture/cached_features')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        configs = yaml.safe_load(f)
    data_config = configs['data']

    csv_path = Path(data_config['data_root'])
    data_root = csv_path.parent
    df = pd.read_csv(csv_path)

    def exists(row):
        base = data_root / row['source'] / str(row['filename'])
        return (base / 'vocals.wav').exists() and (base / 'accompaniment.wav').exists()
    df = df[df.apply(exists, axis=1)].reset_index(drop=True)

    split_ratio = 0.9
    real_df = df[df['source'] == 'real'].sample(frac=1, random_state=42).reset_index(drop=True)
    fake_df = df[df['source'] == 'fake'].sample(frac=1, random_state=42).reset_index(drop=True)

    if args.split == 'train':
        tracks = pd.concat([
            real_df[:int(split_ratio * len(real_df))],
            fake_df[:int(split_ratio * len(fake_df))]
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        tracks = pd.concat([
            real_df[int(split_ratio * len(real_df)):],
            fake_df[int(split_ratio * len(fake_df)):]
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

    save_dir = Path(args.output_dir) / args.split
    save_dir.mkdir(parents=True, exist_ok=True)

    # Check what's already cached
    existing = set(p.stem for p in save_dir.glob('*.pt'))
    print(f"Split: {args.split}, total: {len(tracks)}, already cached: {len(existing)}")

    dataset = AudioOnlyDataset(tracks, data_root, sr=data_config['sample_rate'],
                                duration=data_config['duration'])
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=args.num_workers,
        collate_fn=lambda b: b[0], pin_memory=True, shuffle=False)

    # Load models
    sr = data_config['sample_rate']
    print("Loading models (no MERT — it runs on-the-fly during training)...")
    whisper_model = whisper.load_model(data_config.get('whisper_size', 'small'), device='cuda')
    chordnet = FeatureExtractor()
    beat_model = load_beat_model('/home/lennon/AI_music/beat_this/final0.ckpt', device='cuda')
    bt_spec = LogMelSpect(sample_rate=22050, n_fft=1024, hop_length=441, n_mels=128, device='cuda')
    print("Models loaded.")

    count = 0
    skipped = 0

    # Resamplers: master rate (24kHz) -> each model's native rate
    resample_16k = torchaudio.transforms.Resample(sr, 16000)
    resample_22k = torchaudio.transforms.Resample(sr, 22050)

    for sample in tqdm(loader, desc=f"Caching {args.split}"):
        if sample is None:
            continue

        song_id = sample['song_id']
        if song_id in existing:
            skipped += 1
            continue

        vocal = sample['vocal']     # (T,) at 24kHz
        accomp = sample['accomp']   # (T,) at 24kHz
        label = sample['label']

        try:
            # Resample for each model
            vocal_16k = resample_16k(vocal)      # Whisper, Crepe expect 16kHz
            accomp_22k = resample_22k(accomp)    # Chord-Net, Beat This expect 22050

            # Whisper (16kHz vocal, 30s chunks)
            whisper_embs = []
            chunk_len = 16000 * 30
            for i in range(0, vocal_16k.shape[0], chunk_len):
                chunk = vocal_16k[i:i + chunk_len]
                if chunk.shape[0] < chunk_len:
                    break
                mel = whisper.log_mel_spectrogram(chunk)
                with torch.no_grad():
                    out = whisper_model.encoder(mel.unsqueeze(0).cuda())
                    whisper_embs.append(out.cpu())
            whisper_emb = torch.cat(whisper_embs, dim=1).squeeze(0)  # (T_w, 384)

            # Crepe (16kHz vocal)
            crepe_emb = torchcrepe.embed(
                vocal_16k.unsqueeze(0), 16000,
                hop_length=int(16000 / 100),
                model="tiny", batch_size=512,
                device='cuda', pad=True
            ).flatten(start_dim=2)
            crepe_emb = crepe_emb[:, 1:, :].squeeze(0).cpu()  # (T_c, 256)

            # Chord-Net (22050)
            chord_emb = chordnet.extract_features_from_audio(accomp_22k.numpy(), 22050)
            chord_emb = torch.from_numpy(chord_emb.astype(np.float32))  # (T_ch, 240)

            # Beat This! (22050)
            mono_22k = accomp_22k.cuda()
            spect = bt_spec(mono_22k).unsqueeze(0)
            with torch.inference_mode():
                beat_emb = beat_model(spect)["feat"].squeeze(0).cpu()  # (T_b, 512)

            # Save as float16
            torch.save({
                'whisper': whisper_emb.half(),
                'crepe': crepe_emb.half(),
                'chord': chord_emb.half(),
                'beat': beat_emb.half(),
                'label': label,
            }, save_dir / f"{song_id}.pt")

            count += 1

        except Exception as e:
            print(f"Error on {song_id}: {e}")
            continue

    print(f"Done. Cached {count} new, skipped {skipped} existing. Total in {save_dir}: {count + skipped}")


if __name__ == '__main__':
    main()
"""
Cache the four specialized features (whisper, crepe, chord, beat) for the
flat-layout SONICS EvalSet: source-separated stems live under
``<ssep_root>/<song_id>/{vocals,accompaniment}.wav`` (no real_songs / fake_songs
split, no labels here — those come from a sidecar CSV at score time).

Mirrors the schema written by ``precompute_features.py`` so EvalSetCachedDataset
can load these .pt files directly:
    {'whisper': fp16(T_w, 384),
     'crepe':   fp16(T_c, 256),
     'chord':   fp16(T_ch, 240),
     'beat':    fp16(T_b, 512),
     'label':   None}      # filled at score time from labels CSV (if any)

Resume-safe (skips existing .pt files). MERT is *not* cached — it runs on the
original mp3 mix from /data/SONICS/EvalSet/<song_id>.mp3 inside the LightningModel.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m ai_music.precompute_features_evalset \
        --ssep_root /data/SONICS/ssep_EvalSet \
        --output_dir /data/structture/cached_features/ssep_evalset \
        --num_workers 4
"""

import sys
sys.path.insert(0, "/home/lennon/AI_music")
sys.path.insert(0, "/home/lennon/AI_music/beat_this")
sys.path.insert(0, "/home/lennon/AI_music/ISMIR2019-Large-Vocabulary-Chord-Recognition")

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchcrepe
import whisper
import yaml
from tqdm import tqdm

from feature_extractor import FeatureExtractor
from beat_this.inference import load_model as load_beat_model
from beat_this.preprocessing import LogMelSpect


class EvalSetAudioDataset(torch.utils.data.Dataset):
    """Loads the source-separated stems for a flat eval-set layout on CPU."""

    def __init__(self, song_dirs, sr=24000, duration=60):
        self.song_dirs = song_dirs
        self.sr = sr
        self.duration_samples = duration * sr

    def __len__(self):
        return len(self.song_dirs)

    def __getitem__(self, idx):
        song_dir = Path(self.song_dirs[idx])
        song_id = song_dir.name
        v_path = song_dir / "vocals.wav"
        a_path = song_dir / "accompaniment.wav"

        try:
            v_audio, sr = torchaudio.load(v_path)
            a_audio, _ = torchaudio.load(a_path)
        except Exception:
            return None

        if v_audio.shape[0] > 1:
            v_audio = v_audio.float().mean(dim=0, keepdim=True)
            a_audio = a_audio.float().mean(dim=0, keepdim=True)
        else:
            v_audio = v_audio.float()
            a_audio = a_audio.float()

        if sr != self.sr:
            resample = torchaudio.transforms.Resample(sr, self.sr)
            v_audio = resample(v_audio)
            a_audio = resample(a_audio)

        # Pad short clips with zeros rather than dropping them — the eval set is
        # small and we'd rather score every song than skip a few.
        if v_audio.shape[1] < self.duration_samples or a_audio.shape[1] < self.duration_samples:
            v_audio = torch.nn.functional.pad(
                v_audio, (0, max(0, self.duration_samples - v_audio.shape[1]))
            )
            a_audio = torch.nn.functional.pad(
                a_audio, (0, max(0, self.duration_samples - a_audio.shape[1]))
            )

        v_clip = v_audio[:, : self.duration_samples]
        a_clip = a_audio[:, : self.duration_samples]

        return {
            "vocal": v_clip.squeeze(0),
            "accomp": a_clip.squeeze(0),
            "song_id": song_id,
        }


def collect_song_dirs(ssep_root: Path):
    return sorted(
        d for d in ssep_root.iterdir()
        if d.is_dir() and (d / "vocals.wav").exists() and (d / "accompaniment.wav").exists()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="/home/lennon/AI_music/ai_music/configs/SpecTTTra.yaml",
        help="Used only for sample_rate / duration / whisper_size — must match training.",
    )
    parser.add_argument(
        "--ssep_root", default="/data/SONICS/ssep_EvalSet",
        help="Folder of <song_id>/{vocals,accompaniment}.wav directories.",
    )
    parser.add_argument(
        "--output_dir", default="/data/structture/cached_features/ssep_evalset",
        help="Where to write <song_id>.pt feature caches.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        data_config = yaml.safe_load(f)["data"]
    sr = data_config["sample_rate"]
    duration = data_config["duration"]

    ssep_root = Path(args.ssep_root)
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    song_dirs = collect_song_dirs(ssep_root)
    if not song_dirs:
        raise RuntimeError(f"No song dirs with vocals+accompaniment under {ssep_root}")

    existing = {p.stem for p in save_dir.glob("*.pt")}
    print(
        f"ssep_root: {ssep_root}\n"
        f"output_dir: {save_dir}\n"
        f"songs: {len(song_dirs)}, already cached: {len(existing)}"
    )

    dataset = EvalSetAudioDataset(song_dirs, sr=sr, duration=duration)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=lambda b: b[0],
        pin_memory=True,
        shuffle=False,
    )

    print("Loading models (no MERT — runs on-the-fly during inference)...")
    whisper_model = whisper.load_model(data_config.get("whisper_size", "tiny"), device="cuda")
    chordnet = FeatureExtractor()
    beat_model = load_beat_model("/home/lennon/AI_music/beat_this/final0.ckpt", device="cuda")
    bt_spec = LogMelSpect(sample_rate=22050, n_fft=1024, hop_length=441, n_mels=128, device="cuda")
    print("Models loaded.")

    resample_16k = torchaudio.transforms.Resample(sr, 16000)
    resample_22k = torchaudio.transforms.Resample(sr, 22050)

    count = 0
    skipped = 0
    failed = 0

    for sample in tqdm(loader, desc="Caching ssep_evalset"):
        if sample is None:
            failed += 1
            continue

        song_id = sample["song_id"]
        if song_id in existing:
            skipped += 1
            continue

        vocal = sample["vocal"]
        accomp = sample["accomp"]

        try:
            vocal_16k = resample_16k(vocal)
            accomp_22k = resample_22k(accomp)

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
            if not whisper_embs:
                # duration shorter than 30s; pad to one chunk
                pad_len = chunk_len - vocal_16k.shape[0]
                chunk = torch.nn.functional.pad(vocal_16k, (0, pad_len))
                mel = whisper.log_mel_spectrogram(chunk)
                with torch.no_grad():
                    whisper_embs.append(whisper_model.encoder(mel.unsqueeze(0).cuda()).cpu())
            whisper_emb = torch.cat(whisper_embs, dim=1).squeeze(0)

            crepe_emb = torchcrepe.embed(
                vocal_16k.unsqueeze(0), 16000,
                hop_length=int(16000 / 100),
                model="tiny", batch_size=512,
                device="cuda", pad=True,
            ).flatten(start_dim=2)
            crepe_emb = crepe_emb[:, 1:, :].squeeze(0).cpu()

            chord_emb = chordnet.extract_features_from_audio(accomp_22k.numpy(), 22050)
            chord_emb = torch.from_numpy(chord_emb.astype(np.float32))

            mono_22k = accomp_22k.cuda()
            spect = bt_spec(mono_22k).unsqueeze(0)
            with torch.inference_mode():
                beat_emb = beat_model(spect)["feat"].squeeze(0).cpu()

            torch.save(
                {
                    "whisper": whisper_emb.detach().half(),
                    "crepe": crepe_emb.detach().half(),
                    "chord": chord_emb.detach().half(),
                    "beat": beat_emb.detach().half(),
                    "label": None,
                },
                save_dir / f"{song_id}.pt",
            )
            count += 1
        except Exception as e:
            failed += 1
            print(f"Error on {song_id}: {e}")
            continue

    print(
        f"Done. cached={count}  skipped_existing={skipped}  failed={failed}  "
        f"total_in_dir={count + skipped}"
    )


if __name__ == "__main__":
    main()

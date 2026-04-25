"""
Representational similarity (CKA) between MERT layers and domain-specific models.

For each MERT layer, computes CKA similarity with Crepe, Chord-Net, Beat This!, and Whisper.
Uses the same audio samples so representations are comparable.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m ai_music.probe_cka \
        --csv /data/SONICS/dataset_50k/combined_songs.csv \
        --data_root /data/SONICS/dataset_50k \
        --ood_results /data/structture/probing_results/attention/results.json \
        --batch_size 8 --num_workers 4 --max_samples 1000
"""

import sys
sys.path.insert(0, "/home/lennon/AI_music")
sys.path.insert(0, "/home/lennon/AI_music/beat_this")
sys.path.insert(0, "/home/lennon/AI_music/ISMIR2019-Large-Vocabulary-Chord-Recognition")

import torch
import torch.nn.functional as F
import torchaudio
import torchcrepe
import whisper
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from beat_this.inference import load_model as load_beat_model
from beat_this.preprocessing import LogMelSpect
from feature_extractor import FeatureExtractor


# ── CKA computation ─────────────────────────────────────────────────────

def linear_cka(X, Y):
    """
    Linear CKA between two representation matrices.
    X: (N, D1), Y: (N, D2) — N samples, different feature dims.
    Returns scalar similarity in [0, 1].
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


# ── Dataset ──────────────────────────────────────────────────────────────

class AudioPairDataset(Dataset):
    """Loads vocal and accompaniment for all models."""
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
        v = v[:, :self.duration].squeeze(0)
        a = a[:, :self.duration].squeeze(0)
        mix = v + a
        return {'vocal': v, 'accomp': a, 'mix': mix}


def pair_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        'vocal': torch.stack([b['vocal'] for b in batch]),
        'accomp': torch.stack([b['accomp'] for b in batch]),
        'mix': torch.stack([b['mix'] for b in batch]),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--ood_results', type=str,
                        default='/data/structture/probing_results/attention/results.json')
    parser.add_argument('--output_dir', type=str, default='/data/structture/cka_results')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--duration', type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load OOD detection curve
    ood_auc_curve = None
    if Path(args.ood_results).exists():
        with open(args.ood_results) as f:
            det = json.load(f)
        ood_auc_curve = [det[f'layer_{i}']['ood']['auc']['mean'] for i in range(13)]
        print(f"OOD AUC curve: {[f'{v:.3f}' for v in ood_auc_curve]}")

    # Load tracks
    df = pd.read_csv(args.csv)
    data_root = Path(args.data_root)
    df = df[df.apply(lambda r:
        (data_root / r['source'] / str(r['filename']) / 'vocals.wav').exists() and
        (data_root / r['source'] / str(r['filename']) / 'accompaniment.wav').exists(),
        axis=1)].reset_index(drop=True)

    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=42).reset_index(drop=True)

    print(f"Samples: {len(df)}")

    dataset = AudioPairDataset(df, data_root, duration=args.duration)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                       collate_fn=pair_collate, pin_memory=True, shuffle=False)

    # Load all models
    print("Loading models...")
    mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).cuda().eval()
    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    whisper_model = whisper.load_model('tiny', device='cuda')
    chordnet = FeatureExtractor()
    beat_model = load_beat_model('/home/lennon/AI_music/beat_this/final0.ckpt', device='cuda')
    bt_spec = LogMelSpect(sample_rate=22050, n_fft=1024, hop_length=441, n_mels=128, device='cuda')
    print("Models loaded.")

    resample_16k = torchaudio.transforms.Resample(24000, 16000)
    resample_22k = torchaudio.transforms.Resample(24000, 22050)

    # ── Extract all representations ──────────────────────────────────────

    all_mert_layers = [[] for _ in range(13)]
    all_crepe = []
    all_chord = []
    all_beat = []
    all_whisper = []

    print("\nExtracting representations...")
    torch.set_grad_enabled(False)  # disable grad for all extraction
    for batch in tqdm(loader, desc="  Extracting"):
        if batch is None:
            continue

        B = batch['mix'].shape[0]

        for i in range(B):
            vocal = batch['vocal'][i]       # (T,) at 24kHz
            accomp = batch['accomp'][i]     # (T,) at 24kHz
            mix = batch['mix'][i]           # (T,) at 24kHz

            if len(all_crepe) == 0 and i == 0:
                print(f"  DEBUG: vocal shape={vocal.shape}, accomp shape={accomp.shape}, mix shape={mix.shape}")

            vocal_16k = resample_16k(vocal).detach()
            accomp_22k = resample_22k(accomp).detach()

            try:
                # MERT (full mix)
                inputs = mert_processor(mix.numpy(), sampling_rate=24000, return_tensors="pt")
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                out = mert_model(**inputs, output_hidden_states=True)
                for l in range(13):
                    layer_out = out.hidden_states[l].squeeze(0).cpu().numpy()
                    pooled = layer_out.mean(axis=0)
                    all_mert_layers[l].append(pooled)

                # Crepe (vocal 16kHz)
                crepe_emb = torchcrepe.embed(
                    vocal_16k.unsqueeze(0), 16000,
                    hop_length=160, model="tiny", batch_size=512,
                    device='cuda', pad=True
                ).flatten(start_dim=2).squeeze(0).cpu().numpy()
                all_crepe.append(crepe_emb.mean(axis=0))

                # Beat This! (accomp 22kHz)
                mono_22k = accomp_22k.cuda()
                spect = bt_spec(mono_22k).unsqueeze(0)
                beat_feat = beat_model(spect)["feat"].squeeze(0).cpu().numpy()
                all_beat.append(beat_feat.mean(axis=0))

                # Whisper (vocal 16kHz) — pad to 30s as Whisper expects
                chunk = whisper.pad_or_trim(vocal_16k)  # pads/trims to 480000 (30s at 16kHz)
                mel = whisper.log_mel_spectrogram(chunk)
                whisper_out = whisper_model.encoder(mel.unsqueeze(0).cuda())
                all_whisper.append(whisper_out.squeeze(0).cpu().numpy().mean(axis=0))

                # Chord-Net (accomp resampled to 22050, matching precompute_features.py)
                chord_feat = chordnet.extract_features_from_audio(accomp_22k.detach().numpy(), 22050)
                all_chord.append(chord_feat.mean(axis=0))

            except Exception as e:
                import traceback
                if len(all_crepe) < 3:  # only print detail for first few
                    print(f"  Error on sample {i}:")
                    traceback.print_exc()
                else:
                    print(f"  Error on sample {i}: {e}")
                # Keep lists aligned — remove any partial additions
                min_len = min(len(all_crepe), len(all_beat), len(all_whisper), len(all_chord))
                for l in range(13):
                    while len(all_mert_layers[l]) > min_len:
                        all_mert_layers[l].pop()
                while len(all_crepe) > min_len: all_crepe.pop()
                while len(all_beat) > min_len: all_beat.pop()
                while len(all_whisper) > min_len: all_whisper.pop()
                while len(all_chord) > min_len: all_chord.pop()
                continue

    N = len(all_crepe)
    torch.set_grad_enabled(True)  # re-enable for any later use
    print(f"\n{N} samples extracted successfully.")

    # Free GPU
    del mert_model, mert_processor, whisper_model, beat_model
    torch.cuda.empty_cache()

    # ── Compute CKA ──────────────────────────────────────────────────────

    print("\n=== Computing CKA ===")

    crepe_mat = np.stack(all_crepe)       # (N, 256)
    chord_mat = np.stack(all_chord)       # (N, 240)
    beat_mat = np.stack(all_beat)         # (N, 512)
    whisper_mat = np.stack(all_whisper)   # (N, 384)

    domain_models = {
        'Crepe (pitch)': crepe_mat,
        'Chord-Net (chords)': chord_mat,
        'Beat This! (rhythm)': beat_mat,
        'Whisper (lyrics)': whisper_mat,
    }

    cka_results = {name: [] for name in domain_models}

    for layer in range(13):
        mert_mat = np.stack(all_mert_layers[layer])  # (N, 768)
        for name, domain_mat in domain_models.items():
            cka = linear_cka(mert_mat, domain_mat)
            cka_results[name].append(float(cka))
        print(f"  Layer {layer:2d} — " + 
              "  ".join(f"{name.split()[0]}: {cka_results[name][layer]:.3f}" for name in domain_models))

    # ── Spearman correlation with OOD detection ──────────────────────────

    print("\n=== Spearman Correlation: CKA curves vs OOD Detection ===")
    correlations = {}
    if ood_auc_curve:
        for name, cka_curve in cka_results.items():
            rho, pval = spearmanr(cka_curve, ood_auc_curve)
            correlations[name] = {'rho': float(rho), 'p_value': float(pval)}
            print(f"  {name:25s}  rho = {rho:+.3f}  (p = {pval:.4f})")

    # ── Save ─────────────────────────────────────────────────────────────

    output = {
        'cka_results': cka_results,
        'correlations': correlations,
        'ood_auc_curve': ood_auc_curve,
        'n_samples': N,
    }
    with open(output_dir / 'cka_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Summary table
    print(f"\n{'Layer':<8}", end="")
    for name in domain_models:
        short = name.split('(')[0].strip()
        print(f"{short:<14}", end="")
    print()
    print("-" * (8 + 14 * len(domain_models)))
    for i in range(13):
        print(f"{i:<8}", end="")
        for name in domain_models:
            print(f"{cka_results[name][i]:<14.3f}", end="")
        print()

    print(f"\nSaved to {output_dir / 'cka_results.json'}")


if __name__ == '__main__':
    main()
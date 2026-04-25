"""
MIR property probing across MERT layers.
Probes MERT on pitch (NSynth), tempo (GTZAN), and genre (GTZAN).
Computes Spearman correlation with OOD detection AUC curve.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m ai_music.probe_mir_tasks \
        --nsynth_dir /data/nsynth-test \
        --gtzan_audio /data/tempo-datasets/gtzan/audio \
        --gtzan_tempo /data/tempo-datasets/gtzan/tempo \
        --ood_results /data/structture/probing_results/attention/results.json \
        --batch_size 16 --num_workers 4 --n_seeds 3
"""

import sys
sys.path.insert(0, "/home/lennon/AI_music")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from transformers import Wav2Vec2FeatureExtractor, AutoModel


# ── Datasets ─────────────────────────────────────────────────────────────

class NSynthDataset(Dataset):
    """NSynth pitch classification. Each sample is a 4s audio clip."""
    def __init__(self, nsynth_dir, sr=24000, max_samples=None):
        self.sr = sr
        nsynth_dir = Path(nsynth_dir)
        
        with open(nsynth_dir / 'examples.json') as f:
            meta = json.load(f)
        
        self.samples = []
        for name, info in meta.items():
            wav_path = nsynth_dir / 'audio' / f'{name}.wav'
            if wav_path.exists():
                self.samples.append({
                    'path': wav_path,
                    'pitch': info['pitch'],  # MIDI pitch 21-108
                })
        
        if max_samples and len(self.samples) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]
        
        # Map pitches to contiguous class indices
        all_pitches = sorted(set(s['pitch'] for s in self.samples))
        self.pitch_to_idx = {p: i for i, p in enumerate(all_pitches)}
        self.num_classes = len(all_pitches)
        print(f"NSynth: {len(self.samples)} samples, {self.num_classes} pitch classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            audio, sr = torchaudio.load(s['path'])
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
            if sr != self.sr:
                audio = torchaudio.transforms.Resample(sr, self.sr)(audio)
            return audio.squeeze(0), self.pitch_to_idx[s['pitch']]
        except Exception:
            return None


class GTZANTempoDataset(Dataset):
    """GTZAN tempo classification. BPM binned into classes."""
    def __init__(self, audio_dir, tempo_dir, sr=24000, duration=10, max_samples=None):
        self.sr = sr
        self.duration = duration * sr
        audio_dir = Path(audio_dir)
        tempo_dir = Path(tempo_dir)

        self.samples = []
        for bpm_file in sorted(tempo_dir.glob('*.bpm')):
            # Parse filename: gtzan_blues_00000.bpm -> blues/blues.00000.wav
            parts = bpm_file.stem.split('_')  # ['gtzan', 'blues', '00000']
            if len(parts) < 3:
                continue
            genre = parts[1]
            num = parts[2]
            wav_path = audio_dir / genre / f'{genre}.{num}.wav'
            
            if wav_path.exists():
                bpm = float(bpm_file.read_text().strip())
                self.samples.append({'path': wav_path, 'bpm': bpm})

        if max_samples and len(self.samples) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]

        # Bin BPM into 8 classes
        bpms = np.array([s['bpm'] for s in self.samples])
        self.bin_edges = np.percentile(bpms, np.linspace(0, 100, 9))
        for s in self.samples:
            s['class'] = min(np.searchsorted(self.bin_edges[1:-1], s['bpm']), 7)
        
        self.num_classes = 8
        print(f"GTZAN Tempo: {len(self.samples)} samples, {self.num_classes} tempo bins")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            audio, sr = torchaudio.load(s['path'])
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
            if sr != self.sr:
                audio = torchaudio.transforms.Resample(sr, self.sr)(audio)
            if audio.shape[1] > self.duration:
                audio = audio[:, :self.duration]
            elif audio.shape[1] < self.duration:
                audio = F.pad(audio, (0, self.duration - audio.shape[1]))
            return audio.squeeze(0), s['class']
        except Exception:
            return None


class GTZANGenreDataset(Dataset):
    """GTZAN genre classification from directory structure."""
    def __init__(self, audio_dir, sr=24000, duration=10, max_samples=None):
        self.sr = sr
        self.duration = duration * sr
        audio_dir = Path(audio_dir)

        self.genre_to_idx = {}
        self.samples = []
        
        for genre_dir in sorted(audio_dir.iterdir()):
            if not genre_dir.is_dir():
                continue
            genre = genre_dir.name
            if genre not in self.genre_to_idx:
                self.genre_to_idx[genre] = len(self.genre_to_idx)
            for wav in sorted(genre_dir.glob('*.wav')):
                self.samples.append({'path': wav, 'genre': genre})

        if max_samples and len(self.samples) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]

        self.num_classes = len(self.genre_to_idx)
        print(f"GTZAN Genre: {len(self.samples)} samples, {self.num_classes} genres: {list(self.genre_to_idx.keys())}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            audio, sr = torchaudio.load(s['path'])
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
            if sr != self.sr:
                audio = torchaudio.transforms.Resample(sr, self.sr)(audio)
            if audio.shape[1] > self.duration:
                audio = audio[:, :self.duration]
            elif audio.shape[1] < self.duration:
                audio = F.pad(audio, (0, self.duration - audio.shape[1]))
            return audio.squeeze(0), self.genre_to_idx[s['genre']]
        except Exception:
            return None


def mir_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    audio = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    return audio, labels


# ── Probe ────────────────────────────────────────────────────────────────

class AttentionProbeMulticlass(nn.Module):
    def __init__(self, dim=768, num_classes=10):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.key_proj = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, num_classes)
    def forward(self, x):
        keys = self.key_proj(x)
        query = self.query.expand(x.shape[0], -1, -1)
        w = F.softmax(torch.bmm(query, keys.transpose(1, 2)) / (x.shape[-1] ** 0.5), dim=-1)
        pooled = torch.bmm(w, x).squeeze(1)
        return self.linear(pooled)


# ── Training ─────────────────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def train_multiclass_probe(probe, tx, ty, vx, vy, lr=1e-3, bs=64, epochs=30, patience=5):
    probe = probe.cuda()
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    ty = ty.long()
    vy_cpu = vy.long()

    best_acc, best_state, wait = 0, None, 0
    n = len(tx)

    for _ in range(epochs):
        perm = torch.randperm(n)
        probe.train()
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            bx = tx[idx].cuda()
            by = ty[idx].cuda()
            loss = F.cross_entropy(probe(bx), by)
            opt.zero_grad(); loss.backward(); opt.step()

        probe.eval()
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(vx), bs):
                bx = vx[i:i+bs].cuda()
                all_preds.append(probe(bx).argmax(1).cpu())
        preds = torch.cat(all_preds).numpy()
        acc = accuracy_score(vy_cpu.numpy(), preds)

        if acc > best_acc:
            best_acc, best_state, wait = acc, {k: v.clone() for k, v in probe.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break

    probe.load_state_dict(best_state)
    return probe, best_acc


def eval_multiclass(probe, x, y, bs=64):
    probe.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            bx = x[i:i+bs].cuda()
            all_preds.append(probe(bx).argmax(1).cpu())
    preds = torch.cat(all_preds).numpy()
    return float(accuracy_score(y.numpy(), preds))


# ── MERT extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_mert_features(loader, mert_model, mert_processor, sr=24000, temporal_stride=10):
    """Extract MERT features from a DataLoader. Returns temporal features and labels."""
    all_temporal = []
    all_labels = []

    for batch in tqdm(loader, desc="  Extracting"):
        if batch is None:
            continue
        audio, labels = batch

        inputs = mert_processor(
            [a.numpy() for a in audio],
            sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        out = mert_model(**inputs, output_hidden_states=True)

        layers = torch.stack(out.hidden_states)       # (13, B, T, 768)
        layers = layers.permute(1, 0, 2, 3).cpu()     # (B, 13, T, 768)
        strided = layers[:, :, ::temporal_stride, :]

        all_temporal.append(strided)
        all_labels.append(labels)

    # Pad to uniform temporal length
    max_T = max(t.shape[2] for t in all_temporal)
    padded = []
    for t in all_temporal:
        if t.shape[2] < max_T:
            t = F.pad(t, (0, 0, 0, max_T - t.shape[2]))
        padded.append(t)

    features = torch.cat(padded)  # (N, 13, T', 768)
    labels = torch.cat(all_labels)
    print(f"  Features: {features.shape}, {features.nelement() * 4 / 1e9:.2f} GB")
    return features, labels


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsynth_dir', type=str, default='/data/nsynth-test')
    parser.add_argument('--gtzan_audio', type=str, default='/data/tempo-datasets/gtzan/audio')
    parser.add_argument('--gtzan_tempo', type=str, default='/data/tempo-datasets/gtzan/tempo')
    parser.add_argument('--ood_results', type=str, 
                        default='/data/structture/probing_results/attention/results.json',
                        help='Path to OOD detection probing results for Spearman correlation')
    parser.add_argument('--output_dir', type=str, default='/data/structture/mir_probing_results')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--temporal_stride', type=int, default=10)
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--nsynth_max', type=int, default=4000, help='Max NSynth samples')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load OOD detection curve for Spearman correlation
    ood_auc_curve = None
    if args.ood_results and Path(args.ood_results).exists():
        with open(args.ood_results) as f:
            det_results = json.load(f)
        ood_auc_curve = [det_results[f'layer_{i}']['ood']['auc']['mean'] 
                         for i in range(13)]
        print(f"OOD detection AUC curve loaded: {[f'{v:.3f}' for v in ood_auc_curve]}")

    # Load MERT
    print("\nLoading MERT...")
    mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).cuda().eval()
    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

    # ── Define tasks ─────────────────────────────────────────────────────

    tasks = {}

    # Pitch (NSynth)
    if Path(args.nsynth_dir).exists():
        nsynth = NSynthDataset(args.nsynth_dir, max_samples=args.nsynth_max)
        tasks['pitch'] = {'dataset': nsynth, 'num_classes': nsynth.num_classes}
    else:
        print(f"WARNING: NSynth not found at {args.nsynth_dir}, skipping pitch task")

    # Tempo (GTZAN)
    if Path(args.gtzan_tempo).exists():
        gtzan_tempo = GTZANTempoDataset(args.gtzan_audio, args.gtzan_tempo, duration=10)
        tasks['tempo'] = {'dataset': gtzan_tempo, 'num_classes': gtzan_tempo.num_classes}
    else:
        print(f"WARNING: GTZAN tempo not found, skipping tempo task")

    # Genre (GTZAN)
    if Path(args.gtzan_audio).exists():
        gtzan_genre = GTZANGenreDataset(args.gtzan_audio, duration=10)
        tasks['genre'] = {'dataset': gtzan_genre, 'num_classes': gtzan_genre.num_classes}
    else:
        print(f"WARNING: GTZAN audio not found, skipping genre task")

    # ── Extract MERT features per task ───────────────────────────────────

    print("\n=== Extracting MERT features ===")
    task_features = {}

    for task_name, task_info in tasks.items():
        print(f"\n{task_name}:")
        dataset = task_info['dataset']
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                           collate_fn=mir_collate, pin_memory=True, shuffle=False)
        features, labels = extract_mert_features(
            loader, mert_model, mert_processor, temporal_stride=args.temporal_stride)
        task_features[task_name] = (features, labels)

    del mert_model, mert_processor
    torch.cuda.empty_cache()
    print("\nMERT freed from GPU.")

    # ── Train probes per task per layer ───────────────────────────────────

    print("\n=== Training MIR probes ===")
    all_results = {}

    for task_name, task_info in tasks.items():
        features, labels = task_features[task_name]
        num_classes = task_info['num_classes']
        N = len(labels)

        # Train/val split (80/20)
        perm = torch.randperm(N)
        split = int(0.8 * N)
        train_idx, val_idx = perm[:split], perm[split:]

        print(f"\n{'='*50}")
        print(f"Task: {task_name} ({num_classes} classes, {split} train, {N-split} val)")
        print(f"{'='*50}")

        task_results = {}
        for layer in range(13):
            tx = features[train_idx, layer, :, :]
            vx = features[val_idx, layer, :, :]
            ty = labels[train_idx]
            vy = labels[val_idx]

            seed_accs = []
            for seed in range(args.n_seeds):
                set_seed(seed * 100 + layer)
                probe = AttentionProbeMulticlass(768, num_classes)
                probe, _ = train_multiclass_probe(probe, tx, ty, vx, vy)
                acc = eval_multiclass(probe, vx, vy)
                seed_accs.append(acc)
                del probe
                torch.cuda.empty_cache()

            mean_acc = float(np.mean(seed_accs))
            std_acc = float(np.std(seed_accs))
            task_results[f'layer_{layer}'] = {'mean': mean_acc, 'std': std_acc}
            print(f"  Layer {layer:2d} — Acc: {mean_acc:.3f} ± {std_acc:.3f}")

        all_results[task_name] = task_results

    # ── Spearman correlation ─────────────────────────────────────────────

    print("\n=== Spearman Correlation with OOD Detection ===")
    correlations = {}

    if ood_auc_curve:
        for task_name, task_results in all_results.items():
            task_curve = [task_results[f'layer_{i}']['mean'] for i in range(13)]
            rho, pval = spearmanr(task_curve, ood_auc_curve)
            correlations[task_name] = {'rho': float(rho), 'p_value': float(pval)}
            print(f"  {task_name:10s}  rho = {rho:+.3f}  (p = {pval:.4f})")

    # ── Save ─────────────────────────────────────────────────────────────

    output = {
        'task_results': all_results,
        'correlations': correlations,
        'ood_auc_curve': ood_auc_curve,
    }
    with open(output_dir / 'mir_probing_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Summary table
    print(f"\n{'Layer':<8}", end="")
    for task_name in all_results:
        print(f"{task_name:<18}", end="")
    print()
    print("-" * (8 + 18 * len(all_results)))
    for i in range(13):
        print(f"{i:<8}", end="")
        for task_name in all_results:
            r = all_results[task_name][f'layer_{i}']
            print(f"{r['mean']:.3f} ± {r['std']:.3f}    ", end="")
        print()

    print(f"\nSaved to {output_dir / 'mir_probing_results.json'}")


if __name__ == '__main__':
    main()
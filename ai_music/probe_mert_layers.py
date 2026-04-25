"""
MERT layer probing with seed control.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m ai_music.probe_mert_layers \
        --csv /data/SONICS/dataset_50k/combined_songs.csv \
        --data_root /data/SONICS/dataset_50k \
        --ood_csv /data/structture/ood_eval.csv \
        --ood_data_root /data/structture/ood_ssep \
        --batch_size 8 --num_workers 4 --duration 60 --max_samples 10000 --n_seeds 3
"""

import sys
sys.path.insert(0, "/home/lennon/AI_music")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from ai_music.data.probe_dataset import get_tracks, get_probe_dataloader


# ── Probes ───────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
    def forward(self, x):
        return self.linear(x).squeeze(-1)


class AttentionProbe(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.key_proj = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, 1)
    def forward(self, x):
        keys = self.key_proj(x)
        query = self.query.expand(x.shape[0], -1, -1)
        w = F.softmax(torch.bmm(query, keys.transpose(1, 2)) / (x.shape[-1] ** 0.5), dim=-1)
        return self.linear(torch.bmm(w, x).squeeze(1)).squeeze(-1)


# ── MERT extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_mert_batch(batch_mix, mert_model, mert_processor, sr=24000):
    inputs = mert_processor(
        [m.numpy() for m in batch_mix],
        sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    out = mert_model(**inputs, output_hidden_states=True)
    layers = torch.stack(out.hidden_states)       # (13, B, T, 768)
    layers = layers.permute(1, 0, 2, 3).cpu()     # (B, 13, T, 768)
    return layers


def extract_split(name, loader, mert_model, mert_processor, temporal_stride, tmp_dir, max_n):
    pooled_path = tmp_dir / f"{name}_pooled.npy"
    temporal_path = tmp_dir / f"{name}_temporal.npy"
    labels_path = tmp_dir / f"{name}_labels.npy"

    pooled_mmap = np.memmap(pooled_path, dtype='float32', mode='w+', shape=(max_n, 13, 768))
    labels_mmap = np.memmap(labels_path, dtype='int64', mode='w+', shape=(max_n,))
    temporal_mmap = None
    T_strided = None

    count = 0
    for batch in tqdm(loader, desc=f"  {name}"):
        if batch is None:
            continue

        layers = extract_mert_batch(batch['mix'], mert_model, mert_processor)
        B = layers.shape[0]

        pooled = layers.mean(dim=2).numpy()
        strided = layers[:, :, ::temporal_stride, :].numpy()

        if temporal_mmap is None:
            T_strided = strided.shape[2]
            temporal_mmap = np.memmap(temporal_path, dtype='float32', mode='w+',
                                      shape=(max_n, 13, T_strided, 768))
            print(f"  Temporal shape per sample: (13, {T_strided}, 768)")

        if strided.shape[2] > T_strided:
            strided = strided[:, :, :T_strided, :]
        elif strided.shape[2] < T_strided:
            pad = np.zeros((B, 13, T_strided - strided.shape[2], 768), dtype='float32')
            strided = np.concatenate([strided, pad], axis=2)

        pooled_mmap[count:count + B] = pooled
        temporal_mmap[count:count + B] = strided
        labels_mmap[count:count + B] = batch['label'].numpy()
        count += B

    pooled_mmap.flush()
    temporal_mmap.flush()
    labels_mmap.flush()

    size_gb = count * 13 * T_strided * 768 * 4 / 1e9
    print(f"  {name}: {count} samples, temporal {size_gb:.1f}GB on disk")

    return count, T_strided


# ── Probe training & eval ────────────────────────────────────────────────

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def train_probe(probe, tx, ty, vx, vy, lr=1e-3, bs=64, epochs=30, patience=5):
    probe = probe.cuda()
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    # Keep data on CPU, move batches to GPU
    ty = ty.float()
    vy_gpu = vy.float().cuda()

    best_auc, best_state, wait = 0, None, 0
    n = len(tx)

    for _ in range(epochs):
        perm = torch.randperm(n)
        probe.train()
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            bx = tx[idx].cuda()
            by = ty[idx].cuda()
            loss = F.binary_cross_entropy_with_logits(probe(bx), by)
            opt.zero_grad(); loss.backward(); opt.step()

        # Eval in batches too
        probe.eval()
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(vx), bs):
                bx = vx[i:i+bs].cuda()
                all_probs.append(torch.sigmoid(probe(bx)).cpu())
        val_probs = torch.cat(all_probs).numpy()
        auc = roc_auc_score(vy.numpy(), val_probs)

        if auc > best_auc:
            best_auc, best_state, wait = auc, {k: v.clone() for k, v in probe.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    probe.load_state_dict(best_state)
    return probe


def evaluate(probe, x, y, bs=64):
    probe.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            bx = x[i:i+bs].cuda()
            all_probs.append(torch.sigmoid(probe(bx)).cpu())
    probs = torch.cat(all_probs).numpy()
    preds = (probs > 0.5).astype(int)
    y = y.numpy()
    return {'accuracy': float(accuracy_score(y, preds)),
            'f1': float(f1_score(y, preds)),
            'auc': float(roc_auc_score(y, probs))}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--ood_csv', type=str, default=None)
    parser.add_argument('--ood_data_root', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='/data/structture/probing_results')
    parser.add_argument('--mode', type=str, default='attention', choices=['linear', 'attention'])
    parser.add_argument('--temporal_stride', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--duration', type=int, default=10)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--n_seeds', type=int, default=3, help='Number of seeds per layer')
    parser.add_argument('--skip_extraction', action='store_true',
                        help='Skip Phase 1, reuse existing tmp files from a previous run')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.output_dir) / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    meta_path = tmp_dir / 'meta.json'

    print(f"Temp files: {tmp_dir}")
    print(f"Mode: {args.mode}, seeds: {args.n_seeds}, duration: {args.duration}s")

    if args.skip_extraction:
        # Load saved metadata from previous Phase 1
        with open(meta_path) as f:
            split_info = {k: tuple(v) for k, v in json.load(f).items()}
        print(f"\nSkipping extraction, loaded metadata: {split_info}")
    else:
        # Load MERT
        print("Loading MERT...")
        mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).cuda().eval()
        mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

        # ── Phase 1: Extract to disk ─────────────────────────────────────────

        print("\n=== Phase 1: Extract features to disk ===")
        split_info = {}

        for split in ['train', 'val']:
            tracks = get_tracks(args.csv, args.data_root, split)
            if split == 'train' and args.max_samples and len(tracks) > args.max_samples:
                # Balanced subsample: equal real and fake
                half = args.max_samples // 2
                real_subset = tracks[tracks['source'] == 'real'].head(half)
                fake_subset = tracks[tracks['source'] == 'fake'].head(half)
                tracks = pd.concat([real_subset, fake_subset]).sample(
                    frac=1, random_state=42).reset_index(drop=True)
            max_n = len(tracks)

            # Check class balance
            n_real = (tracks['source'] == 'real').sum()
            n_fake = (tracks['source'] == 'fake').sum()
            print(f"\n{split}: {max_n} songs (real: {n_real}, fake: {n_fake}), {args.duration}s clips")

            loader = get_probe_dataloader(tracks, args.data_root, duration=args.duration,
                                           batch_size=args.batch_size, num_workers=args.num_workers)
            n, T = extract_split(split, loader, mert_model, mert_processor,
                                  args.temporal_stride, tmp_dir, max_n)
            split_info[split] = (n, T)

        if args.ood_csv:
            ood_tracks = get_tracks(args.ood_csv, args.ood_data_root, 'all')
            max_n = len(ood_tracks)
            n_real = (ood_tracks['source'] == 'real').sum()
            n_fake = (ood_tracks['source'] == 'fake').sum()
            print(f"\nOOD: {max_n} songs (real: {n_real}, fake: {n_fake})")
            loader = get_probe_dataloader(ood_tracks, args.ood_data_root, duration=args.duration,
                                           batch_size=args.batch_size, num_workers=args.num_workers)
            n, T = extract_split('ood', loader, mert_model, mert_processor,
                                  args.temporal_stride, tmp_dir, max_n)
            split_info['ood'] = (n, T)

        # Save metadata for --skip_extraction
        with open(meta_path, 'w') as f:
            json.dump(split_info, f)

        del mert_model, mert_processor
        torch.cuda.empty_cache()
        print("\nMERT freed from GPU.")

    # ── Phase 2: Train probes with multiple seeds ────────────────────────

    print(f"\n=== Phase 2: Train probes ({args.n_seeds} seeds per layer) ===")

    N_train, T_s = split_info['train']
    N_val, _ = split_info['val']

    train_pooled = np.memmap(tmp_dir / 'train_pooled.npy', dtype='float32', mode='r', shape=(N_train, 13, 768))
    train_temporal = np.memmap(tmp_dir / 'train_temporal.npy', dtype='float32', mode='r', shape=(N_train, 13, T_s, 768))
    train_labels = torch.from_numpy(np.memmap(tmp_dir / 'train_labels.npy', dtype='int64', mode='r', shape=(N_train,)).copy())

    val_pooled = np.memmap(tmp_dir / 'val_pooled.npy', dtype='float32', mode='r', shape=(N_val, 13, 768))
    val_temporal = np.memmap(tmp_dir / 'val_temporal.npy', dtype='float32', mode='r', shape=(N_val, 13, T_s, 768))
    val_labels = torch.from_numpy(np.memmap(tmp_dir / 'val_labels.npy', dtype='int64', mode='r', shape=(N_val,)).copy())

    has_ood = 'ood' in split_info
    if has_ood:
        N_ood, _ = split_info['ood']
        ood_pooled = np.memmap(tmp_dir / 'ood_pooled.npy', dtype='float32', mode='r', shape=(N_ood, 13, 768))
        ood_temporal = np.memmap(tmp_dir / 'ood_temporal.npy', dtype='float32', mode='r', shape=(N_ood, 13, T_s, 768))
        ood_labels = torch.from_numpy(np.memmap(tmp_dir / 'ood_labels.npy', dtype='int64', mode='r', shape=(N_ood,)).copy())

    all_results = {}

    for layer in range(13):
        print(f"\nLayer {layer}")

        # Load features for this layer
        if args.mode == 'linear':
            tx = torch.from_numpy(train_pooled[:, layer, :].copy())
            vx = torch.from_numpy(val_pooled[:, layer, :].copy())
        else:
            tx = torch.from_numpy(train_temporal[:, layer, :, :].copy())
            vx = torch.from_numpy(val_temporal[:, layer, :, :].copy())

        if has_ood:
            if args.mode == 'linear':
                ox = torch.from_numpy(ood_pooled[:, layer, :].copy())
            else:
                ox = torch.from_numpy(ood_temporal[:, layer, :, :].copy())

        # Run multiple seeds
        seed_results = []
        for seed in range(args.n_seeds):
            set_seed(seed * 100 + layer)

            if args.mode == 'linear':
                probe = LinearProbe(768)
            else:
                probe = AttentionProbe(768)

            probe = train_probe(probe, tx, train_labels, vx, val_labels)
            ind = evaluate(probe, vx, val_labels)
            seed_result = {'in_distribution': ind}

            if has_ood:
                ood_result = evaluate(probe, ox, ood_labels)
                seed_result['ood'] = ood_result

            seed_results.append(seed_result)

            if seed == 0:
                # Save best probe (first seed, will update if better)
                best_probe_state = {k: v.clone().cpu() for k, v in probe.state_dict().items()}
                best_ood_auc = seed_result.get('ood', {}).get('auc', ind['auc'])

            if has_ood and seed_result['ood']['auc'] > best_ood_auc:
                best_probe_state = {k: v.clone().cpu() for k, v in probe.state_dict().items()}
                best_ood_auc = seed_result['ood']['auc']

        # Aggregate across seeds
        def aggregate_metric(results, split_key, metric):
            vals = [r[split_key][metric] for r in results if split_key in r]
            return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

        layer_result = {
            'in_distribution': {
                'accuracy': aggregate_metric(seed_results, 'in_distribution', 'accuracy'),
                'f1': aggregate_metric(seed_results, 'in_distribution', 'f1'),
                'auc': aggregate_metric(seed_results, 'in_distribution', 'auc'),
            }
        }
        if has_ood:
            layer_result['ood'] = {
                'accuracy': aggregate_metric(seed_results, 'ood', 'accuracy'),
                'f1': aggregate_metric(seed_results, 'ood', 'f1'),
                'auc': aggregate_metric(seed_results, 'ood', 'auc'),
            }

        layer_result['per_seed'] = seed_results
        all_results[f'layer_{layer}'] = layer_result

        # Print mean ± std
        ia = layer_result['in_distribution']['auc']
        print(f"  In-dist  — AUC: {ia['mean']:.3f} ± {ia['std']:.3f}")
        if has_ood:
            oa = layer_result['ood']['auc']
            print(f"  OOD      — AUC: {oa['mean']:.3f} ± {oa['std']:.3f}")

        torch.save(best_probe_state, output_dir / f'probe_layer_{layer}.pt')

        # Free
        del tx, vx
        if has_ood:
            del ox
        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────────

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'Layer':<8}{'In-dist AUC':<20}", end="")
    if has_ood: print(f"{'OOD AUC':<20}{'Gap':<10}", end="")
    print("\n" + "-" * (58 if has_ood else 28))
    for i in range(13):
        r = all_results[f'layer_{i}']
        ia = r['in_distribution']['auc']
        print(f"{i:<8}{ia['mean']:.3f} ± {ia['std']:.3f}    ", end="")
        if has_ood:
            oa = r['ood']['auc']
            gap = ia['mean'] - oa['mean']
            print(f"{oa['mean']:.3f} ± {oa['std']:.3f}    {gap:.3f}", end="")
        print()

    print(f"\nResults saved to {output_dir / 'results.json'}")
    print(f"Tmp files kept at {tmp_dir} (rerun with --skip_extraction to reuse)")


if __name__ == '__main__':
    main()
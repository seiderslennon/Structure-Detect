"""
Run inference over a precomputed-feature split (val by default) and write a
scored CSV. Reuses the same cache as training, so this is ~10x faster than
infer.py (which re-extracts whisper/crepe/chord/beat per song).

Expects the split's .pt cache to already exist under
/data/structture/cached_features/<split>/. Run precompute_features.py first
if it's missing.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m ai_music.predict_cached \
        --checkpoint lightning_logs/full_model/checkpoints/last.ckpt \
        --fusion cross_attention \
        --split val \
        --csv_out ai_music/inference_results/full_model_val.csv

The script also prints an in-line metrics summary (acc / F1 / AUC / EER),
so you don't need evaluate_inference.py for dataset_50k splits — its
filename-prefix heuristic doesn't match dataset_50k's `fake_..._udio_*` IDs.
"""

import argparse
import csv
import yaml
from pathlib import Path

import numpy as np
import torch
import lightning as L

from ai_music.train import LightningModel
from ai_music.models import resnet
from ai_music.models.sonics import SpecTTTraAttentionClassifier
from ai_music.data import cross_attention
from ai_music.data.cached_dataset import get_cached_dataloader


def build_model(args, model_config, train_config):
    classifier_type = model_config.get('classifier_type', 'ResNet').lower()
    if classifier_type == 'resnet':
        classifier = resnet.ResNet(max_tokens_per_modality=model_config['max_tokens_per_modality'])
    elif classifier_type == 'spectttra':
        classifier = SpecTTTraAttentionClassifier(
            feature_dim=model_config.get('feature_dim'),
            embed_dim=model_config.get('embed_dim'),
            num_heads=model_config.get('num_heads'),
            num_layers=model_config.get('num_layers'),
            tokenizer_clip_size=model_config.get('tokenizer_clip_size'),
            num_classes=2,
            pre_norm=model_config.get('pre_norm'),
            pe_learnable=model_config.get('pe_learnable'),
            pos_drop_rate=model_config.get('pos_drop_rate'),
            attn_drop_rate=model_config.get('attn_drop_rate'),
            proj_drop_rate=model_config.get('proj_drop_rate'),
            mlp_ratio=model_config.get('mlp_ratio'),
        )
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}. Must be 'ResNet' or 'SpecTTTra'")

    if args.fusion == 'cross_attention':
        fuser = cross_attention.MultiModalMERTFusion(use_layer_mix=True)
    elif args.fusion == 'concat':
        fuser = cross_attention.ConcatLinearFusion()
    elif args.fusion == 'mert_only':
        fuser = cross_attention.MERTOnlyFusion()
    else:
        raise ValueError(
            f"Unknown fusion: {args.fusion}. Must be 'cross_attention', 'concat', or 'mert_only'"
        )

    model = LightningModel.load_from_checkpoint(
        args.checkpoint,
        classifier=classifier,
        fuser=fuser,
        configs=train_config,
        map_location='cpu',
    )
    model.eval()
    return model


def collect_predictions(outputs):
    """Flatten Lightning's per-batch dicts into per-song rows.
    Lightning preserves dataset order when shuffle=False, but we still pull
    song_ids/labels from each batch so the result stays correct even if
    the collate fn drops a sample (batch -> None)."""
    rows = []
    for batch_out in outputs:
        if batch_out is None:
            continue
        probs = batch_out['probs'].detach().cpu()
        preds = batch_out['predictions'].detach().cpu()
        song_ids = batch_out.get('song_ids') or [None] * len(preds)
        labels = batch_out.get('labels') or [None] * len(preds)

        for sid, lab, pred, prob in zip(song_ids, labels, preds, probs):
            pred_int = int(pred.item())
            real_prob = float(prob[1].item())
            fake_prob = float(prob[0].item())
            rows.append({
                'song_id': sid,
                'label': lab,
                'prediction': pred_int,
                'prediction_label': 'real' if pred_int == 1 else 'fake',
                'confidence': max(real_prob, fake_prob),
                'real_prob': real_prob,
                'fake_prob': fake_prob,
            })
    return rows


def write_csv(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'song_id', 'label', 'prediction', 'prediction_label',
        'confidence', 'real_prob', 'fake_prob',
    ]
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def print_metrics(rows):
    """Compute acc / F1 (positive=fake) / AUC / EER on the GT-labelled subset.
    Skips silently if labels are missing."""
    labelled = [r for r in rows if r['label'] in ('real', 'fake')]
    if not labelled:
        print("No ground-truth labels found in batches; skipping metrics.")
        return

    y_true_fake = np.array([1 if r['label'] == 'fake' else 0 for r in labelled])
    y_pred_fake = np.array([1 if r['prediction_label'] == 'fake' else 0 for r in labelled])
    fake_score = np.array([r['fake_prob'] for r in labelled])

    n = len(labelled)
    n_real = int((y_true_fake == 0).sum())
    n_fake = int((y_true_fake == 1).sum())
    acc = float((y_true_fake == y_pred_fake).mean())

    print()
    print(f"{'='*60}")
    print(f"  n={n}  real={n_real}  fake={n_fake}")
    print(f"  accuracy = {acc:.4f}")

    try:
        from sklearn.metrics import f1_score, roc_auc_score, roc_curve
        f1 = float(f1_score(y_true_fake, y_pred_fake, zero_division=0))
        print(f"  f1 (fake)= {f1:.4f}")
        if n_real > 0 and n_fake > 0:
            auc = float(roc_auc_score(y_true_fake, fake_score))
            fpr, tpr, _ = roc_curve(y_true_fake, fake_score)
            fnr = 1.0 - tpr
            idx = int(np.nanargmin(np.abs(fpr - fnr)))
            eer = float((fpr[idx] + fnr[idx]) / 2.0)
            print(f"  auc      = {auc:.4f}")
            print(f"  eer      = {eer:.4f}")
    except ImportError:
        print("  (install scikit-learn for f1 / auc / eer)")
    print(f"{'='*60}\n")


def main():
    p = argparse.ArgumentParser(
        description='Run inference over the precomputed-feature cache for a split.')
    p.add_argument('--config', default='/home/lennon/AI_music/ai_music/configs/SpecTTTra.yaml')
    p.add_argument('--split', default='val', choices=['train', 'val'],
                   help='Which cached split to score. Use val for held-out evaluation.')
    p.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.ckpt).')
    p.add_argument('--fusion', default='cross_attention',
                   choices=['cross_attention', 'concat', 'mert_only'],
                   help='Fusion architecture. Must match the checkpoint.')
    p.add_argument('--batch_size', type=int, default=None,
                   help='Override batch_size from train config (default: use config).')
    p.add_argument('--num_workers', type=int, default=None,
                   help='Override num_workers from train config (default: use config).')
    p.add_argument('--mix_source', default=None, choices=['mp3', 'stems_sum'],
                   help="Override data.mix_source from the YAML. The pipeline default "
                        "is 'stems_sum' (vocals+accompaniment fed to MERT). Pass 'mp3' "
                        "to score post-b66ca5e checkpoints that were trained on the "
                        "original mp3 mix instead.")
    p.add_argument('--csv_out', required=True, help='Output CSV path.')
    args = p.parse_args()

    with open(args.config) as f:
        configs = yaml.safe_load(f)
    data_config = configs['data']
    train_config = dict(configs['train'])
    model_config = configs['model']
    if args.batch_size is not None:
        train_config['batch_size'] = args.batch_size
    if args.num_workers is not None:
        train_config['num_workers'] = args.num_workers

    print(f"Split: {args.split}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Fusion: {args.fusion}")
    effective_mix_source = args.mix_source or data_config.get('mix_source', 'stems_sum')
    print(f"Mix source: {effective_mix_source}")

    loader = get_cached_dataloader(args.split, data_config, train_config, shuffle=False,
                                   mix_source=args.mix_source)

    print("\nLoading model...")
    model = build_model(args, model_config, train_config)

    trainer = L.Trainer(
        devices=1,
        accelerator='auto',
        precision=train_config.get('precision', '16-mixed'),
        logger=False,
    )

    outputs = trainer.predict(model, loader)
    rows = collect_predictions(outputs or [])

    write_csv(rows, args.csv_out)
    print(f"\nWrote {len(rows)} predictions to {args.csv_out}")
    print_metrics(rows)


if __name__ == '__main__':
    main()

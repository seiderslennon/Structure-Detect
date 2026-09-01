"""
Score a checkpoint against the SONICS ssep_EvalSet using precomputed features
(produced by precompute_features_evalset.py). Like predict_cached.py, but for
the flat eval-set layout: per-modality features come from
``<cache_dir>/<song_id>.pt`` and MERT runs on the original mp3 mix at
``<mix_root>/<song_id>.mp3``.

Output CSV schema mirrors predict_cached.py:
    song_id, label, prediction, prediction_label, confidence, real_prob, fake_prob

Labels (real/fake) are pulled from a sidecar CSV (default
``/data/SONICS/evalset_complete.csv`` with ``filepath,target`` columns,
target=1 -> real, target=0 -> fake). When labels are present we also print
acc / F1 / AUC / EER, matching predict_cached.py.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m ai_music.predict_evalset \
        --checkpoint /home/lennon/AI_music/lightning_logs/version_306/checkpoints/epoch=0-step=2520.ckpt \
        --config     /home/lennon/AI_music/ai_music/configs/SpecTTTra.yaml \
        --csv_out    ai_music/inference_results/v306_ep0_ssep_evalset.csv
"""

import argparse
import csv
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml

from ai_music.data import cross_attention
from ai_music.data.evalset_dataset import get_evalset_dataloader
from ai_music.models import resnet
from ai_music.models.sonics import SpecTTTraAttentionClassifier
from ai_music.train import LightningModel


def build_model(args, model_config, train_config):
    classifier_type = model_config.get("classifier_type", "ResNet").lower()
    if classifier_type == "resnet":
        classifier = resnet.ResNet(max_tokens_per_modality=model_config["max_tokens_per_modality"])
    elif classifier_type == "spectttra":
        classifier = SpecTTTraAttentionClassifier(
            feature_dim=model_config.get("feature_dim"),
            embed_dim=model_config.get("embed_dim"),
            num_heads=model_config.get("num_heads"),
            num_layers=model_config.get("num_layers"),
            tokenizer_clip_size=model_config.get("tokenizer_clip_size"),
            num_classes=2,
            pre_norm=model_config.get("pre_norm"),
            pe_learnable=model_config.get("pe_learnable"),
            pos_drop_rate=model_config.get("pos_drop_rate"),
            attn_drop_rate=model_config.get("attn_drop_rate"),
            proj_drop_rate=model_config.get("proj_drop_rate"),
            mlp_ratio=model_config.get("mlp_ratio"),
        )
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    if args.fusion == "cross_attention":
        fuser = cross_attention.MultiModalMERTFusion(use_layer_mix=True)
    elif args.fusion == "concat":
        fuser = cross_attention.ConcatLinearFusion()
    elif args.fusion == "mert_only":
        fuser = cross_attention.MERTOnlyFusion()
    else:
        raise ValueError(f"Unknown fusion: {args.fusion}")

    model = LightningModel.load_from_checkpoint(
        args.checkpoint,
        classifier=classifier,
        fuser=fuser,
        configs=train_config,
        map_location="cpu",
    )
    model.eval()
    return model


def collect_predictions(outputs):
    rows = []
    for batch_out in outputs:
        if batch_out is None:
            continue
        probs = batch_out["probs"].detach().cpu()
        preds = batch_out["predictions"].detach().cpu()
        song_ids = batch_out.get("song_ids") or [None] * len(preds)
        labels = batch_out.get("labels") or [None] * len(preds)

        for sid, lab, pred, prob in zip(song_ids, labels, preds, probs):
            pred_int = int(pred.item())
            real_prob = float(prob[1].item())
            fake_prob = float(prob[0].item())
            rows.append({
                "song_id": sid,
                "label": lab,
                "prediction": pred_int,
                "prediction_label": "real" if pred_int == 1 else "fake",
                "confidence": max(real_prob, fake_prob),
                "real_prob": real_prob,
                "fake_prob": fake_prob,
            })
    return rows


def write_csv(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "song_id", "label", "prediction", "prediction_label",
        "confidence", "real_prob", "fake_prob",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def print_metrics(rows):
    labelled = [r for r in rows if r["label"] in ("real", "fake")]
    if not labelled:
        print("No ground-truth labels found; skipping metrics.")
        return

    y_true_fake = np.array([1 if r["label"] == "fake" else 0 for r in labelled])
    y_pred_fake = np.array([1 if r["prediction_label"] == "fake" else 0 for r in labelled])
    fake_score = np.array([r["fake_prob"] for r in labelled])

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
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default="/home/lennon/AI_music/ai_music/configs/SpecTTTra.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fusion", default="cross_attention",
                   choices=["cross_attention", "concat", "mert_only"])
    p.add_argument("--cache_dir", default="/data/structture/cached_features/ssep_evalset")
    p.add_argument("--mix_root", default="/data/SONICS/EvalSet",
                   help="Flat dir of <song_id>.mp3 mixes consumed by MERT "
                        "when --mix_source=mp3.")
    p.add_argument("--mix_source", default="stems_sum",
                   choices=["mp3", "stems_sum"],
                   help="MERT input source. Default 'stems_sum' uses "
                        "vocals.wav + accompaniment.wav from --ssep_root/<id>/ "
                        "(pre-b66ca5e pipeline, e.g. lightning_logs/ablate-beat, "
                        "fusion-concat, mert-only — also the current default). "
                        "'mp3' uses --mix_root/<id>.mp3 (post-b66ca5e pipeline).")
    p.add_argument("--ssep_root", default="/data/SONICS/ssep_EvalSet",
                   help="Per-song stems dir used when --mix_source=stems_sum.")
    p.add_argument("--labels_csv", default="/data/SONICS/evalset_complete.csv",
                   help="Optional CSV with 'filepath,target' columns "
                        "(target 1=real, 0=fake) for metrics.")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--csv_out", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        configs = yaml.safe_load(f)
    data_config = configs["data"]
    train_config = dict(configs["train"])
    model_config = configs["model"]

    print(f"Config:      {args.config}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Fusion:      {args.fusion}")
    print(f"Cache dir:   {args.cache_dir}")
    print(f"Mix source:  {args.mix_source}")
    if args.mix_source == "mp3":
        print(f"Mix root:    {args.mix_root}")
    else:
        print(f"Ssep root:   {args.ssep_root}")
    print(f"Labels:      {args.labels_csv}")

    loader = get_evalset_dataloader(
        data_config,
        train_config,
        cache_dir=args.cache_dir,
        mix_root=args.mix_root,
        labels_csv=args.labels_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mix_source=args.mix_source,
        ssep_root=args.ssep_root,
    )

    print("\nLoading model...")
    model = build_model(args, model_config, train_config)

    trainer = L.Trainer(
        devices=1,
        accelerator="auto",
        precision=train_config.get("precision", "16-mixed"),
        logger=False,
    )

    outputs = trainer.predict(model, loader)
    rows = collect_predictions(outputs or [])

    write_csv(rows, args.csv_out)
    print(f"\nWrote {len(rows)} predictions to {args.csv_out}")
    print_metrics(rows)


if __name__ == "__main__":
    main()

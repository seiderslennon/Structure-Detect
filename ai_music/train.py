"""
Training script with ablation support.

Usage:
    # Full model
    python -m ai_music.train --config ai_music/configs/SpecTTTra.yaml

    # Feature ablations (remove one branch)
    python -m ai_music.train --config ai_music/configs/SpecTTTra.yaml --ablate whisper
    python -m ai_music.train --config ai_music/configs/SpecTTTra.yaml --ablate crepe
    python -m ai_music.train --config ai_music/configs/SpecTTTra.yaml --ablate chord
    python -m ai_music.train --config ai_music/configs/SpecTTTra.yaml --ablate beat

    # Multiple branches at once
    python -m ai_music.train --config ai_music/configs/SpecTTTra.yaml --ablate whisper crepe

    # MERT only (remove all specialized branches)
    python -m ai_music.train --config ai_music/configs/SpecTTTra.yaml --ablate whisper crepe chord beat

    # Fusion ablation (concat + linear instead of cross attention)
    python -m ai_music.train --config ai_music/configs/SpecTTTra.yaml --fusion concat

    # Custom run name for logging
    python -m ai_music.train --config ai_music/configs/SpecTTTra.yaml --ablate whisper --run_name ablate_whisper
"""

import torch
import torch.nn.functional as F
import lightning as L
from ai_music.models import resnet
from ai_music.models.sonics import SpecTTTraAttentionClassifier
from ai_music.data import cross_attention
import yaml
from ai_music.utils import log_print
from ai_music.data.cached_dataset import get_cached_dataloader as get_dataloader

MODALITY_NAMES = ['whisper', 'crepe', 'chord', 'beat']
MODALITY_INDEX = {name: i for i, name in enumerate(MODALITY_NAMES)}


class LightningModel(L.LightningModule):
    def __init__(self, classifier, fuser, configs, ablate_modalities=None):
        super().__init__()
        self.classifier = classifier
        self.fuser = fuser
        self.configs = configs
        self.ablate_modalities = ablate_modalities or []
        self.val_step_counter = 0
        
        self.save_hyperparameters(ignore=['classifier', 'fuser'])

    def _apply_ablation(self, attention_features):
        """Zero out ablated modalities."""
        for name in self.ablate_modalities:
            idx = MODALITY_INDEX[name]
            attention_features[idx] = torch.zeros_like(attention_features[idx])
        return attention_features

    def training_step(self, batch):
        if batch is None:
            return None
        
        feats = [emb.squeeze(1) for emb in batch["emb"][:4]]
        mert_layers = batch["emb"][4]
        
        attention_features = self.fuser.forward(feats=feats, mert_layers=mert_layers)
        attention_features = self._apply_ablation(attention_features)
        
        label_strings = [l for l in batch["label"]]
        labels = [1 if l == "real" else 0 for l in label_strings]
        labels = torch.tensor(labels, dtype=torch.long, device=attention_features[0].device)

        z = self.classifier(attention_features)
        loss = F.cross_entropy(z, labels)
        
        preds = torch.argmax(z, dim=1)
        acc = (preds == labels).float().mean()
        
        values = {"train_loss": loss, "train_acc": acc}
        self.log_dict(values, prog_bar=True, batch_size=len(labels))
        
        return loss

    def validation_step(self, batch):
        if batch is None:
            return None

        feats = [emb.squeeze(1) for emb in batch["emb"][:4]]
        mert_layers = batch["emb"][4]
        
        attention_features = self.fuser.forward(feats=feats, mert_layers=mert_layers)
        attention_features = self._apply_ablation(attention_features)
        
        label_strings = [l for l in batch["label"]]
        labels = [1 if l == "real" else 0 for l in label_strings]
        labels = torch.tensor(labels, dtype=torch.long, device=attention_features[0].device)

        z = self.classifier(attention_features)
        loss = F.cross_entropy(z, labels)
        
        preds = torch.argmax(z, dim=1)
        acc = (preds == labels).float().mean()
        
        values = {"val_loss": loss, "val_acc": acc}
        self.log_dict(values, prog_bar=True, batch_size=len(labels))
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        if batch is None:
            return None
            
        feats = [emb.squeeze(1) for emb in batch["emb"][:4]]
        mert_layers = batch["emb"][4]
        
        attention_features = self.fuser.forward(feats=feats, mert_layers=mert_layers)
        attention_features = self._apply_ablation(attention_features)
        z = self.classifier(attention_features)
        
        probs = torch.softmax(z, dim=1)
        preds = torch.argmax(z, dim=1)
        
        return {
            'logits': z,
            'probs': probs,
            'predictions': preds,
            'labels': batch.get('label', None)
        }

    def configure_optimizers(self):
        lr = float(self.configs['learning_rate'])
        wd = float(self.configs['weight_decay'])
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        return optimizer
    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/lennon/AI_music/ai_music/configs/SpecTTTra.yaml')
    parser.add_argument('--ablate', nargs='*', default=[], choices=MODALITY_NAMES,
                        help='Modalities to zero out. E.g. --ablate whisper crepe')
    parser.add_argument('--fusion', type=str, default='cross_attention', choices=['cross_attention', 'concat'],
                        help='Fusion method. concat replaces cross attention with concat + linear.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Custom run name for logging. Auto-generated if not provided.')
    args = parser.parse_args()

    with open(args.config) as f:
        configs = yaml.safe_load(f)
        data_config = configs['data']
        train_config = configs['train']
        model_config = configs['model']

    # Generate run name
    if args.run_name:
        run_name = args.run_name
    elif args.ablate:
        if set(args.ablate) == set(MODALITY_NAMES):
            run_name = "mert_only"
        else:
            run_name = "ablate_" + "_".join(sorted(args.ablate))
    elif args.fusion == 'concat':
        run_name = "fusion_concat"
    else:
        run_name = "full_model"

    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    if args.ablate:
        print(f"Ablated modalities: {args.ablate}")
    print(f"Fusion: {args.fusion}")
    print(f"{'='*60}\n")

    train_loader = get_dataloader('train', data_config, train_config, shuffle=True)
    val_loader = get_dataloader('val', data_config, train_config, shuffle=False)
    
    csv_logger, progress_logger = log_print.print_dataset_statistics(train_loader, val_loader)

    trainer = L.Trainer(
        devices=1,
        accelerator="auto",
        max_epochs=train_config['max_epochs'],
        precision=train_config['precision'],
        accumulate_grad_batches=train_config['accumulate_grad_batches'],
        logger=csv_logger,
        callbacks=[progress_logger],
        num_sanity_val_steps=5,
    )

    # Fusion module
    if args.fusion == 'cross_attention':
        fuser = cross_attention.MultiModalMERTFusion(use_layer_mix=True)
    elif args.fusion == 'concat':
        # If you have a ConcatFusion class, use it here.
        # Otherwise fall back to cross_attention (you'll need to implement ConcatFusion)
        try:
            from ai_music.data.cross_attention import ConcatLinearFusion
            fuser = ConcatLinearFusion()
        except ImportError:
            print("WARNING: ConcatLinearFusion not implemented. Using cross_attention as fallback.")
            print("TODO: implement ai_music.data.cross_attention.ConcatLinearFusion")
            fuser = cross_attention.MultiModalMERTFusion(use_layer_mix=True)
    
    # Classifier
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
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    model = LightningModel(classifier, fuser, train_config, ablate_modalities=args.ablate)
    print(f"Classifier: {classifier_type}")
    print(f"Ablated: {args.ablate if args.ablate else 'none'}")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
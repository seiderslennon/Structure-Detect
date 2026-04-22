import torch
import torch.nn.functional as F
import lightning as L
from ai_music.models import resnet
from ai_music.models.sonics import SpecTTTraAttentionClassifier
from ai_music.data import cross_attention
import yaml
from ai_music.utils import log_print
from ai_music.data.dataset import get_dataloader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class LightningModel(L.LightningModule):
    def __init__(self, classifier, fuser, configs):
        super().__init__()
        self.classifier = classifier
        self.fuser = fuser
        self.configs = configs
        self.val_step_counter = 0
        
        # Save hyperparameters for checkpoint loading
        # Note: classifier and fuser are saved as part of the model state_dict
        # Only save configs as hyperparameters
        self.save_hyperparameters(ignore=['classifier', 'fuser'])

    def training_step(self, batch):
        if batch is None:
            return None
        
        feats = [emb.squeeze(1) for emb in batch["emb"][:4]]  # feature embeddings
        mert_layers = batch["emb"][4]  # MERT layers: [B, 13, T, 768]
        
        attention_features = self.fuser.forward(feats=feats, mert_layers=mert_layers)

        
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

        feats = [emb.squeeze(1) for emb in batch["emb"][:4]]  # feature embeddings
        mert_layers = batch["emb"][4]  # MERT layers: [B, 13, T, 768]
        
        attention_features = self.fuser.forward(feats=feats, mert_layers=mert_layers)
        
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
            
        feats = [emb.squeeze(1) for emb in batch["emb"][:4]]  # feature embeddings
        mert_layers = batch["emb"][4]  # MERT layers: [B, 13, T, 768]
        
        attention_features = self.fuser.forward(feats=feats, mert_layers=mert_layers)
        z = self.classifier(attention_features)
        
        # Get probabilities and predictions
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
    parser.add_argument('--config', type=str, default=str(PROJECT_ROOT / 'ai_music/configs/SpecTTTra.yaml'))
    args = parser.parse_args()

    with open(args.config) as f:
        configs = yaml.safe_load(f)
        data_config = configs['data']
        train_config = configs['train']
        model_config = configs['model']

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
        # limit_train_batches=0.1, limit_val_batches=0.1
    )

    fuser = cross_attention.MultiModalMERTFusion(use_layer_mix=True)
    
    # Choose classifier based on config
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

    model = LightningModel(classifier, fuser, train_config)
    print(model.classifier)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

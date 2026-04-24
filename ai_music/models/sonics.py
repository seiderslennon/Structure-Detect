import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_music.layers import Transformer
from ai_music.layers.tokenizer import FeatureTokenizer


class SpecTTTraAttentionClassifier(nn.Module):
    """
    SpecTTTra-based classifier for attention features after cross attention.
    Processes attention features [B, T, D] by:
    1. Tokenizing each modality's features using FeatureTokenizer
    2. Processing through transformer
    3. Pooling and classification
    """
    
    def __init__(
        self,
        feature_dim=512,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        tokenizer_clip_size=5,
        num_classes=2,
        pre_norm=False,
        pe_learnable=False,
        pos_drop_rate=0.0,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        mlp_ratio=4.0,
    ):
        super(SpecTTTraAttentionClassifier, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        self.tokenizer = FeatureTokenizer(
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            clip_size=tokenizer_clip_size,
            pre_norm=pre_norm,
            pe_learnable=pe_learnable,
        )
        
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        
        self.transformer = Transformer(
            embed_dim,
            num_heads,
            num_layers,
            attn_drop=attn_drop_rate,
            proj_drop=proj_drop_rate,
            mlp_ratio=mlp_ratio,
        )
        
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, attention_features):
        """
        Args:
            attention_features: list of 4 tensors, each [B, T, D] where T varies
        
        Returns:
            logits: [B, num_classes]
        """
        modality_outputs = []
        for feat in attention_features:
            # Tokenize: [B, T, D] -> [B, T', embed_dim]
            tokens = self.tokenizer(feat)
            tokens = self.pos_drop(tokens)
            
            # Transformer: [B, T', embed_dim] -> [B, T', embed_dim]
            output = self.transformer(tokens)
            
            # Pool: [B, T', embed_dim] -> [B, embed_dim]
            pooled = output.mean(dim=1)
            modality_outputs.append(pooled)
        
        # Combine modalities: average or concatenate
        # Using average for simplicity - can be changed to concatenate if needed
        combined = torch.stack(modality_outputs, dim=1).mean(dim=1)  # [B, embed_dim]
        
        # Classify
        logits = self.classifier(combined)  # [B, num_classes]
        return logits


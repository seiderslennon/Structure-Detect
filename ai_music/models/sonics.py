from ai_music.layers.feature import FeatureExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_music.layers import Transformer
from ai_music.layers.tokenizer import STTokenizer, FeatureTokenizer

class SpecTTTra(nn.Module):
    def __init__(
        self,
        input_spec_dim,
        input_temp_dim,
        embed_dim,
        t_clip,
        f_clip,
        num_heads,
        num_layers,
        pre_norm=False,
        pe_learnable=False,
        pos_drop_rate=0.0,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        mlp_ratio=4.0,
    ):
        super(SpecTTTra, self).__init__()
        self.input_spec_dim = input_spec_dim
        self.input_temp_dim = input_temp_dim
        self.embed_dim = embed_dim
        self.t_clip = t_clip
        self.f_clip = f_clip
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pre_norm = (
            pre_norm  # applied after tokenization before transformer (used in CLIP)
        )
        self.pe_learnable = pe_learnable  # learned positional encoding
        self.pos_drop_rate = pos_drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.proj_drop_rate = proj_drop_rate
        self.mlp_ratio = mlp_ratio

        self.st_tokenizer = STTokenizer(
            input_spec_dim,
            input_temp_dim,
            t_clip,
            f_clip,
            embed_dim,
            pre_norm=pre_norm,
            pe_learnable=pe_learnable,
        )
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.transformer = Transformer(
            embed_dim,
            num_heads,
            num_layers,
            attn_drop=self.attn_drop_rate,
            proj_drop=self.proj_drop_rate,
            mlp_ratio=self.mlp_ratio,
        )

    def forward(self, x):
        # Squeeze the channel dimension if it exists
        if x.dim() == 4:
            x = x.squeeze(1)

        spectro_temporal_tokens = self.st_tokenizer(x)
        spectro_temporal_tokens = self.pos_drop(spectro_temporal_tokens)

        output = self.transformer(spectro_temporal_tokens)  # shape: (B, T/t + F/f, dim)
        return output


# Example usage:
# input_spec_dim = 384
# input_temp_dim = 128
# embed_dim = 512
# t_clip = 20  # This means t
# f_clip = 10  # This means f
# num_heads = 8
# num_layers = 6
# dim_feedforward = 512
# num_classes = 10

class AudioClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.model_name = cfg.model.name
        self.input_shape = cfg.model.input_shape
        self.num_classes = cfg.num_classes
        self.embed_dim = cfg.model.embed_dim
        self.ft_extractor = FeatureExtractor(cfg)
        self.encoder = self.get_encoder(cfg)
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        self.use_init_weights = getattr(cfg.model, "use_init_weights", True)

        if self.use_init_weights:
            self.initialize_weights()

    def get_encoder(self, cfg):
        if cfg.model.name == "SpecTTTra":
            model = SpecTTTra(
                input_spec_dim=cfg.model.input_shape[0],
                input_temp_dim=cfg.model.input_shape[1],
                embed_dim=cfg.model.embed_dim,
                t_clip=cfg.model.t_clip,
                f_clip=cfg.model.f_clip,
                num_heads=cfg.model.num_heads,
                num_layers=cfg.model.num_layers,
                pre_norm=cfg.model.pre_norm,
                pe_learnable=cfg.model.pe_learnable,
                pos_drop_rate=getattr(cfg.model, "pos_drop_rate", 0.0),
                attn_drop_rate=getattr(cfg.model, "attn_drop_rate", 0.0),
                proj_drop_rate=getattr(cfg.model, "proj_drop_rate", 0.0),
                mlp_ratio=getattr(cfg.model, "mlp_ratio", 4.0),
            )
        else:
            raise ValueError(f"Model {cfg.model.name} not supported in V1.")
        return model

    def forward(self, audio, y=None):
        spec = self.ft_extractor(audio)  # shape: (batch_size, n_mels, n_frames)
        spec = spec.unsqueeze(1)  # shape: (batch_size, 1, n_mels, n_frames)

        spec = F.interpolate(spec, size=tuple(self.input_shape), mode="bilinear")
        features = self.encoder(spec)
        embeds = features.mean(dim=1)
        preds = self.classifier(embeds)
        return preds if y is None else (preds, y)

    def initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name.startswith("classifier"):
                    nn.init.zeros_(module.weight)
                    nn.init.constant_(module.bias, 0.0)
                else:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.normal_(module.bias, std=1e-6)
            elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif hasattr(module, "init_weights"):
                module.init_weights()


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


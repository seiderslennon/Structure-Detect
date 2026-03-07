import math
import torch
import torch.nn as nn
from ai_music.layers.embedding import (
    SinusoidPositionalEncoding,
    LearnedPositionalEncoding,
)


class STTokenizer(nn.Module):
    def __init__(
        self,
        input_spec_dim,
        input_temp_dim,
        t_clip,
        f_clip,
        embed_dim,
        pre_norm=False,
        pe_learnable=False,
    ):
        super(STTokenizer, self).__init__()
        self.input_spec_dim = input_spec_dim
        self.input_temp_dim = input_temp_dim
        self.t_clip = t_clip
        self.f_clip = f_clip
        self.embed_dim = embed_dim
        self.pre_norm = pre_norm
        self.pe_learnable = pe_learnable

        self.num_temporal_tokens = math.floor(
            (input_temp_dim - t_clip) / t_clip + 1
        )  # floor((1280 - 5) / 5 + 1)= 256
        self.num_spectral_tokens = math.floor(
            (input_spec_dim - f_clip) / f_clip + 1
        )  # floor((128 - 3) / 3 + 1) = 42
        # L_out = floor((L_in + 2*p - d*(k - 1) - 1) / s + 1) (ref: PyTorch docs)
        self.num_tokens = (
            self.num_temporal_tokens + self.num_spectral_tokens
        )  # 255 + 42 = 299
        # For ViT, num_tokens = (1280 * 128)//(5 * 3) = 10922 :)

        self.temporal_tokenizer = Tokenizer1D(
            input_spec_dim,
            embed_dim,
            clip_size=t_clip,
            num_clips=self.num_temporal_tokens,
            pre_norm=pre_norm,
            pe_learnable=pe_learnable,
        )
        self.spectral_tokenizer = Tokenizer1D(
            input_temp_dim,
            embed_dim,
            clip_size=f_clip,
            num_clips=self.num_spectral_tokens,
            pre_norm=pre_norm,
            pe_learnable=pe_learnable,
        )

    def forward(self, x):
        # Temporal tokenization
        temporal_input = x  # shape: (B, F, T)
        temporal_tokens = self.temporal_tokenizer(
            temporal_input
        )  # shape: (B, T/t, dim)

        # Spectral tokenization
        spectral_input = x.permute(0, 2, 1)  # shape: (batch_size, T, F)
        spectral_tokens = self.spectral_tokenizer(
            spectral_input
        )  # shape: (B, F/f, dim)

        spectro_temporal_tokens = torch.cat(
            (temporal_tokens, spectral_tokens), dim=1
        )  # shape: (B, T/t + F/f, dim)
        return spectro_temporal_tokens


class Tokenizer1D(nn.Module):
    """Temporal/Spectral Tokenizer

    Whisper uses temporal tokenizer but time_clip_size is too small, stride=1,  thus
    complexity is very high. We use stride=clip_size - 1 to reduce complexity.
    """

    def __init__(
        self,
        input_dim,
        token_dim,
        clip_size,
        num_clips,
        pre_norm=False,
        pe_learnable=False,
    ):
        super(Tokenizer1D, self).__init__()
        self.conv1d = nn.Conv1d(
            input_dim,
            token_dim,
            clip_size,
            stride=clip_size,
            bias=not pre_norm,  #  # disable bias if pre-norm is used (e.g. CLIP)
        )
        self.act = nn.GELU()
        self.pos_encoder = (
            SinusoidPositionalEncoding(token_dim)
            if not pe_learnable
            else LearnedPositionalEncoding(token_dim, num_clips)
        )
        self.norm_pre = nn.LayerNorm(token_dim, eps=1e-6) if pre_norm else nn.Identity()

    def forward(self, x):
        x = x  # (F, T)
        x = self.conv1d(x)  # (F, T) -> (dim, T/t)
        x = self.act(x)
        x = x.transpose(1, 2)  # (dim, T/t) -> (T/t, dim)
        x = self.pos_encoder(x)  # add position embeds
        x = self.norm_pre(x)
        return x


class FeatureTokenizer(nn.Module):
    """Simple tokenizer for attention features after cross attention.
    
    Tokenizes features of shape [B, T, D] by applying temporal tokenization.
    This reduces the sequence length while preserving feature information.
    """
    
    def __init__(
        self,
        feature_dim=512,
        embed_dim=512,
        clip_size=5,
        pre_norm=False,
        pe_learnable=False,
    ):
        super(FeatureTokenizer, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.clip_size = clip_size
        
        # Calculate approximate number of tokens for positional encoding
        # Use a reasonable default since T varies
        max_tokens = 6000 // clip_size + 1  # Approximate max tokens
        
        self.tokenizer = Tokenizer1D(
            input_dim=feature_dim,
            token_dim=embed_dim,
            clip_size=clip_size,
            num_clips=max_tokens,  # Used for learned PE initialization
            pre_norm=pre_norm,
            pe_learnable=pe_learnable,
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T, D] tensor where B=batch, T=variable sequence length, D=feature_dim
            
        Returns:
            tokens: [B, T', embed_dim] where T' = floor((T - clip_size) / clip_size + 1)
        """
        # Transpose: [B, T, D] -> [B, D, T] for conv1d
        x = x.transpose(1, 2)  # [B, D, T]
        
        # Tokenize along temporal dimension
        tokens = self.tokenizer(x)  # [B, T', embed_dim]
        
        return tokens

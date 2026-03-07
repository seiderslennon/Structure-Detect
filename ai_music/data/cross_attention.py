import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import math

# # --- utilities ---
class SinusoidalPE(nn.Module):
    def __init__(self, dim, max_len=120000):  # plenty of headroom
        super().__init__()
        pe = torch.zeros(max_len, dim)  # [L, D]
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe, persistent=False)
    def forward(self, x, offset=0):
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[offset:offset+T].unsqueeze(0).to(x.dtype)

class ScalarMix(nn.Module):
    """Learned softmax mix over L layers. inputs: list/stack of [B,T,D]."""
    def __init__(self, num_layers):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(num_layers))
    def forward(self, layers):  # layers: [L, B, T, D] or list of [B,T,D] or [B, L, T, D]
        if isinstance(layers, list):
            x = torch.stack(layers, dim=0)  # [L,B,T,D]
        else:
            x = layers
            # If tensor is [B, L, T, D], transpose to [L, B, T, D]
            if x.dim() == 4 and x.size(1) == self.w.size(0):
                x = x.transpose(0, 1)  # [B, L, T, D] -> [L, B, T, D]
        weights = torch.softmax(self.w, dim=0)  # [L]
        return (weights[:, None, None, None] * x).sum(dim=0)  # [B,T,D]


class MERTCrossAttention(nn.Module):
    def __init__(self, q_in_dim, mert_dim=768, out_dim=512, n_heads=8, ff_mult=4,
                 use_layer_mix=True, mert_layers=13, batch_first=True,
                 mert_downsample_stride=None):
        super().__init__()

        self.out_dim = out_dim
        self.input_norm = nn.LayerNorm(q_in_dim)  # Normalize input features
        self.q_proj = nn.Linear(q_in_dim, out_dim)
        self.use_layer_mix = use_layer_mix
        self.mert_downsample_stride = mert_downsample_stride

        if use_layer_mix:
            self.mixer = ScalarMix(mert_layers)

        self.q_pe = SinusoidalPE(out_dim)
        self.kv_pe = SinusoidalPE(mert_dim)

        self.attn = nn.MultiheadAttention(embed_dim=out_dim, num_heads=n_heads, 
                                          kdim=mert_dim, vdim=mert_dim, batch_first=batch_first)

        self.norm_q = nn.LayerNorm(out_dim)
        self.norm_out = nn.LayerNorm(out_dim)

        hid = ff_mult * out_dim
        self.ff = nn.Sequential(
            nn.Linear(out_dim, hid),
            nn.GELU(),
            nn.Linear(hid, out_dim),
        )

    @staticmethod
    def _to_device(x, device):
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return [t.to(device) for t in x]
        return x.to(device)

    def forward(self, q, mert_layers, q_mask=None, mert_mask=None):
        """
        q:            [B, Tq, Dq]
        mert_layers:  list of 13 tensors [B, Tm, 768] OR stacked [13, B, Tm, 768] OR already-mixed [B, Tm, 768]
        q_mask:       [B, Tq] (True=valid)  (optional)
        mert_mask:    [B, Tm] (True=valid)  (optional)
        returns:      [B, Tq, out_dim]
        """
        device = q.device

        # (1) Ensure the module itself lives on the query's device (simple + robust).
        #     This moves all parameters/buffers (q_proj, attn, PEs, etc.) just once per device switch.
        if next(self.parameters()).device != device:
            self.to(device)

        # (2) Move inputs/masks to that same device.
        mert_layers = self._to_device(mert_layers, device)
        q_mask = self._to_device(q_mask, device)
        mert_mask = self._to_device(mert_mask, device)

        B, Tq, _ = q.shape

        # (2.5) Normalize input query features
        q = self.input_norm(q)

        # (3) Mix MERT layers -> [B, Tm, 768]
        if self.use_layer_mix:
            kv = self.mixer(mert_layers)  # handle list or stacked internally
        else:
            kv = mert_layers  # already chosen layer or pre-mixed tensor

        # (4) Optional downsample MERT tokens
        if self.mert_downsample_stride and self.mert_downsample_stride > 1:
            s = self.mert_downsample_stride
            Tm = kv.size(1)
            cut = (Tm // s) * s
            kv = kv[:, :cut].view(B, cut // s, s, kv.size(2)).mean(dim=2)
            if mert_mask is not None:
                mert_mask = mert_mask[:, :cut].view(B, cut // s, s).all(dim=2)

        # (5) Projections + PEs
        q_proj = self.q_proj(q)         # [B, Tq, out_dim]
        q_proj = self.q_pe(q_proj)
        kv_pe = self.kv_pe(kv)

        # (6) Build key_padding_mask (True = pad) per PyTorch convention
        def to_kpm(mask):
            if mask is None:
                return None
            return ~mask  # invert True(valid) -> False(valid); True(pad) expected

        q_kpm = to_kpm(q_mask)
        kv_kpm = to_kpm(mert_mask)

        # (7) Cross-attention
        attn_out, _ = self.attn(query=self.norm_q(q_proj),
                                key=kv_pe,
                                value=kv_pe,
                                key_padding_mask=kv_kpm,
                                need_weights=False)

        # (8) Residual + FFN
        x = self.norm_out(q_proj + attn_out)
        x = x + self.ff(x)
        return x


class MultiModalMERTFusion(nn.Module):
    def __init__(self,
                 out_dim=512,
                 heads=8,
                 mert_dim=768,
                 use_layer_mix=False,
                 mert_layers=13,
                 mert_downsample_stride_by_mod=None):
        """
        mert_downsample_stride_by_mod: dict modality -> stride (e.g., {'crepe': 2, 'chord': 2})
        """
        super().__init__()
        self.out_dim = out_dim
        ds = mert_downsample_stride_by_mod or {}

        # per-modality cross-attenders (Q from modality, K/V from MERT)
        self.whisper = MERTCrossAttention(384, mert_dim, out_dim, heads, use_layer_mix=use_layer_mix,
                                          mert_layers=mert_layers, mert_downsample_stride=ds.get('whisper'))
        self.crepe   = MERTCrossAttention(256, mert_dim, out_dim, heads, use_layer_mix=use_layer_mix,
                                          mert_layers=mert_layers, mert_downsample_stride=ds.get('crepe', 2))
        self.chord   = MERTCrossAttention(240, mert_dim, out_dim, heads, use_layer_mix=use_layer_mix,
                                          mert_layers=mert_layers, mert_downsample_stride=ds.get('chord', 2))
        self.beat    = MERTCrossAttention(512, mert_dim, out_dim, heads, use_layer_mix=use_layer_mix,
                                          mert_layers=mert_layers, mert_downsample_stride=ds.get('beat'))


    def forward(self, feats, mert_layers, masks=None, mert_mask=None):
        """
        mert_layers: list or stacked tensor of 13 layers -> each [B, Tm, 768]
        masks: optional dict modality -> [B, T] boolean (True=valid)
        mert_mask: optional [B, Tm] boolean (True=valid)
        pooled: if True, returns pooled vectors per modality
        """
        masks = masks or {}
        out = [
            self.whisper(feats[0], mert_layers, q_mask=masks.get('whisper'), mert_mask=mert_mask),
            self.crepe(  feats[1],   mert_layers, q_mask=masks.get('crepe'),   mert_mask=mert_mask),
            self.chord(  feats[2],   mert_layers, q_mask=masks.get('chord'),   mert_mask=mert_mask),
            self.beat(   feats[3],    mert_layers, q_mask=masks.get('beat'),    mert_mask=mert_mask)
        ]

        return out
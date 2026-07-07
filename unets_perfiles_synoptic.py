"""
1D U-Net with synoptic conditioning for vertical meteorological profiles.
"""

import torch
import torch.nn as nn
import math
import numpy as np


class GroupNorm32(nn.GroupNorm):
    """GroupNorm cast to float32 for numerical stability."""
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    return GroupNorm32(32, channels)


def count_flops_attn(model, _x, y):
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttention(nn.Module):
    """QKV attention with split-heads."""
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Args:
            qkv: [N x (3 * H * C) x T]
        Returns:
            [N x (H * C) x T]
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class AttentionBlock(nn.Module):
    """Self-attention block for 1D sequences."""
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        dropout=0.1,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x_flat))
        h = self.attention(qkv)
        h = self.proj_out(h)
        h = self.dropout(h)
        return (x_flat + h).reshape(b, c, *spatial)


class TimeEmbedding(nn.Module):
    """Sinusoidal time-step embeddings."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class SynopticEmbedding(nn.Module):
    """
    MLP that maps synoptic variables to an embedding of the same dimension
    as the time embedding, enabling additive conditioning.
    """
    def __init__(self, num_synoptic_vars, emb_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_synoptic_vars, emb_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
        )

    def forward(self, synoptic_conditions):
        """
        Args:
            synoptic_conditions: (B, num_vars) normalised synoptic variables
        Returns:
            (B, emb_dim)
        """
        return self.net(synoptic_conditions)


class ResBlock1D(nn.Module):
    """1D residual block with time-embedding injection."""
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.shortcut = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(torch.relu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None]
        h = self.conv2(self.dropout(torch.relu(self.norm2(h))))
        return h + self.shortcut(x)


class UNet1D_Synoptic(nn.Module):
    """
    1D U-Net with synoptic conditioning for 311-level vertical profiles.

    Architecture:
        - Time embedding  : sinusoidal encoding -> 256-dim projection
        - Synoptic embedding: 23-var MLP -> 256-dim, added to time embedding
        - Encoder         : 64 -> 128 -> 256 channels (stride-2 downsampling)
        - Bottleneck      : 256 channels with optional self-attention
        - Decoder         : 128 -> 64 -> 64 channels (transposed convolutions)
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        dropout=0.1,
        num_classes=None,
        image_size=311,
        num_synoptic_vars=23,
        use_synoptic_cond=True,
        use_attention=True,
        num_heads=4,
    ):
        super().__init__()

        self.use_synoptic_cond = use_synoptic_cond
        self.use_attention = use_attention

        time_emb_dim = base_channels * 4  # 256 by default
        self.time_mlp = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        if self.use_synoptic_cond:
            self.synoptic_mlp = SynopticEmbedding(num_synoptic_vars, time_emb_dim, dropout)

        self.conv_in = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder
        self.enc1 = ResBlock1D(base_channels,   base_channels,   time_emb_dim, dropout)
        self.down1 = nn.Conv1d(base_channels,   base_channels,   kernel_size=3, stride=2, padding=1)
        self.enc2 = ResBlock1D(base_channels,   base_channels*2, time_emb_dim, dropout)
        self.down2 = nn.Conv1d(base_channels*2, base_channels*2, kernel_size=3, stride=2, padding=1)
        self.enc3 = ResBlock1D(base_channels*2, base_channels*4, time_emb_dim, dropout)
        self.down3 = nn.Conv1d(base_channels*4, base_channels*4, kernel_size=3, stride=2, padding=1)

        if self.use_attention:
            self.attn_enc3 = AttentionBlock(base_channels*4, num_heads=num_heads)

        # Bottleneck
        bottleneck_layers = [
            ResBlock1D(base_channels*4, base_channels*4, time_emb_dim, dropout),
        ]
        if self.use_attention:
            bottleneck_layers.append(AttentionBlock(base_channels*4, num_heads=num_heads))
        bottleneck_layers.append(
            ResBlock1D(base_channels*4, base_channels*4, time_emb_dim, dropout)
        )
        self.bottleneck = nn.ModuleList(bottleneck_layers)

        # Decoder
        self.up3   = nn.ConvTranspose1d(base_channels*4, base_channels*4, kernel_size=4, stride=2, padding=1)
        self.dec3  = ResBlock1D(base_channels*8, base_channels*2, time_emb_dim, dropout)
        if self.use_attention:
            self.attn_dec3 = AttentionBlock(base_channels*2, num_heads=num_heads)
        self.up2   = nn.ConvTranspose1d(base_channels*2, base_channels*2, kernel_size=4, stride=2, padding=1)
        self.dec2  = ResBlock1D(base_channels*4, base_channels,   time_emb_dim, dropout)
        self.up1   = nn.ConvTranspose1d(base_channels,   base_channels,   kernel_size=4, stride=2, padding=1)
        self.dec1  = ResBlock1D(base_channels*2, base_channels,   time_emb_dim, dropout)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(32, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, y=None, synoptic=None):
        """
        Args:
            x       : (B, 3, 311) noisy profiles
            t       : (B,) diffusion timesteps
            y       : unused (kept for API compatibility)
            synoptic: (B, 23) normalised synoptic conditions
        Returns:
            (B, 3, 311) predicted noise
        """
        t_emb = self.time_mlp(t)

        if self.use_synoptic_cond and synoptic is not None:
            t_emb = t_emb + self.synoptic_mlp(synoptic)

        h = self.conv_in(x)

        h1 = self.enc1(h, t_emb)
        h  = self.down1(h1)
        h2 = self.enc2(h, t_emb)
        h  = self.down2(h2)
        h3 = self.enc3(h, t_emb)
        h  = self.down3(h3)

        if self.use_attention:
            h3 = self.attn_enc3(h3)

        for block in self.bottleneck:
            if isinstance(block, AttentionBlock):
                h = block(h)
            else:
                h = block(h, t_emb)

        h = self.up3(h)
        if h.shape[2] != h3.shape[2]:
            h = h[:, :, :h3.shape[2]]
        h = torch.cat([h, h3], dim=1)
        h = self.dec3(h, t_emb)
        if self.use_attention:
            h = self.attn_dec3(h)

        h = self.up2(h)
        if h.shape[2] != h2.shape[2]:
            h = h[:, :, :h2.shape[2]]
        h = torch.cat([h, h2], dim=1)
        h = self.dec2(h, t_emb)

        h = self.up1(h)
        if h.shape[2] != h1.shape[2]:
            h = h[:, :, :h1.shape[2]]
        h = torch.cat([h, h1], dim=1)
        h = self.dec1(h, t_emb)

        return self.conv_out(h)


# Aliases for compatibility
UNet1D = UNet1D_Synoptic
unet_1d_perfiles = UNet1D_Synoptic
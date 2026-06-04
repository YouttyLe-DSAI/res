'''
    file:   c2ssm.py
    author: (your name)
    date:   2026/06/05

    C2-SSM: Cross-Camera Selective State Space Model
    ------------------------------------------------
    Drop-in replacement for VOXCrossViewLayer.
    Replaces SpatialCrossAttention (parallel, O(K^2))
    with C2SSMBlock sequential SSM scan (O(K)).
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Selective SSM Core (pure PyTorch, no extra packages needed on Kaggle)
# ─────────────────────────────────────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    """
    Mamba-style S6 selective scan, pure PyTorch.
    Input/Output shape:  (B, L, D)    L = camera sequence length
    """
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.dt_rank  = dt_rank or max(1, math.ceil(d_model / 16))

        self.x_proj   = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj  = nn.Linear(self.dt_rank, d_model, bias=True)

        A = torch.arange(1, d_state + 1).float().unsqueeze(0).repeat(d_model, 1)
        self.A_log    = nn.Parameter(torch.log(A))        # (d_model, d_state)
        self.D        = nn.Parameter(torch.ones(d_model)) # skip-connection scale
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.constant_(self.dt_proj.bias, -4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        N = self.d_state

        x_dbl         = self.x_proj(x)
        dt, B_s, C_s  = x_dbl.split([self.dt_rank, N, N], dim=-1)
        dt             = F.softplus(self.dt_proj(dt))              # (B,L,D)

        A  = -torch.exp(self.A_log.float())                        # (D,N)
        dA = torch.exp(dt.unsqueeze(-1) * A[None, None])           # (B,L,D,N)
        dB = dt.unsqueeze(-1) * B_s.unsqueeze(2)                   # (B,L,D,N)

        h  = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            h   = dA[:, i] * h + dB[:, i] * x[:, i].unsqueeze(-1)
            y_i = (h * C_s[:, i].unsqueeze(1)).sum(-1)             # (B,D)
            ys.append(y_i)

        y = torch.stack(ys, dim=1)                                  # (B,L,D)
        y = y + x * self.D[None, None]
        return self.out_proj(y)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  C2-SSM Block
# ─────────────────────────────────────────────────────────────────────────────

class C2SSMBlock(nn.Module):
    """
    Cross-Camera SSM aggregation per voxel position.

    vox_feat   (B, Q, D)     Q = Z*H*W
    cam_feats  (B, K, Q, D)  K = num cameras
    returns    (B, Q, D)
    """
    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.ssm     = SelectiveSSM(d_model, d_state=d_state)
        self.norm_in = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)
        self.gate    = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, vox_feat: torch.Tensor, cam_feats: torch.Tensor) -> torch.Tensor:
        B, Q, D = vox_feat.shape
        K       = cam_feats.shape[1]

        seq = torch.cat([vox_feat.unsqueeze(1), cam_feats], dim=1)  # (B,K+1,Q,D)
        seq = seq.permute(0, 2, 1, 3).reshape(B * Q, K + 1, D)
        seq = self.norm_in(seq)

        out     = self.ssm(seq)
        out     = self.drop(out)
        summary = out[:, -1, :].reshape(B, Q, D)

        g = self.gate(vox_feat)
        return vox_feat + g * summary


# ─────────────────────────────────────────────────────────────────────────────
# 3.  VOXSSMCrossViewLayer  —  drop-in for VOXCrossViewLayer
# ─────────────────────────────────────────────────────────────────────────────

try:
    from ..register import VOXEL_POOLING
    from ...bricks  import build_norm_layer, build_conv_layer
    _REGISTRY_AVAILABLE = True
except ImportError:
    _REGISTRY_AVAILABLE = False
    class _FakeReg:
        def register_module(self):
            def _d(cls): return cls
            return _d
    VOXEL_POOLING = _FakeReg()
    def build_conv_layer(cfg, **kw):
        return nn.Conv3d(kw['in_channels'], kw['out_channels'],
                         kw['kernel_size'], kw['stride'], kw['padding'], bias=False)
    def build_norm_layer(cfg, n):
        return None, nn.GroupNorm(cfg['num_groups'], n)


@VOXEL_POOLING.register_module()
class VOXSSMCrossViewLayer(nn.Module):
    """
    Drop-in replacement for VOXCrossViewLayer.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CONFIG — chỉ thay 1 dòng:

        transformerlayers=dict(
            type='VOXSSMCrossViewLayer',  # <── đổi từ VOXCrossViewLayer
            d_state=16,                   # optional
            feedforward_channels=...,
            ffn_dropout=0.1,
            operation_order=('cross_attn','norm','ffn','norm','conv')
        ),
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    def __init__(self,
        embed_dims,
        feedforward_channels,
        ffn_dropout       = 0.1,
        operation_order   = None,
        act_cfg           = dict(type='ReLU', inplace=True),
        norm_cfg          = dict(type='LN'),
        batch_first       = True,
        d_state           = 16,
        num_cams          = 3,
        attn_cfgs         = None,
        **kwargs
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams   = num_cams

        self.c2ssm    = C2SSMBlock(embed_dims, d_state=d_state, dropout=ffn_dropout)
        self.cam_proj = nn.Linear(embed_dims, embed_dims, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(ffn_dropout),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(ffn_dropout),
        )

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

        conv_layer = build_conv_layer(
            dict(type='Conv3d', bias=False),
            in_channels=embed_dims, out_channels=embed_dims,
            kernel_size=3, stride=1, padding=1
        )
        self.deblock = nn.Sequential(
            conv_layer,
            build_norm_layer(dict(type='GN', num_groups=16, requires_grad=True), embed_dims)[1],
            nn.ReLU(inplace=True)
        )

    def _build_cam_feats(self, key, bev_mask, query):
        K, HW, B, D = key.shape
        Q = query.shape[1]
        k          = key.permute(2, 0, 1, 3).contiguous()
        k          = self.cam_proj(k)
        cam_global = k.mean(dim=2)                           # (B,K,D)
        cam_feats  = cam_global.unsqueeze(2).expand(B, K, Q, D).clone()
        mask       = bev_mask.permute(1, 0, 2, 3).float()
        return cam_feats * mask

    def forward(self,
        query,
        key=None, value=None,
        bev_pos=None,
        vox_z=None, vox_h=None, vox_w=None,
        query_pos=None, key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        ref_3d=None, reference_points=None, mask=None,
        spatial_shapes=None, level_start_index=None,
        prev_bev=None,
        dynamic_cam_embed=None, dynamic_cam_embed_cfg=None,
        bev_mask=None,
        **kwargs
    ):
        B, Q, D = query.shape

        # 1. Cross-camera SSM
        if key is not None and bev_mask is not None:
            cam_feats = self._build_cam_feats(key, bev_mask, query)
        else:
            cam_feats = query.unsqueeze(1).expand(B, self.num_cams, Q, D)

        query = self.c2ssm(query, cam_feats)

        # 2. Norm
        query = self.norm1(query)

        # 3. FFN + residual
        query = self.ffn(query) + query

        # 4. Norm
        query = self.norm2(query)

        # 5. 3D Conv (giống hệt VOXCrossViewLayer)
        residual = query
        query = rearrange(query, 'b (z h w) c -> b c z h w', z=vox_z, h=vox_h, w=vox_w)
        query = self.deblock(query)
        query = rearrange(query, 'b c z h w -> b (z h w) c') + residual

        return query


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Smoke-test:   python c2ssm.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 52)
    print("  C2-SSM — Smoke Test")
    print("=" * 52)
    B, K, Z, H, W, D = 2, 3, 4, 8, 8, 32
    Q = Z * H * W

    layer    = VOXSSMCrossViewLayer(embed_dims=D, feedforward_channels=D*2,
                                    num_cams=K, d_state=8)
    query    = torch.randn(B, Q, D)
    key      = torch.randn(K, H*W, B, D)
    bev_mask = torch.ones(K, B, Q, 1).bool()

    out = layer(query, key=key, bev_mask=bev_mask, vox_z=Z, vox_h=H, vox_w=W)
    assert out.shape == query.shape
    n = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    print(f"  Input  : {tuple(query.shape)}")
    print(f"  Output : {tuple(out.shape)}")
    print(f"  Params : {n:,}")
    print("  PASSED ✓")
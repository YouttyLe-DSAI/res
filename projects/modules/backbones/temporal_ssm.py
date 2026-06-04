'''
    file:   temporal_ssm.py
    Temporal Amodal State Space Module (TA-SSM)
    Cắm vào sau image_feature_backbone trong VOXCounter
'''

import torch
import torch.nn as nn
from einops import rearrange

class TemporalAmodalSSM(nn.Module):
    """
    Thay thế Mamba thật bằng selective gating đơn giản
    để tránh cài mamba-ssm (hay bị lỗi trên Kaggle).
    Complexity: O(N) theo sequence length.
    """
    def __init__(self, d_model, num_levels=4):
        super().__init__()
        self.d_model = d_model
        # Occlusion gate: tự detect vùng bị khuất
        self.occ_gate = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_model // 4, 1),
                nn.ReLU(),
                nn.Conv2d(d_model // 4, 1, 1),
                nn.Sigmoid()
            ) for _ in range(num_levels)
        ])
        # Temporal fusion: trộn frame hiện tại với memory
        self.temporal_fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model * 2, d_model, 1),
                nn.GroupNorm(16, d_model),
                nn.ReLU()
            ) for _ in range(num_levels)
        ])
        self.hidden = None  # temporal memory

    def reset_hidden(self):
        """Gọi khi bắt đầu sequence mới"""
        self.hidden = None

    def forward(self, mlvl_feats):
        """
        mlvl_feats: list of [(b*n) c h w] — output của image_feature_fusion
        Returns: mlvl_feats đã được enhance với temporal context
        """
        enhanced = []
        new_hidden = []

        for lvl, (feat, gate, fuse) in enumerate(
            zip(mlvl_feats, self.occ_gate, self.temporal_fuse)
        ):
            bn, c, h, w = feat.shape

            if self.hidden is None or self.hidden[lvl].shape != feat.shape:
                # Frame đầu tiên: dùng chính feat làm memory
                hidden_lvl = feat.detach()
            else:
                hidden_lvl = self.hidden[lvl]

            # Occlusion mask: vùng nào bị khuất (feature bất thường)
            occ_mask = gate(feat)  # [bn, 1, h, w]

            # Vùng bị khuất → dùng temporal memory; vùng visible → dùng feat hiện tại
            x_filled = feat * (1 - occ_mask) + hidden_lvl * occ_mask

            # Fuse với temporal memory
            x_fused = fuse(torch.cat([x_filled, hidden_lvl], dim=1))

            enhanced.append(x_fused)
            new_hidden.append(x_fused.detach())  # update memory

        self.hidden = new_hidden
        return enhanced
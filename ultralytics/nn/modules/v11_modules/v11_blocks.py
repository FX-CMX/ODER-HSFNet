"""
v11 Ablation Experiment Modules: SDP + Attention Combinations

Variants:
    - DSC3k2_SDP_SE:       DSBottleneck + SparseDensePathway + SE (ChannelAttention)
    - DSC3k2_SDP_PSA:      DSBottleneck + SparseDensePathway + PolarizedSelfAttention
    - DSC3k2_DeformSDP_SE: DeformSDPBottleneck + SparseDensePathway + SE (ChannelAttention)
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import (
    C2f, DSBottleneck, SparseDensePathway,
    DeformableDSConv, PolarizedSelfAttention
)
from ultralytics.nn.modules.conv import DSConv, ChannelAttention


class DSC3k2_SDP_SE(C2f):
    """
    SDP + SE Channel Attention.
    
    Best single improvement (D_SDP, +0.4% mAP50) enhanced with lightweight
    SE channel attention for orthogonal spatial-routing + channel-selection.
    """
    def __init__(self, c1, c2, n=1, use_sparse_dense=True, e=0.5, g=1, shortcut=True, k1=3, k2=7, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        
        self.m = nn.ModuleList(
            DSBottleneck(self.c, self.c, shortcut=shortcut, e=1.0, k1=k1, k2=k2, d2=d2)
            for _ in range(n)
        )
        
        self.sparse_dense = SparseDensePathway(c2) if use_sparse_dense else nn.Identity()
        self.se = ChannelAttention(c2)
        
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        out = self.sparse_dense(out)
        return self.se(out)


class DSC3k2_SDP_PSA(C2f):
    """
    SDP + Polarized Self-Attention.
    
    Comparison variant: SDP spatial routing + PSA dual-branch (channel + spatial)
    attention for maximum feature refinement.
    """
    def __init__(self, c1, c2, n=1, use_sparse_dense=True, e=0.5, g=1, shortcut=True, k1=3, k2=7, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        
        self.m = nn.ModuleList(
            DSBottleneck(self.c, self.c, shortcut=shortcut, e=1.0, k1=k1, k2=k2, d2=d2)
            for _ in range(n)
        )
        
        self.sparse_dense = SparseDensePathway(c2) if use_sparse_dense else nn.Identity()
        self.psa = PolarizedSelfAttention(c2)
        
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        out = self.sparse_dense(out)
        return self.psa(out)


class DSC3k2_DeformSDP_SE(C2f):
    """
    Deform + SDP + SE Channel Attention.
    
    Full combination: DeformableDSConv bottleneck + SparseDensePathway +
    SE channel attention. Tests whether adding SE rescues the AD_DeformSDP
    combination that underperformed D_SDP alone.
    """
    def __init__(self, c1, c2, n=1, use_sparse_dense=True, e=0.5, g=1, shortcut=True, k1=3, k2=7, d2=1):
        super().__init__(c1, c2, n, shortcut, g, e)
        
        self.m = nn.ModuleList(
            _DeformBottleneck(self.c, self.c, shortcut=shortcut, e=1.0, k1=k1, k2=k2)
            for _ in range(n)
        )
        
        self.sparse_dense = SparseDensePathway(c2) if use_sparse_dense else nn.Identity()
        self.se = ChannelAttention(c2)
        
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        out = self.sparse_dense(out)
        return self.se(out)


class _DeformBottleneck(nn.Module):
    """Deform bottleneck: DeformableDSConv cv1 + DSConv cv2 (reused from DSC3k2_DeformSDP)."""
    def __init__(self, c1, c2, shortcut=True, e=0.5, k1=3, k2=5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DeformableDSConv(c1, c_, k=k1)
        self.cv2 = DSConv(c_, c2, k=k2)
        self.add = shortcut and c1 == c2
        
    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

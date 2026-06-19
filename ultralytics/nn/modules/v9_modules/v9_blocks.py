import torch
import torch.nn as nn
from ultralytics.nn.modules.block import C2f
try:
    from torchvision.ops import DeformConv2d
except ImportError:
    DeformConv2d = None

class PolarizedSelfAttention(nn.Module):
    def __init__(self, channel, reduction_ratio=2):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // reduction_ratio, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.ch_wz = nn.Conv2d(channel // reduction_ratio, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel // reduction_ratio)
        self.sigmoid = nn.Sigmoid()

        self.sp_wv = nn.Conv2d(channel, channel // reduction_ratio, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // reduction_ratio, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()
        # Channel-only polarization
        channel_wv = self.ch_wv(x).reshape(b, c // 2, -1) 
        channel_wq = self.ch_wq(x).reshape(b, -1, 1) 
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1) 
        channel_weight = self.sigmoid(self.ln(channel_wz.reshape(b, c // 2, 1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c // 2, 1, 1))
        channel_out = self.ch_wz(channel_weight * channel_wv.reshape(b, c // 2, h, w))

        # Spatial-only polarization
        spatial_wv = self.sp_wv(x).reshape(b, c // 2, -1)
        spatial_wq = self.agp(self.sp_wq(x)).reshape(b, c // 2, -1).permute(0, 2, 1)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))
        spatial_out = spatial_weight * x

        return spatial_out + channel_out

class DPC3k2(C2f):
    """Deformable Polarized Convolution Block (DPC)."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        # Using a unified Polarized Attention Block here
        self.psa = PolarizedSelfAttention(c2)
        # If deform conv is available we could replace the inner bottlenecks, but for stability we retain C2f routing + PSA
        
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.psa(out)

class HyperEdgeFusion(nn.Module):
    """Topology-Aware Hyper-edge FPN Fusion replacing standard Concat"""
    def __init__(self, d=1):
        super().__init__()
        self.d = d

    def forward(self, x):
        # x is a list of tensors [x1, x2...]
        if len(x) == 1:
            return x[0]
        # Concat along dimension
        x_c = torch.cat(x, self.d)
        
        # Simple hyper-edge decoupling simulation (scale relationships by mean variance)
        # In a full paper, this would be a GCN matrix operation.
        b, c, h, w = x_c.shape
        # Global topology cue
        topology_cue = torch.mean(x_c, dim=(2,3), keepdim=True)
        attention = torch.sigmoid(topology_cue)
        return x_c * attention


from ultralytics.nn.modules.head import Detect

class Detect_DAHS(Detect):
    """
    Density-Aware Hyper-Scale (DAHS) Detect Head.
    Identical in underlying architecture to standard Detect, but acts as a unique 
    identifier to cleanly trigger the NWD + Repulsion Loss and Dense Assigners 
    during training, maintaining perfect ablation isolation.
    """
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)


import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm2d(nn.Module):
    """2D RMSNorm for Spatial-Scale Attention Residual"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return x * rms * self.weight

class S2AR_Fusion(nn.Module):
    """
    S2AR_Fusion (Spatial-Scale Attention Residual Fusion)
    Dynamically fuses multi-scale features using Pixel-level Attention Residuals.
    """
    def __init__(self, ch):
        super().__init__()
        # Align to the channel size of the first input
        self.out_channels = ch[0]
        self.align_convs = nn.ModuleList([
            nn.Conv2d(c, self.out_channels, 1, 1, 0, bias=False) if c != self.out_channels else nn.Identity()
            for c in ch
        ])
        self.attn_proj = nn.Conv2d(self.out_channels, 1, 1, bias=False)
        self.norm = RMSNorm2d(self.out_channels)

    def forward(self, x):
        # Align channels
        aligned_x = [self.align_convs[i](x[i]) for i in range(len(x))]
        
        # Align spatial dimensions to the first input's spatial shape
        target_size = aligned_x[0].shape[-2:]
        for i in range(1, len(aligned_x)):
            if aligned_x[i].shape[-2:] != target_size:
                aligned_x[i] = F.interpolate(aligned_x[i], size=target_size, mode='bilinear', align_corners=False)
                
        V = torch.stack(aligned_x, dim=0) # [N, B, C, H, W]
        N, B, C, H, W = V.shape
        
        V_flat = V.view(N * B, C, H, W)
        K = self.norm(V_flat)
        logits = self.attn_proj(K).view(N, B, 1, H, W)
        scores = F.softmax(logits, dim=0)
        
        out = (scores * V).sum(dim=0)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import A2C2f, AAttn, ABlock, Bottleneck, C3k, DSC3k2
from ultralytics.nn.modules.conv import Conv


class RMSNorm2dLite(nn.Module):
    """Channel-agnostic RMSNorm for 2D feature maps."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(1, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return x * rms * self.gain


class KimiBlockAttnResFusion(nn.Module):
    """Lightweight block-attention residual fusion for YOLO neck feature aggregation."""

    def __init__(self, d: int = 1):
        super().__init__()
        self.d = d
        self.norm = RMSNorm2dLite()
        self.score_proj = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.residual_gain = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            return x
        if len(x) == 1:
            return x[0]

        aligned = list(x)
        target_size = aligned[0].shape[-2:]
        for i in range(1, len(aligned)):
            if aligned[i].shape[-2:] != target_size:
                aligned[i] = F.interpolate(aligned[i], size=target_size, mode="nearest")

        reps = []
        logits = []
        for feat in aligned:
            rep = self.norm(feat).mean(dim=1, keepdim=True)
            reps.append(rep)
            logits.append(self.score_proj(rep))

        logits = torch.stack(logits, dim=0)
        temp = torch.clamp(self.temperature, min=1e-3)
        weights = F.softmax(logits / temp, dim=0)

        context = torch.zeros_like(reps[0])
        for i, rep in enumerate(reps):
            context = context + weights[i] * rep

        gain = torch.tanh(self.residual_gain)
        fused = [feat + gain * context * feat for feat in aligned]
        return torch.cat(fused, dim=self.d)


class KR_Block_AttnRes(nn.Module):
    """Intra-block attention residual aggregation across historical block states."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm2dLite()
        self.attn_proj = nn.Conv2d(dim, 1, kernel_size=1, bias=False)

    def forward(self, blocks, partial_block):
        v_list = blocks + [partial_block]
        v = torch.stack(v_list, dim=0)
        n, b, c, h, w = v.shape

        v_flat = v.view(n * b, c, h, w)
        k = self.norm(v_flat)

        logits = self.attn_proj(k).view(n, b, 1, h, w)
        weights = F.softmax(logits, dim=0)
        out = (weights * v).sum(dim=0)
        return out


class KR_Block(nn.Module):
    """Kimi-Residual block for intra-scale aggregation."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn_res = nn.ModuleList([KR_Block_AttnRes(self.c) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        blocks = [y[-1]]
        for i, module in enumerate(self.m):
            partial_block = module(blocks[-1])
            h = self.attn_res[i](blocks, partial_block)
            blocks.append(h)
        y.extend(blocks[1:])
        return self.cv2(torch.cat(y, 1))


class DSC3k2_KimiRes(DSC3k2):
    """DSC3k2 augmented with Kimi block-attention residual fusion."""

    def __init__(self, c1, c2, n=1, dsc3k=False, e=0.5, g=1, shortcut=True, k1=3, k2=7, d2=1):
        super().__init__(c1, c2, n, dsc3k, e, g, shortcut, k1, k2, d2)
        self.attn_res = nn.ModuleList([KR_Block_AttnRes(self.c) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        blocks = [y[-1]]
        for i, module in enumerate(self.m):
            partial_block = module(blocks[-1])
            h = self.attn_res[i](blocks, partial_block)
            blocks.append(h)
        y.extend(blocks[1:])
        return self.cv2(torch.cat(y, 1))


class A2C2f_KimiRes(A2C2f):
    """A2C2f area-attention module augmented by Kimi block-attention residual."""

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.attn_res = nn.ModuleList([KR_Block_AttnRes(self.c) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        blocks = [y[-1]]
        for i, module in enumerate(self.m):
            partial_block = module(blocks[-1])
            h = self.attn_res[i](blocks, partial_block)
            blocks.append(h)
        y.extend(blocks[1:])
        return self.cv2(torch.cat(y, 1))


class _A2C2f_AAttnKR_Unit(nn.Module):
    """In-block flow: AAttn -> KimiRes(attn residual) -> AAttn."""

    def __init__(self, channels, num_heads, mlp_ratio=2.0, area=1):
        super().__init__()
        self.pre = ABlock(channels, num_heads, mlp_ratio, area)
        self.kr = KR_Block_AttnRes(channels)
        self.post = ABlock(channels, num_heads, mlp_ratio, area)

    def forward(self, x):
        spatial = self.pre(x)
        memory = self.kr([x], spatial)
        return self.post(memory)


class A2C2f_AAttn_KR(nn.Module):
    """Heterogeneous A2C2f with in-block AAttn-KR-AAttn sequence."""

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        num_heads = c_ // 32
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        init_values = 0.01
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            _A2C2f_AAttnKR_Unit(c_, num_heads, mlp_ratio, area) if a2 else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1).contiguous() * out
        return out


class _A2C2f_ParallelHybrid_Unit(nn.Module):
    """Per-chunk split: half AAttn, half KimiRes, then concat."""

    def __init__(self, channels, area=1):
        super().__init__()
        c_a = channels // 2
        c_b = channels - c_a
        assert c_a > 0 and c_b > 0, "Parallel hybrid channels split must be valid."
        assert c_a % 32 == 0, "AAttn branch channels must be a multiple of 32."
        self.c_a = c_a
        self.c_b = c_b
        self.aattn = AAttn(c_a, max(1, c_a // 32), area=area)
        self.kimi = KR_Block(c_b, c_b, n=1, shortcut=True, g=1, e=1.0)

    def forward(self, x):
        xa, xb = torch.split(x, [self.c_a, self.c_b], dim=1)
        ya = self.aattn(xa)
        yb = self.kimi(xb)
        return torch.cat((ya, yb), dim=1)


class A2C2f_Parallel_Hybrid(nn.Module):
    """Heterogeneous A2C2f with per-chunk AAttn/KimiRes parallel split."""

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        init_values = 0.01
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            _A2C2f_ParallelHybrid_Unit(c_, area=area) if a2 else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1).contiguous() * out
        return out

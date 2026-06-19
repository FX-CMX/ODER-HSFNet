import torch
import torch.nn as nn
from torch import Tensor

class RMSNorm2D(nn.Module):
    """
    针对 2D 视觉特征图特化的 RMSNorm
    消除了原版中仅针对 1D T序列的限制，使其直接匹配 [B, C, H, W]
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 权重维度映射为通道，从而支持广播
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        # 在通道维度求均方根进行归一化
        rms = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return x * rms * self.weight

class BlockAttnRes2D(nn.Module):
    """
    魔改版：2D 像素级动态残差块间注意力 (Pixel-wise Dynamic Inter-Block Attention)
    核心亮点：将 [N, B, C, H, W] 的跨块注意力降维至像素级 Softmax。
    不使用 O((HW)^2) 的极其耗时的 Spatial Self-Attention。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(dim, 1, kernel_size=1, bias=False)
        self.norm = RMSNorm2D(dim)
        
    def forward(self, blocks: list, partial_block: Tensor) -> Tensor:
        # blocks: 历史特征块列表，每个元素 shape [B, C, H, W]
        # partial_block: 当前增量状态，shape [B, C, H, W]
        
        # 将历史与现在拼接。V: [N+1, B, C, H, W]
        V = torch.stack(blocks + [partial_block]) 
        N1, B, C, H, W = V.shape
        
        # 降维合并 N1 和 B 进行归一化
        V_flat = V.view(N1 * B, C, H, W)
        K = self.norm(V_flat)
        
        # 1x1 Conv 降维至 1 个通道，计算 Logits
        logits = self.proj(K).view(N1, B, H, W) 
        
        # 沿第0维度 (也就是 N+1 个块维度) 做 Softmax
        # 物理意义：网络学会在整张图的每一个角落(Pixel-wise)，决定应该采信哪一个 Block 的先验！
        scores = torch.softmax(logits, dim=0) # [N+1, B, H, W]
        
        # 自适应混合 (Broadcast相乘)，最后压缩掉块的维度
        h = (scores.unsqueeze(2) * V).sum(dim=0) # [B, C, H, W]
        return h

# =====================================================================
# 改版一：Neck 级跨尺度动态同化 (Intra-Stage Multi-Scale Dynamic Residual Neck)
# 替代原先直接用 Concat + 1x1 Conv 强行挤压的方式
# =====================================================================
class DynamicFuseModule(nn.Module):
    def __init__(self, c1: list, c2: int):
        super().__init__()
        # c1 比如为 [512, 512, 1024]
        # 初始化将他们各自映射到同等通道空间的卷积
        self.align_convs = nn.ModuleList([nn.Conv2d(c, c2, 1) for c in c1])
        # 关键替换：动态层内残差
        self.dynamic_attn = BlockAttnRes2D(dim=c2)
        self.pool = nn.AvgPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, Xs):
        # 假设 Xs 为三种不同尺度的 1/8, 1/16, 1/32
        x_4 = self.align_convs[0](self.pool(Xs[0]))   # [B, C, H, W]
        x_7 = self.align_convs[1](Xs[1])              # [B, C, H, W]
        x_9 = self.align_convs[2](self.up(Xs[2]))     # [B, C, H, W]
        
        # 把原始的 x_7 当作 partial_block (锚点态)
        # x_4, x_9 当作历史 blocks (跨尺度态)
        # 不再用暴力的 torch.cat，而是用动态残差机制在每个像素端自适应决定看深层还是浅层！
        out = self.dynamic_attn(blocks=[x_4, x_9], partial_block=x_7)
        return out


# =====================================================================
# 改版二：超阶裂变末端的仲裁网 (Dynamic Attentional Fission Routing)
# 替代 HyperACE 结尾 5个分支直接 Concat 的粗暴操作
# =====================================================================
class DynamicHyperACE_Routing(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dynamic_routing = BlockAttnRes2D(dim)
        
    def forward(self, y0, y1_out1, y1_out2, y2_kept, y2_deform):
        # 原版直接把 5个 256C 的切片暴力 Concat 成了 1280C 再 1x1 降维。
        # 这里，我们把 4 个正交提取流视作 Blocks
        blocks = [y1_out1, y1_out2, y2_deform, y2_kept]
        # 基准缓冲池 (Identity flow y0) 视作 Partial Block
        partial_block = y0
        
        # 产生极度平滑的自适应响应，无需再通过 1x1 Squeeze
        fused = self.dynamic_routing(blocks, partial_block)
        return fused

# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.registry import register_model

class GlobalFeatureAggregation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的权重参数

        # Channel re-calibration
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Feature normalization
        x_norm = F.layer_norm(x.permute(0, 2, 3, 1), (C,)).permute(0, 3, 1, 2)

        # Compute spatial correlation matrix
        x_flat = rearrange(x_norm, 'b c h w -> b c (h w)')
        correlation = torch.bmm(x_flat.transpose(1, 2), x_flat)  # B, HW, HW
        correlation = F.softmax(correlation, dim=-1)

        # Global context aggregation
        context = torch.bmm(x_flat, correlation.transpose(1, 2))
        context = rearrange(context, 'b c (h w) -> b c h w', h=H)

        # Channel attention
        channel_weight = self.channel_attention(context + x)

        # Combine with original features
        out = x + self.gamma * (context * channel_weight)

        return out


class ImprovedSwapFusionBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 添加全局特征聚合模块
        self.gfa = GlobalFeatureAggregation(input_dim)

    def forward(self, x, mask=None):
        B, M, C, H, W = x.shape

        # 全局特征聚合
        x = rearrange(x, 'b m c h w -> (b m) c h w')
        x = self.gfa(x)
        x = rearrange(x, '(b m) c h w -> b m c h w', b=B)

        return x


class EnhancedFuseBEVT(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = 4




        # 使用改进的SwapFusionBlock
        for i in range(self.depth):
            block = ImprovedSwapFusionBlock(input_dim)
            self.layers.append(block)

    def forward(self, x, mask=None):
        for stage in self.layers:
            x = stage(x, mask=mask)
        return x



@register_model
def enf(**kwargs):
    model1 = EnhancedFuseBEVT(input_dim=128,**kwargs)
    return model1



if __name__ == "__main__":

    x = torch.rand([2, 3, 128, 32, 32])

    # t, B, C, H, W

    model1 = enf()


    y1 = model1(x)
    # y2 = model1(x_cat)
    # y = m(x_cat)
    print(y1.shape)
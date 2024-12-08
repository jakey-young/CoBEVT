# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.registry import register_model

class DenseRegionEnhancer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 局部密度感知
        self.local_density = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 2, 1),
            nn.BatchNorm2d(input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 2, 1, 1),
            nn.Sigmoid()
        )

        # 密集区域注意力
        self.dense_attention = nn.Sequential(
            nn.Conv2d(input_dim + 1, input_dim // 2, 1),
            nn.BatchNorm2d(input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 2, input_dim, 1),
            nn.Sigmoid()
        )

        # 局部对比增强
        self.local_contrast = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=1, groups=input_dim),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, input_dim, 1)
        )

    def forward(self, x):
        """
        x: (B*L, C, H, W)
        """
        # 1. 计算局部密度
        density = self.local_density(x)

        # 2. 密集区域注意力
        attention = self.dense_attention(torch.cat([x, density], dim=1))

        # 3. 局部对比度增强
        contrast = self.local_contrast(x)

        # 4. 在密集区域增强局部对比度
        enhanced = x + contrast * attention

        return enhanced


class EnhancedFuseBEVT(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.dense_enhancer = DenseRegionEnhancer(input_dim)

    def forward(self, x, mask=None):
        """
        x: (B, L, C, H, W)
        """
        b, l, c, h, w = x.shape

        # 1. 展平batch和agent维度
        x_flat = rearrange(x, 'b l c h w -> (b l) c h w')

        # 2. 特征增强
        enhanced = self.dense_enhancer(x_flat)

        # 3. 恢复维度
        enhanced = rearrange(enhanced, '(b l) c h w -> b l c h w', b=b, l=l)

        return enhanced



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
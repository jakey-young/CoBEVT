# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from timm.models.registry import register_model


class DensityEstimator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.density_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1)
        )

        # 自适应阈值预测
        self.threshold_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B*L, C, H, W)
        """
        # 1. 计算密度图
        density_map = self.density_conv(x)  # (B*L, 1, H, W)
        density_map = torch.sigmoid(density_map)

        # 2. 预测阈值
        thresholds = self.threshold_predictor(x)  # (B*L, 2, 1, 1)
        thresholds = thresholds.squeeze(-1).squeeze(-1)  # (B*L, 2)
        thresholds, _ = torch.sort(thresholds, dim=-1)  # 确保阈值有序

        # 3. 计算密度等级
        # 将阈值调整为与密度图相同的空间维度
        b = density_map.size(0)  # B*L
        h, w = density_map.size(2), density_map.size(3)

        # 扩展阈值维度以匹配密度图
        thresh_low = thresholds[:, 0].view(b, 1, 1, 1).expand(-1, 1, h, w)
        thresh_high = thresholds[:, 1].view(b, 1, 1, 1).expand(-1, 1, h, w)

        # 根据阈值计算密度等级
        density_levels = torch.zeros_like(density_map)
        density_levels = torch.where(density_map > thresh_high,
                                     torch.tensor(2.0).to(x.device),
                                     density_levels)
        density_levels = torch.where((density_map > thresh_low) &
                                     (density_map <= thresh_high),
                                     torch.tensor(1.0).to(x.device),
                                     density_levels)

        return density_map, density_levels, thresholds

class InstanceEnhancer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.instance_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 1)  # 点卷积进行特征重组
            ),
            nn.Sequential(
                nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 1)
            ),
            nn.Sequential(
                nn.Conv2d(dim, dim, 7, padding=3, groups=dim),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 1)
            )
        ])

        # 每个密度级别的注意力
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, dim, 1),
                nn.Sigmoid()
            ) for _ in range(3)
        ])

        # 自适应特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, density_levels):
        instance_feats = []
        for i, (conv, attn) in enumerate(zip(self.instance_conv, self.attention_layers)):
            mask = (density_levels == i).float()
            # 特征提取
            feat = conv(x)
            # 注意力增强
            feat_attn = attn(feat)
            # 密度mask和注意力加权
            feat = feat * mask * feat_attn
            instance_feats.append(feat)

        # 自适应融合
        fused_feat = self.fusion(torch.cat(instance_feats, dim=1))
        return fused_feat

class DensityAwareInstanceEnhancement(nn.Module):
    """主模块：密度感知的实例增强"""

    def __init__(self, input_dim):
        super().__init__()

        self.density_estimator = DensityEstimator(input_dim)
        self.instance_enhancer = InstanceEnhancer(input_dim)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(input_dim * 2, input_dim, 1),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True)
        )

        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, input_dim, 1, 1))

    def forward(self, x):
        """
        Args:
            x: (B, L, C, H, W) - B批次，L个agents的BEV特征
        Returns:
            enhanced_features: (B, L, C, H, W)
        """
        b, l, c, h, w = x.shape
        # 2. 添加位置编码
        x = x + self.pos_embedding.expand(-1, -1, -1, h, w)

        # 1. 展平batch和agent维度
        x_flat = rearrange(x, 'b l c h w -> (b l) c h w')


        # x_flat = x_flat + self.pos_embedding.expand(-1, -1, -1, h, w)

        # 3. 密度估计
        density_map, density_levels, _ = self.density_estimator(x_flat)

        # 4. 实例增强
        enhanced_feat = self.instance_enhancer(x_flat, density_levels)

        # 5. 残差连接和特征融合
        enhanced_feat = self.fusion(torch.cat([enhanced_feat, x_flat], dim=1))

        # 6. 恢复维度
        enhanced_feat = rearrange(enhanced_feat, '(b l) c h w -> b l c h w', b=b, l=l)

        return enhanced_feat
class EnhancedFuseBEVT(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.pre_enhance = DensityAwareInstanceEnhancement(input_dim=input_dim)
        # self.fusebevt = SwapFusionEncoder(args)

        # 添加辅助损失
        self.aux_head = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 2, 1, 1)  # 预测密度图
        )

    def forward(self, x, mask=None):
        # 密度感知增强
        enhanced_x = self.pre_enhance(x)
        return enhanced_x


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
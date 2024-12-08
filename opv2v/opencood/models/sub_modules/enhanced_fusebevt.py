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
            nn.Conv2d(in_channels // 2, 1, 1)
        )

        # 阈值定义为float类型
        self.density_thresholds = nn.Parameter(torch.tensor([0.3, 0.6], dtype=torch.float))

    def forward(self, x):
        # 1. 密度估计
        density_map = self.density_conv(x)
        density_map = torch.sigmoid(density_map)  # [0,1]范围

        # 2. 密度等级划分 (使用float类型)
        density_levels = torch.zeros_like(density_map)  # float类型
        density_levels = torch.where(density_map > self.density_thresholds[1],
                                     torch.tensor(2.0).to(x.device),
                                     density_levels)
        density_levels = torch.where((density_map > self.density_thresholds[0]) &
                                     (density_map <= self.density_thresholds[1]),
                                     torch.tensor(1.0).to(x.device),
                                     density_levels)

        return density_map, density_levels


class InstanceEnhancer(nn.Module):
    """基于密度感知的实例特征增强"""

    def __init__(self, dim):
        super().__init__()

        # 不同密度区域使用不同感受野的卷积
        self.instance_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim),  # 低密度区域
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(dim, dim, 5, padding=2, groups=dim),  # 中密度区域
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(dim, dim, 7, padding=3, groups=dim),  # 高密度区域
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            )
        ])

        # 多尺度特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        # 边界检测器
        self.boundary_detector = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 1),
            nn.Sigmoid()
        )

        # 自注意力模块用于增强实例特征
        self.instance_attention = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, density_levels):
        """
        Args:
            x: (B*L, C, H, W)
            density_levels: (B*L, 1, H, W)
        """
        # 对不同密度区域应用不同的处理
        instance_feats = []
        for i, conv in enumerate(self.instance_conv):
            mask = (density_levels == i).float()
            feat = conv(x) * mask
            instance_feats.append(feat)

        # 融合多尺度特征
        fused_feat = self.fusion(torch.cat(instance_feats, dim=1))

        # 检测实例边界
        boundaries = self.boundary_detector(fused_feat)

        # 实例注意力
        instance_attn = self.instance_attention(fused_feat)

        # 综合增强
        enhanced_feat = fused_feat * (1 + boundaries) * instance_attn

        return enhanced_feat, boundaries


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
        density_map, density_levels = self.density_estimator(x_flat)

        # 4. 实例增强
        enhanced_feat, boundaries = self.instance_enhancer(x_flat, density_levels)

        # 5. 残差连接和特征融合
        enhanced_feat = self.fusion(torch.cat([enhanced_feat, x_flat], dim=1))

        # 6. 恢复维度
        enhanced_feat = rearrange(enhanced_feat, '(b l) c h w -> b l c h w', b=b, l=l)

        return enhanced_feat


class EnhancedFuseBEVT(nn.Module):
    """包装FuseBEVT的增强模块"""

    def __init__(self, input_dim=128):
        super().__init__()
        self.pre_enhance = DensityAwareInstanceEnhancement(input_dim=input_dim)
        # self.fusebevt = SwapFusionEncoder(args)  # 原始FuseBEVT

    def forward(self, x):
        """
        Args:
            x: (B, L, C, H, W)
            mask: fusion时使用的mask
        """
        # 1. 密度感知的实例增强
        enhanced_x = self.pre_enhance(x)

        # 2. 原始FuseBEVT处理
        # output = self.fusebevt(enhanced_x, mask)

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


# 使用示例





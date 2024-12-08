import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.registry import register_model

class CameraViewProcessor(nn.Module):
    """处理单个尺度下的多相机特征"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.spatial_enhance = nn.Sequential(
            # 保持相机视角的独立性
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 相机视角间的交互
        self.cross_view_mixer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, out_channels, 1)
        )

        # 视角注意力c
        self.view_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B*L*M, C, H, W]
        identity = x.clone()

        # 空间特征增强
        spatial_feat = self.spatial_enhance(identity)

        # 视角注意力
        attention = self.view_attention(spatial_feat)
        enhanced = spatial_feat * attention

        # 特征混合
        output = self.cross_view_mixer(enhanced)

        return output + x


class HierarchicalFeatureEnhancement(nn.Module):
    """层次化特征增强模块"""

    def __init__(self, channels_list):
        """
        Args:
            channels_list: 每个尺度的通道数列表 [128, 256, 512]
        """
        super().__init__()

        self.processors = nn.ModuleList([
            CameraViewProcessor(c, c) for c in channels_list
        ])

        # 尺度间交互
        self.scale_interaction = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels_list[i] + channels_list[i + 1], channels_list[i], 1),
                nn.BatchNorm2d(channels_list[i]),
                nn.ReLU(inplace=True)
            ) for i in range(len(channels_list) - 1)
        ])

        # 尺度注意力
        self.scale_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, c // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c // 4, c, 1),
                nn.Sigmoid()
            ) for c in channels_list
        ])

    def forward(self, features):
        """
        Args:
            features: 特征金字塔列表 [(B,L,M,C1,H1,W1), (B,L,M,C2,H2,W2), (B,L,M,C3,H3,W3)]
        """
        b, l, m = features[0].shape[:3]
        enhanced_features = []

        # 1. 各尺度特征增强
        for i, (feat, processor) in enumerate(zip(features, self.processors)):
            # 重组维度以处理所有batch和视角
            feat = rearrange(feat, 'b l m c h w -> (b l m) c h w')

            # 特征增强
            enhanced = processor(feat)

            # 恢复维度
            enhanced = rearrange(enhanced, '(b l m) c h w -> b l m c h w',
                                 b=b, l=l, m=m)
            enhanced_features.append(enhanced)

        # 2. 尺度间交互 (从高层到低层)
        for i in range(len(enhanced_features) - 2, -1, -1):
            # 上采样高层特征
            high_feat = F.interpolate(
                rearrange(enhanced_features[i + 1], 'b l m c h w -> (b l m) c h w'),
                size=enhanced_features[i].shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            # 当前层特征
            curr_feat = rearrange(enhanced_features[i], 'b l m c h w -> (b l m) c h w')

            # 特征融合
            fused = self.scale_interaction[i](torch.cat([curr_feat, high_feat], dim=1))

            # 尺度注意力
            scale_attn = self.scale_attention[i](fused)
            fused = fused * scale_attn

            # 更新特征
            enhanced_features[i] = rearrange(fused, '(b l m) c h w -> b l m c h w',
                                             b=b, l=l, m=m)

        return enhanced_features


class EnhancedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = config['encoder']  # 原始encoder
        channels = [128, 256, 512]  # 三个尺度的通道数
        self.feature_enhancement = HierarchicalFeatureEnhancement(channels)

    def forward(self, features):
        # 1. 原始特征提取
        # features = self.encoder(x)  # 得到多尺度特征列表

        # 2. 特征增强
        enhanced_features = self.feature_enhancement(features)

        return enhanced_features


@register_model
def ssa(**kwargs):
    model1 = EnhancedEncoder( **kwargs)
    return model1



if __name__ == "__main__":
    x0 = torch.rand([2, 1, 4, 128, 64, 64])
    x1 = torch.rand([2, 1, 4, 256, 32, 32])
    x2 = torch.rand([2, 1, 4, 512, 16, 16])
    x_list = [x0, x1, x2]
    # t, B, C, H, W

    model1 = ssa()


    y1 = model1(x_list)
    # y2 = model1(x_cat)
    # y = m(x_cat)
    print(y1.shape)
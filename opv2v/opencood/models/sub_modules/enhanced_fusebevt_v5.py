# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.registry import register_model


class ImprovedSwapFusionBlock(nn.Module):
    def __init__(self, input_dim, window_size):
        super().__init__()
        self.window_size = window_size

        # 1. Local Feature Enhancement Module
        self.local_enhancer = nn.ModuleDict({
            'channel_mixer': nn.Sequential(
                nn.Conv2d(input_dim, input_dim, 1),
                nn.GroupNorm(8, input_dim),
                nn.GELU()
            ),
            'spatial_mixer': nn.Sequential(
                nn.Conv2d(input_dim, input_dim, 3, padding=1, groups=input_dim),
                nn.GroupNorm(8, input_dim),
                nn.GELU()
            ),
            'gate': nn.Sequential(
                nn.Conv2d(input_dim, input_dim, 1),
                nn.Sigmoid()
            )
        })

        # 2. Global Feature Enhancement Module
        self.global_enhancer = nn.ModuleDict({
            'context': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(input_dim, input_dim, 1),
                nn.GroupNorm(8, input_dim),
                nn.GELU(),
                nn.Conv2d(input_dim, input_dim, 1)
            ),
            'mixer': nn.Sequential(
                nn.Conv2d(input_dim * 2, input_dim, 1),
                nn.GroupNorm(8, input_dim),
                nn.GELU()
            )
        })


    def enhance_local_features(self, x):
        # x shape: (B*M, C, H, W)
        channel_feats = self.local_enhancer['channel_mixer'](x)
        spatial_feats = self.local_enhancer['spatial_mixer'](x)

        # Compute attention gate
        gate = self.local_enhancer['gate'](channel_feats + spatial_feats)
        enhanced = x * gate

        return enhanced

    def enhance_global_features(self, x):
        # x shape: (B*M, C, H, W)
        # Extract global context
        context = self.global_enhancer['context'](x)
        context = context.expand(-1, -1, x.shape[2], x.shape[3])

        # Combine local and global features
        combined = torch.cat([x, context], dim=1)
        enhanced = self.global_enhancer['mixer'](combined)

        return enhanced


class ContrastiveFeatureSeparator(nn.Module):
    def __init__(self, dim, window_size):
        super().__init__()
        self.window_size = window_size

        # 实例特征投影器
        self.instance_projector = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, x, return_loss=True):
        """
        Args:
            x: (B, M, C, H, W) - 经过local特征增强的特征
        """
        B, M, C, H, W = x.shape

        # 1. 投影到实例特征空间
        x_flat = rearrange(x, 'b m c h w -> (b m) c h w')
        instance_feats = self.instance_projector(x_flat)  # (B*M, C, H, W)

        # 2. 在window内计算对比损失
        if self.training and return_loss:
            # 重排为window-level特征
            instance_feats_window = rearrange(
                instance_feats,
                'n c (h p1) (w p2) -> (n h w) (p1 p2) c',
                p1=self.window_size, p2=self.window_size
            )  # (N*H'*W', window_size^2, C)

            # 计算window内的特征相似度
            sim_matrix = torch.einsum(
                'bpc,bqc->bpq',
                instance_feats_window,
                instance_feats_window
            )  # (N*H'*W', window_size^2, window_size^2)
            sim_matrix = sim_matrix / self.temperature

            # 生成对比学习的mask - 每个位置只与自己为正样本
            pos_mask = torch.eye(
                self.window_size ** 2,
                device=x.device
            )[None].expand(B * M * H // self.window_size * W // self.window_size, -1, -1)

            # 计算对比损失
            loss = -torch.log_softmax(sim_matrix, dim=-1) * pos_mask
            self.contrast_loss = loss.sum() / (loss.shape[0] * self.window_size ** 2)

        return rearrange(instance_feats, '(b m) c h w -> b m c h w', b=B)

    # def forward(self, x, mask=None):
    #     """
    #     Args:
    #         x: shape (B, M, C, H, W)
    #         mask: optional attention mask
    #     """
    #     B, M, C, H, W = x.shape
    #
    #     # 1. Local window attention with enhancement
    #     # 1.1 Enhance features
    #     x_local = rearrange(x, 'b m c h w -> (b m) c h w')
    #     x_local = self.enhance_local_features(x_local)
    #     x_local = rearrange(x_local, '(b m) c h w -> b m c h w', b=B)
    #
    #     # 1.2 Apply local window attention
    #     x_local = rearrange(x_local,
    #                         'b m c (x w1) (y w2) -> b m x y w1 w2 c',
    #                         w1=self.window_size, w2=self.window_size)
    #     x_local = self.local_attention(x_local, mask)
    #     x_local = self.local_ffn(x_local)
    #     x_local = rearrange(x_local,
    #                         'b m x y w1 w2 c -> b m c (x w1) (y w2)')
    #
    #     # 2. Global grid attention with enhancement
    #     # 2.1 Enhance features
    #     x_global = rearrange(x_local, 'b m c h w -> (b m) c h w')
    #     x_global = self.enhance_global_features(x_global)
    #     x_global = rearrange(x_global, '(b m) c h w -> b m c h w', b=B)
    #
    #     # 2.2 Apply global grid attention
    #     x_global = rearrange(x_global,
    #                          'b m c (w1 x) (w2 y) -> b m x y w1 w2 c',
    #                          w1=self.window_size, w2=self.window_size)
    #     x_global = self.global_attention(x_global, mask)
    #     x_global = self.global_ffn(x_global)
    #     x_global = rearrange(x_global,
    #                          'b m x y w1 w2 c -> b m c (w1 x) (w2 y)')
    #
    #     return x_global


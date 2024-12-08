# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.registry import register_model


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



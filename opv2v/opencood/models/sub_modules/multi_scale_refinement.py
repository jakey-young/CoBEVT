# -*-coding:utf-8-*-
import torch.nn as nn
import torch
from timm.models.registry import register_model
from torch import einsum
from einops import rearrange, repeat, reduce
import torch.nn.functional as F
from math import log, pi
import math
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Scale_Attention(nn.Module):
    def __init__(self, in_channels=128, dropout=0.1):
        super(Scale_Attention, self).__init__()

        self.norm = nn.LayerNorm(in_channels)
        self.num_heads = 2
        self.act = nn.GELU()
        self.q = nn.Linear(in_channels, in_channels, bias=True)
        self.kv1 = nn.Linear(in_channels, in_channels, bias=True)
        self.kv2 = nn.Linear(in_channels, in_channels, bias=True)
        head_dim = in_channels // self.num_heads
        self.scale = head_dim ** -0.5
        self.attn_drop = nn.Dropout(dropout)
        self.local_conv1 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, stride=1, groups=in_channels // 2)
        self.local_conv2 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, stride=1, groups=in_channels // 2)
        self.sr1 = nn.Conv2d(in_channels, in_channels, kernel_size=8, stride=8)
        self.norm1 = nn.LayerNorm(in_channels)
        self.sr2 = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4)
        self.norm2 = nn.LayerNorm(in_channels)
    def forward(self, attended_features):
        _,_, H, W = attended_features[2].shape
        features_q = self.norm(attended_features[2].flatten(2).transpose(1, 2))
        B, N, C = features_q.shape
        q = self.q(features_q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x_1 = self.act(self.norm1(self.sr1(attended_features[1]).reshape(B, C, -1).permute(0, 2, 1)))
        x_2 = self.act(self.norm2(self.sr2(attended_features[0]).reshape(B, C, -1).permute(0, 2, 1)))
        kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # B, T, N, C = features_q.shape
        # q = self.q(features_q).reshape(B, T, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        k1, v1 = kv1[0], kv1[1]  # B head N C
        k2, v2 = kv2[0], kv2[1]
        attn1 = (q[:, :self.num_heads // 2] @ k1.transpose(-2, -1)) * self.scale
        attn1 = self.attn_drop(attn1)
        v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).
                                   transpose(1, 2).view(B, C // 2, 8, 8)). \
            view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
        x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2)
        attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).
                                   transpose(1, 2).view(B, C // 2, 32, 32)). \
            view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
        x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C//2)

        x = torch.cat([x1,x2], dim=-1)
        return x.view(B,C,H,W)



@register_model
def ssa(**kwargs):
    model1 = Scale_Attention(in_channels=128, dropout=0.1, **kwargs)
    return model1



if __name__ == "__main__":
    x0 = torch.rand([1, 128, 128, 128])
    x1 = torch.rand([1, 128, 64, 64])
    x2 = torch.rand([1, 128, 32, 32])
    x_list = [x0, x1, x2]
    # t, B, C, H, W

    model1 = ssa()


    y1 = model1(x_list)
    # y2 = model1(x_cat)
    # y = m(x_cat)
    print(y1.shape)

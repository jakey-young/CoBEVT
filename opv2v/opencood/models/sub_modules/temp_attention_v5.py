# -*-coding:utf-8-*-
# -*-coding:utf-8-*-

# 时空注意力引入patch


import torch.nn as nn
import torch
from timm.models.registry import register_model
from torch import einsum
from einops import rearrange, repeat, reduce
import torch.nn.functional as F
import math
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Global_Attention(nn.Module):
    def __init__(self, in_channels, num_heads, dropout=0.1, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_dim = in_channels * patch_size * patch_size
        self.attention = nn.MultiheadAttention(self.patch_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(self.patch_dim)

    def reshape_to_patches(self, x, patch_size):
        B, T, C, H, W = x.shape
        P = patch_size
        x = x.reshape(B, T, C, H // P, P, W // P, P)
        x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        x = x.view(B, T, (H // P) * (W // P), C * P * P)
        return x

    def forward(self, features):
        B, T, C, H, W = features.shape
        P = self.patch_size

        # 重塑为patches
        x_patched = self.reshape_to_patches(features, P)

        # 重塑为注意力输入格式
        x_att = x_patched.permute(1, 0, 2, 3).contiguous()
        x_att = x_att.view(T, B * (H // P) * (W // P), self.patch_dim)

        # 应用注意力
        attended, _ = self.attention(x_att, x_att, x_att)

        # 重塑回原始格式
        attended = attended.view(T, B, H // P, W // P, self.patch_dim)
        attended = attended.permute(1, 0, 2, 3, 4)

        # 残差连接和归一化
        output = self.norm(attended.view(B, T, -1, self.patch_dim) + x_patched)

        # 重塑回原始空间维度
        output = output.view(B, T, H // P, W // P, C, P, P)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        output = output.view(B, T, C, H, W)

        return output
class MultiScale_Global_Attention(nn.Module):
    def __init__(self, in_channels=128, num_scales=3, num_heads=8, dropout=0.1):
        super(MultiScale_Global_Attention, self).__init__()

        self.attention = nn.ModuleList([
            Global_Attention(in_channels=in_channels, num_heads=num_heads, dropout=dropout)
            for _ in range(num_scales)
        ])
    def forward(self, multi_scale_features):
        attended_features = []
        for i, features in enumerate(multi_scale_features):
            attended = self.attention[i](features)
            attended_features.append(attended)
        return attended_features

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

    def forward(self, attended_features):
        _,_,_, H, W = attended_features[0].shape
        features_q = self.norm(attended_features[0].flatten(3).transpose(3, 2))
        features_kv1 = self.act(self.norm(attended_features[1].flatten(3).transpose(3, 2)))
        features_kv2 = self.act(self.norm(attended_features[2].flatten(3).transpose(3, 2)))
        B, T, N, C = features_q.shape
        q = self.q(features_q).reshape(B, T, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        kv1 = self.kv1(features_kv1).reshape(B, T, -1, 2, self.num_heads // 2, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        kv2 = self.kv1(features_kv2).reshape(B, T, -1, 2, self.num_heads // 2, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        k1, v1 = kv1[0], kv1[1]  # B head N C
        k2, v2 = kv2[0], kv2[1]
        attn1 = (q[:, :, :self.num_heads // 2] @ k1.transpose(-2, -1)) * self.scale
        attn1 = self.attn_drop(attn1)
        # v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).
        #                            transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)). \
        #     view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
        x1 = (attn1 @ v1).transpose(2, 3).reshape(B, T, N, C // 2)
        attn2 = (q[:, :, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        # v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).
        #                            transpose(1, 2).view(B, C // 2, H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)). \
        #     view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
        x2 = (attn2 @ v2).transpose(2, 3).reshape(B, T, N, C // 2)

        x = torch.cat([x1, x2], dim=-1).view(B, T, C, H, W)
        return x

class AdaptiveTemporalWindow(nn.Module):
    def __init__(self, in_channels, max_window_size=3, threshold=0.1):
        super(AdaptiveTemporalWindow, self).__init__()
        self.max_window_size = max_window_size
        self.temporal_attention = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出单个权重值
        )
        self.threshold = threshold
    def forward(self, x):
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # 计算时间注意力权重
        x_flat = x.view(B, T, C, -1).mean(dim=-1)  # [B, T]
        temporal_logits = self.temporal_attention(x_flat).squeeze(-1)  # [B, T]
        temporal_weights = F.softmax(temporal_logits, dim=1)  # [B, T]
        window_size = (temporal_weights > (1.0 / self.max_window_size)).sum(dim=1)
        window_size = torch.clamp(window_size, min=1, max=self.max_window_size)

        return window_size


class AdaptiveTemporalWindowV2(nn.Module):
    def __init__(self, in_channels, max_window_size=3, threshold=0.1):
        super(AdaptiveTemporalWindowV2, self).__init__()
        self.max_window_size = max_window_size
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1))  # 保留时间维度
        )
        self.temporal_conv = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.importance_estimator = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # 提取时空特征
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        features = self.feature_extractor(x)  # [B, 64, T, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [B, 64, T]

        # 分析时间维度上的变化
        temporal_features = self.temporal_conv(features)  # [B, 32, T]

        # 估计每个时间步的重要性
        importance_scores = self.importance_estimator(temporal_features.permute(0, 2, 1)).squeeze(-1)  # [B, T]

        # 计算累积重要性
        cumulative_importance = torch.cumsum(F.softmax(importance_scores, dim=1), dim=1)

        # 决定窗口大小
        window_size = torch.sum(cumulative_importance < 0.9, dim=1) + 1  # 90% 的重要性作为阈值
        window_size = torch.clamp(window_size, min=1, max=self.max_window_size)

        return window_size


class DynamicWeightGenerator(nn.Module):
    def __init__(self, in_channels, hidden_dim=256):
        super().__init__()
        self.conv3d_3 = nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(in_channels, hidden_dim, kernel_size=(2, 3, 3), padding=(1, 1, 1))
        self.conv3d_1 = nn.Conv3d(in_channels, hidden_dim, kernel_size=(1, 3, 3), padding=(1, 1, 1))
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels)
        )
        self.weight_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

        # 3D convolution to process spatial-temporal features
        if T == 3:
            x = F.relu(self.conv3d_3(x))
        elif T == 2:
            x = F.relu(self.conv3d_2(x))
        else:
            x = F.relu(self.conv3d_1(x))
        x = x.mean(dim=(3, 4)).permute(0,2,1)

        # Reshape for temporal attention
        # x = x.permute(0, 2, 1, 3, 4).reshape(B*H*W, T, -1)  # [B, T, C*H*W]

        # Temporal attention
        x, _ = self.temporal_attention(x, x, x)

        # Generate dynamic weights
        x = x.mean(dim=1)  # [B, C*H*W]
        dynamic_weight = self.fc(x)
        dynamic_weight = torch.tanh(dynamic_weight) * self.weight_scale

        return dynamic_weight


class STAGuidedDynamicWeightNetwork(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.weight_generator = DynamicWeightGenerator(in_channels)
        self.feature_transform = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # Generate dynamic weights
        dynamic_weight = self.weight_generator(x)  # [B, C]

        # Apply dynamic weights to each time step
        transformed = self.feature_transform(x.view(-1, C, H, W)).view(B, T, C, H, W)
        weighted_feature = transformed * dynamic_weight.view(B, 1, C, 1, 1)

        # Temporal weighted sum
        temporal_weights = F.softmax(dynamic_weight, dim=1)
        enhanced_feature = (weighted_feature * temporal_weights.view(B, 1, C, 1, 1)).sum(dim=1)

        return enhanced_feature



# class DilateTempAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
#                  attn_drop=0.,proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.dilation = dilation
#         self.kernel_size = kernel_size
#         self.scale = qk_scale or head_dim ** -0.5
#         self.num_dilation = len(dilation)
#         assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
#         self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
#         self.dilate_attention = nn.ModuleList(
#             [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
#              for i in range(self.num_dilation)])
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#
#         B, C, H, W = x.shape
#         qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
#         # num_dilation,3,B,C//num_dilation,H,W
#         x = x.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)
#         # num_dilation, B, H, W, C//num_dilation
#         for i in range(self.num_dilation):
#             x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, H, W,C//num_dilation
#         x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x



class MSTIA(nn.Module):
    def __init__(self, in_channels=128, num_scales=3, num_heads=8, dropout=0.1, max_window_size=3, threshold=0.1):
        super(MSTIA, self).__init__()
        self.num_scales = num_scales
        self.scales = [1, 2, 4]

        # 多尺度特征提取
        self.down_sampling = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=s, padding=1)
            for s in self.scales[1:]
        ])

        # 全局时空注意力
        self.global_temp_attention = MultiScale_Global_Attention(in_channels=in_channels, num_scales=num_scales, num_heads=num_heads, dropout=dropout)

        # 尺度间交互
        self.scale_interaction = Scale_Attention(in_channels=in_channels, dropout=dropout)

        # 自适应窗口
        self.adapt_window = AdaptiveTemporalWindowV2(in_channels=in_channels, max_window_size=max_window_size, threshold=threshold)

        # 自适应特征聚合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(in_channels * num_scales, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        self.dynamce_weigeht = STAGuidedDynamicWeightNetwork(in_channels=in_channels)
        # self.temp_attention = DilateTempAttention()
    def forward(self, x):

        # 自适应窗口
        windows_num = int(self.adapt_window(x)[0])
        x = x[:, 3-windows_num:, :]
        b, t, c, h, w = x.shape
        # 多尺度特征提取
        multi_scale_features = [x]
        for down in self.down_sampling:
            multi_scale_features.append(rearrange(down(rearrange(x,'b t c h w -> (b t) c h w')), '(b t) c H W -> b t c H W', b=b, t=t))

        # 时空全局注意力
        attended_features = self.global_temp_attention(multi_scale_features)

        # 尺度间交互
        x = self.scale_interaction(attended_features)
        # 基于动态权重融合
        x = self.dynamce_weigeht(x)

        return x


@register_model
def mstia(**kwargs):
    model1 = MSTIA(in_channels=128, num_scales=3, num_heads=8, dropout=0.1, **kwargs)
    return model1






if __name__ == "__main__":

    # x_list = []
    # for i in range(5):
    #     x = torch.rand([i+1, 4, 128, 32, 32])
    #     x_list.append(x)
    # arr = align_tensors(x_list)
#-----------------------------------------
    x_list = []
    for i in range(3):
        x = torch.rand([1, 1, 128, 32, 32])
        x_list.append(x)
    # t, B, C, H, W
    x_cat = torch.cat(x_list, dim=1)
    model1 = mstia()


    y1 = model1(x_cat)
    # y2 = model1(x_cat)
    # y = m(x_cat)
    print(y1.shape)
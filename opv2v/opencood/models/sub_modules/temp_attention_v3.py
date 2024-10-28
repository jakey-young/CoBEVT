# -*-coding:utf-8-*-
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
    def __init__(self, in_channels=128, num_scales=3, num_heads=8, dropout=0.1):
        super(Global_Attention, self).__init__()
        self.attention_layers = nn.MultiheadAttention(in_channels, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(in_channels)
    def forward(self, features):
        b, t, c, h, w = features.shape
        features_att = features.permute(1, 0, 2, 3, 4).contiguous().view(t, b * h * w, c)
        attended, _ = self.attention_layers(features_att, features_att, features_att)
        attended_features = attended.view(t, b, h, w, c).permute(1, 0, 4, 2, 3)
        return self.norm((attended_features + features).permute(0,1,3,4,2)).permute(0,1,4,2,3)

class MultiScale_Global_Attention(nn.Module):
    def __init__(self, in_channels=128, num_scales=3, num_heads=8, dropout=0.1):
        super(MultiScale_Global_Attention, self).__init__()

        self.attention = nn.ModuleList([
            Global_Attention(in_channels=in_channels, num_scales=num_scales, num_heads=num_heads, dropout=dropout)
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

        # 细化时空注意力
        self.temp_attention = Global_Attention(in_channels=in_channels, num_scales=num_scales, num_heads=num_heads, dropout=dropout)
        # 自适应特征聚合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(in_channels * num_scales, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        self.dynamce_weigeht = STAGuidedDynamicWeightNetwork(in_channels=in_channels)
        # self.temp_attention = DilateTempAttention()
    def forward(self, x):
        b, t, c, h, w = x.shape

        # 多尺度特征提取
        multi_scale_features = [x]
        for down in self.down_sampling:
            multi_scale_features.append(rearrange(down(rearrange(x,'b t c h w -> (b t) c h w')), '(b t) c H W -> b t c H W', b=b, t=t))

        # 时空全局注意力
        attended_features = self.global_temp_attention(multi_scale_features)

        # 尺度间交互
        temp_features = self.scale_interaction(attended_features)

        # 自适应窗口
        windows_num = int(self.adapt_window(x)[0])

        # 时序注意力
        # for i in range(windows_num):
        #     x = temp_features[-1-i, :].permute(1, 0, 2, 3, 4)
        #     x = self.temp_attention(x)
        x = self.temp_attention(temp_features[:, 3-windows_num:, :])
        x = self.dynamce_weigeht(x)

        # 自适应特征聚合
        # upsampled_features = []
        # for i, features in enumerate(interacted_features):
        #     if i > 0:
        #         features = F.interpolate(features.view(b * t, c, h, w), scale_factor=self.scales[i], mode='bilinear',
        #                                  align_corners=False)
        #         features = features.view(b, t, c, h * self.scales[i], w * self.scales[i])
        #     upsampled_features.append(features)
        #
        # fused = torch.cat(upsampled_features, dim=2)
        # fused = self.feature_fusion(fused.view(b * t, -1, h, w))
        # fused = fused.view(b, t, c, h, w)
        #
        # # 残差连接和归一化
        # output = self.norm(fused + x)

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
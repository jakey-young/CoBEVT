# -*-coding:utf-8-*-
# 动态时间窗口模块放在第一步

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

# =========================================================================================
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


class AdaptiveTemporalWindowV8(nn.Module):
    def __init__(self, in_channels, max_window_size=3):
        super().__init__()
        self.training = True
        self.max_window_size = max_window_size
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )
        self.temporal_conv = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.window_predictor = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, max_window_size)
        )

    def forward(self, x):
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        features = self.feature_extractor(x).squeeze(-1).squeeze(-1)  # [B, 64, T]
        temporal_features = self.temporal_conv(features)  # [B, 32, T]
        temporal_features = temporal_features.mean(dim=2)  # [B, 32]

        window_logits = self.window_predictor(temporal_features)  # [B, max_window_size]

        # Soft selection (for gradient flow)
        window_probs = F.softmax(window_logits, dim=1)
        selected_window_soft = torch.sum(window_probs * torch.arange(1, self.max_window_size + 1).to(x.device), dim=1)

        # Hard selection (for actual use)
        selected_window_hard = torch.argmax(window_logits, dim=1) + 1

        if self.training:
            # Use Straight-Through Estimator
            selected_window = (selected_window_hard.detach() - selected_window_soft).detach() + selected_window_soft
        else:
            selected_window = selected_window_hard

        return selected_window, window_logits
# ===================================================================================================================


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
        # self.adapt_window = AdaptiveTemporalWindowV2(in_channels=in_channels, max_window_size=max_window_size, threshold=threshold)
        self.adapt_window = AdaptiveTemporalWindowV8(in_channels=in_channels, max_window_size=max_window_size)

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

        # 自适应窗口
        # windows_num = int(self.adapt_window(x)[0])
        temp_features = x.clone()
        selected_window, window_logits = self.adapt_window(x)
        windows_num = int(selected_window)

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

        return x, temp_features, selected_window, window_logits


class WindowSelectionLoss(nn.Module):
    def __init__(self, sim_type='cosine'):
        super().__init__()
        self.sim_type = sim_type

    def compute_dissimilarity(self, x1, x2):
        if self.sim_type == 'cosine':
            x1_flat = x1.view(x1.size(0), -1)
            x2_flat = x2.view(x2.size(0), -1)
            return 1 - F.cosine_similarity(x1_flat, x2_flat)
        elif self.sim_type == 'l2':
            return torch.norm(x1 - x2, dim=(1, 2, 3))

    def forward(self, features, selected_window, window_logits):
        B, T, C, H, W = features.shape

        # 计算完整序列的差异（真值）
        full_seq_dissim = 0
        for b in range(B):
            for t in range(1, T):
                full_seq_dissim += self.compute_dissimilarity(features[b, t], features[b, t - 1])
            full_seq_dissim /= (T - 1)

        # 计算选定窗口的差异
        window_dissim = 0
        for b in range(B):
            window_size = int(selected_window[b].item())
            for t in range(T - window_size, T):
                window_dissim += self.compute_dissimilarity(features[b, t], features[b, t - 1])
            window_dissim /= window_size

        # 计算差异的逆相关性
        divergence = 1 / (1 + torch.exp(-(full_seq_dissim - window_dissim)))

        # 窗口选择的交叉熵损失
        ce_loss = F.cross_entropy(window_logits, (selected_window - 1).long())

        # 总损失
        loss = -torch.log(divergence + 1e-6) + ce_loss

        return loss.mean()

@register_model
def mstia(**kwargs):
    model1 = MSTIA(in_channels=128, num_scales=3, num_heads=8, dropout=0.1, **kwargs)
    return model1

@register_model
def model_loss(**kwargs):
    model2 = WindowSelectionLoss(**kwargs)
    return model2




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
    model_loss = model_loss()


    y1, temp_features, selected_window, window_logits = model1(x_cat)
    loss = model_loss(temp_features, selected_window, window_logits)
    # y2 = model1(x_cat)
    # y = m(x_cat)
    print(y1.shape)
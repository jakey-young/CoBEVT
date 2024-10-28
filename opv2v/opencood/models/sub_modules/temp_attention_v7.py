# -*-coding:utf-8-*-
# without wgh_token

# -*-coding:utf-8-*-


# 时空注意力引入patch


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

class TemporalFeatureEnhancer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 时序位置编码
        self.temporal_pos_embed_3 = nn.Parameter(torch.randn(3, dim))
        self.temporal_pos_embed_2 = nn.Parameter(torch.randn(2, dim))
        self.temporal_pos_embed_1 = nn.Parameter(torch.randn(1, dim))

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        # 重要性权重生成
        self.importance_predictor = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, 1),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1)
        )

    def forward(self, attended_features, features):
        """
        features: [B, T, H, W, C] - 经过时空注意力的特征
        """
        B, T, H, W, C = features.shape

        # 添加时序位置信息（感觉可有可无）
        if T == 3:
            temporal_pos_embed = self.temporal_pos_embed_3
        elif T == 2:
            temporal_pos_embed = self.temporal_pos_embed_2
        else:
            temporal_pos_embed = self.temporal_pos_embed_1

        pos_embed = temporal_pos_embed.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, T, 1, 1, C]
        features = features + pos_embed  # 每个时序位置添加其独特的位置编码
        current_frame = features[:, -1]  # 当前帧特征

        # 调整维度顺序
        current_frame = current_frame.permute(0, 3, 1, 2)  # [B, C, H, W]
        # historical_frames = features.permute(0, 1, 4, 2, 3)  # [B, T-1, C, H, W]

        # 为每个历史帧生成重要性权重
        weighted_features = []
        for t in range(T):
            frame_feat = features[:, t].permute(0,3,1,2)  # [B, C, H, W]
            weight_feat = attended_features[:, t].permute(0,3,1,2)
            # 通道重要性权重
            channel_weights = self.channel_attention(weight_feat)
            frame_feat = frame_feat * channel_weights

            # 生成空间重要性权重
            spitial_weight = self.importance_predictor(weight_feat)  # [B, 1, H, W]
            frame_feat = frame_feat * spitial_weight
            weighted_features.append(frame_feat)

        weighted_features = torch.stack(weighted_features, dim=1)  # [B, T, 1, H, W]

        # 加权融合历史帧信息
        weighted_features = weighted_features.sum(dim=1)  # [B, C, H, W]


        # 最终融合
        enhanced = self.fusion(torch.cat([current_frame, weighted_features], dim=1))

        return enhanced


def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rot_emb(q, k, rot_emb):
    sin, cos = rot_emb
    rot_dim = sin.shape[-1]
    (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :rot_dim], t[..., rot_dim:]), (q, k))
    q, k = map(lambda t: t * cos + rotate_every_two(t) * sin, (q, k))
    q, k = map(lambda t: torch.cat(t, dim = -1), ((q, q_pass), (k, k_pass)))
    return q, k

class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        scales = torch.logspace(0., log(max_freq / 2) / log(2), self.dim // 4, base = 2)
        self.register_buffer('scales', scales)

    def forward(self, h, w, device):
        scales = rearrange(self.scales, '... -> () ...')
        scales = scales.to(device)

        h_seq = torch.linspace(-1., 1., steps = h, device = device)
        h_seq = h_seq.unsqueeze(-1)

        w_seq = torch.linspace(-1., 1., steps = w, device = device)
        w_seq = w_seq.unsqueeze(-1)

        h_seq = h_seq * scales * pi
        w_seq = w_seq * scales * pi

        x_sinu = repeat(h_seq, 'i d -> i j d', j = w)
        y_sinu = repeat(w_seq, 'j d -> i j d', i = h)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j = 2), (sin, cos))
        return sin, cos

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freqs = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, n, device):
        seq = torch.arange(n, device = device)
        freqs = einsum('i, j -> i j', seq, self.inv_freqs)
        freqs = torch.cat((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, 'n d -> () n d')
        return freqs.sin(), freqs.cos()

def exists(val):
    return val is not None

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

# time token shift

def shift(t, amt):
    if amt is 0:
        return t
    return F.pad(t, (0, 0, 0, 0, amt, -amt))

class PreTokenShift(nn.Module):
    def __init__(self, frames, fn):
        super().__init__()
        self.frames = frames
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        f, dim = self.frames, x.shape[-1]
        cls_x, x = x[:, :1], x[:, 1:]
        x = rearrange(x, 'b (f n) d -> b f n d', f = f)

        # shift along time frame before and after

        dim_chunk = (dim // 3)
        chunks = x.split(dim_chunk, dim = -1)
        chunks_to_shift, rest = chunks[:3], chunks[3:]
        shifted_chunks = tuple(map(lambda args: shift(*args), zip(chunks_to_shift, (-1, 0, 1))))
        x = torch.cat((*shifted_chunks, *rest), dim = -1)

        x = rearrange(x, 'b f n d -> b (f n) d')
        x = torch.cat((cls_x, x), dim = 1)
        return self.fn(x, *args, **kwargs)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# attention

def attn(q, k, v, mask = None):
    sim = einsum('b i d, b j d -> b i j', q, k)

    if exists(mask):
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, mask = None, cls_mask = None, rot_emb = None, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        q = q * self.scale
        # rearrange across time or space
        q, k, v = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q, k, v))
        # add rotary embeddings, if applicable
        if exists(rot_emb):
            q, k = apply_rot_emb(q, k, rot_emb)

        # attention
        out = attn(q, k, v, mask = mask)
        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)
        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        return self.to_out(out)




class DirTimeSformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size = 32,
        patch_size = 4,
        depth = 3,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.,
        rotary_emb = True,
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        patch_dim_s = dim * patch_size**2
        patch_dim_t = dim * patch_size

        self.heads = heads
        self.patch_size = patch_size
        self.to_patch_embedding_s = nn.Linear(patch_dim_s, dim)
        self.to_patch_embedding_t = nn.Linear(patch_dim_t, dim)
        self.att_from_t_to_s = nn.Linear(dim, dim//patch_size)
        self.att_from_s_to_t = nn.Linear(dim, dim*patch_size)

        self.dyweight = nn.Parameter(torch.randn(1, dim))

        self.use_rotary_emb = rotary_emb

        self.frame_rot_emb = RotaryEmbedding(dim_head)
        self.image_rot_emb = AxialRotaryEmbedding(dim_head)

        self.out = nn.Linear(dim//patch_size, dim)



        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ff = FeedForward(dim, dropout = ff_dropout)
            time_attn = Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
            spatial_attn = Attention(dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
            time_attn, spatial_attn, ff = map(lambda t: PreNorm(dim, t), (time_attn, spatial_attn, ff))

            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))



    def forward(self, frame_features, mask = None):
        b, f, c, h, w, device, p = *frame_features.shape, frame_features.device, self.patch_size
        # b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

        # calculate num patches in height and width dimension, and number of total patches (n)

        hp, wp = (h // p), (w // p)
        # n = hp * wp
        n_t = h * wp

        # video to patch embeddings

        # frame_features_s = rearrange(frame_features, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1 = p, p2 = p)
        frame_features_t = rearrange(frame_features, 'b f c h (w p2) -> b (f h w) (p2 c)', p2 = p)
        # tokens_s = self.to_patch_embedding_s(frame_features_s)
        x = self.to_patch_embedding_t(frame_features_t)

        # add dynamic weight token
        # dyweight_token = repeat(self.dyweight, 'n d -> b n d', b=b)


        # x = torch.cat((dyweight_token, tokens_s), dim=1)
        # x_t = torch.cat((dyweight_token, tokens_t), dim=1)

        # positional embedding
        frame_pos_emb = self.frame_rot_emb(f, device = device)
        image_pos_emb = self.image_rot_emb(hp, wp, device = device)

        # calculate masking for uneven number of frames

        frame_mask = None
        cls_attn_mask = None


        # time and space attention

        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n = n_t, mask = frame_mask, cls_mask = cls_attn_mask, rot_emb = frame_pos_emb) + x

            x_s = self.att_from_t_to_s(x)
            x_s = rearrange(x_s, 'b (f h w) (p2 c) -> b f h (w p2) c', f=f, h=h, p2=p)
            x = rearrange(x_s, 'b f (h p1) (w p2) c -> b (f h w) (p1 p2 c)', p1=p, p2=p)

            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f, cls_mask = cls_attn_mask, rot_emb = image_pos_emb) + x
            x = ff(x) + x

            x_t = self.att_from_s_to_t(x)
            x_t = rearrange(x_t, 'b (f n) (p1 p2 c) -> b f (n p1 p2) c', f=f, p1=p, p2=p).view(b,f,h,w,-1)
            x = rearrange(x_t, 'b f h (w p2) c -> b (f h w) (p2 c)', p2=p)
        x = self.out(rearrange(x, 'b (f h w) (p2 c) -> b f h (w p2) c', f=f, h=h, p2=p))
        return x

class MSTIA(nn.Module):
    def __init__(self, in_channels=128, image_size=32, patch_size=4,
                        depth=3, num_heads=8, dim_head=64, attn_dropout=0.,
                        ff_dropout=0., rotary_emb=True,  max_window_size=3, threshold=0.1):
        super(MSTIA, self).__init__()
        # 全局时空注意力
        self.global_temp_attention = DirTimeSformer(dim=in_channels, image_size=image_size, patch_size=patch_size,
                                                    depth=depth, heads=num_heads, dim_head=dim_head, attn_dropout=attn_dropout,
                                                    ff_dropout=ff_dropout, rotary_emb=rotary_emb)
        # 自适应窗口
        self.adapt_window = AdaptiveTemporalWindowV8(in_channels=in_channels, max_window_size=max_window_size)

        # 动态权重
        self.dynamic_weigeht = TemporalFeatureEnhancer(dim=in_channels)

    def forward(self, x):

        # 自适应窗口
        windows_num = int(self.adapt_window(x)[0])
        x = x[:, 3-windows_num:, :]

        # 时空全局注意力
        attended_features = self.global_temp_attention(x)

        # 基于动态权重融合

        x = self.dynamic_weigeht(attended_features, x.permute(0,1,3,4,2))
        return x


@register_model
def mstia(**kwargs):
    model1 = MSTIA(in_channels=128, image_size=32, patch_size=4,
                        depth=3, num_heads=8, dim_head=64, attn_dropout=0.,
                        ff_dropout=0., rotary_emb=True,  max_window_size=3, threshold=0.1, **kwargs)
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

# -*-coding:utf-8-*-
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

# class TempSpatialAttention(nn.Module):
#     "Implementation of Dilate-attention"
#     def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=[3,3], dilation=[3,2]):
#         super().__init__()
#         self.head_dim = head_dim
#         self.scale = qk_scale or head_dim ** -0.5
#         self.kernel_size=kernel_size
#         self.cov_3d = nn.Conv3d(128, 2*128, [2,1,1], 1, 0, 1)
#         self.unfold1 = nn.Unfold(kernel_size[0], dilation[0], dilation[0]*(kernel_size[0]-1)//2, 1)
#         self.unfold2 = nn.Unfold(kernel_size[1], dilation[1], dilation[1]*(kernel_size[1]-1)//2, 1)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.num_channel = 2
#
#     def forward(self,x_cat):
#         #B, C//4, H, W
#         q = x_cat.permute(2,0,1,3,4)[0:1, :]
#         B,C,t,H,W = x_cat.shape
#
#         kv = rearrange(self.cov_3d(x_cat),'b (n d) t h w ->t n b d h w', n=2)
#
#
#
#         q = q.reshape([B, C//self.head_dim, self.head_dim, 1 ,H*W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
#
#         # 与frame2和frame3的融合时序特征作注意力计算
#         k1 = self.unfold1(kv[1][0]).reshape([B, C//self.head_dim, self.head_dim, self.kernel_size[0]*self.kernel_size[0], H*W]).permute(0, 1, 4, 2, 3)  #B,h,N,d,k*k
#         attn1 = (q @ k1) * self.scale  # B,h,N,1,k*k
#         attn1 = attn1.softmax(dim=-1)
#         attn1 = self.attn_drop(attn1)
#         v1 = self.unfold1(kv[1][1]).reshape([B, C//self.head_dim, self.head_dim, self.kernel_size[0]*self.kernel_size[0], H*W]).permute(0, 1, 4, 3, 2)  # B,h,N,k*k,d
#         q1 = (attn1 @ v1).transpose(1, 2).reshape(B, H, W, C)
#
#         q1 = q1.reshape([B, C // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
#
#         # 与frame1和frame2的融合时序特征作注意力计算
#         k2 = self.unfold2(kv[0][0]).reshape([B, C//self.head_dim, self.head_dim, self.kernel_size[1]*self.kernel_size[1], H*W]).permute(0, 1, 4, 2, 3)  #B,h,N,d,k*k
#         attn2 = (q1 @ k2) * self.scale  # B,h,N,1,k*k
#         attn2 = attn2.softmax(dim=-1)
#         attn2 = self.attn_drop(attn2)
#         v2 = self.unfold2(kv[0][1]).reshape([B, C//self.head_dim, self.head_dim, self.kernel_size[1]*self.kernel_size[1], H*W]).permute(0, 1, 4, 3, 2)  # B,h,N,k*k,d
#         q2 = (attn2 @ v2).transpose(1, 2).reshape(B, H, W, C)
#         return q2

#----------------------当前时刻数据作为q-------------------------------------------------




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class DilateTempSpatialModule(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = 32
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.cov_3d = nn.Conv3d(128, 128, 3, 1, 1, 3)
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # B, C//4, H, W
        B, d, H, W = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2,
                                                                                                        3)  # B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3,
                                                                                                        2)  # B,h,N,k*k,d
        x = F.relu((attn @ v).transpose(1, 2).reshape(B, H, W, d), inplace=False)
        return x
class MultiDilateTempSpatialModule(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim=128, num_heads=4, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = 32
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.temp_frame_num = 2
        assert num_heads % self.temp_frame_num == 0, f"num_heads{num_heads} must be the times of num_dilation{self.temp_frame_num}!!"
        self.q = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, 2*dim, 1, bias=qkv_bias)
        self.cov_3d = nn.Conv3d(128, 128, 3, 1, 1, 3)
        self.norm = nn.LayerNorm(dim)
        self.TSA = nn.ModuleList(
            [DilateTempSpatialModule(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.temp_frame_num)])
        # self.TSA = TempSpatialAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation)
        self.proj = nn.Sequential(nn.Linear(dim, dim),
                                  nn.Dropout(proj_drop))
        self.proj_drop = nn.Dropout(proj_drop)
        self.layer_norm = nn.LayerNorm(dim)
        self.kv_generate = nn.Conv2d(2*dim, dim, 1)
        self.pose_emb = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
    def forward(self, x, hx):
        B, C, H, W= x.shape
        x0 = x.clone()
        x_hx = self.kv_generate(torch.cat((x0, hx), dim=1))
        # x = x.permute(0, 3, 1, 2)# B, C, H, W
        q = self.q(x_hx).reshape(B, self.temp_frame_num, C // self.temp_frame_num, H, W).permute(1, 0, 2, 3, 4)
        kv = self.kv(x0).reshape(B, 2, self.temp_frame_num, C // self.temp_frame_num, H, W).permute(2, 1, 0, 3, 4, 5)
        #num_dilation,3,B,C//num_dilation,H,W
        x = x.reshape(B, self.temp_frame_num, C // self.temp_frame_num, H, W).permute(1, 0, 3, 4, 2)
        # num_dilation, B, H, W, C//num_dilation
        for i in range(self.temp_frame_num):
            x[i] = self.TSA[i](q[i], kv[i][0], kv[i][1]) # B, H, W,C//num_dilation
        x = x.reshape(B, H, W, C) + x0.permute(0, 2, 3, 1)
        x = self.proj_drop(self.proj(self.norm((x)))) + x
        # x = self.proj(x)
        # x = self.proj_drop(x)
        return self.layer_norm(x)

class DilateTempAttentionBlock(nn.Module):
    "Implementation of Dilate-attention block"
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2, 3],
                 cpe_per_block=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = MultiDilateTempSpatialModule(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x_cat):

        if self.cpe_per_block:
            x_cat = x_cat + self.pos_embed(x_cat)

        x = x_cat[0].permute(0, 2, 3, 1)
        hx = x_cat[1].permute(0, 2, 3, 1)
        x = self.norm1(x).permute(0, 3, 1, 2)
        hx = self.norm1(hx).permute(0, 3, 1, 2)
        x = x + self.drop_path(self.attn(x, hx)).permute(0, 3, 1, 2)
        x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)))).permute(0, 3, 1, 2)
        #B, C, H, W
        return x

class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """
    def __init__(self, in_channels, out_channels, cpe_per_satge, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cpe_per_satge = cpe_per_satge

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            norm_layer(out_channels))

        if self.cpe_per_satge:
            self.pos_embed = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels)

    def forward(self, x):
        #x: B, C, H ,W
        x = self.proj(x)
        if self.cpe_per_satge:
            x = x + self.pos_embed(x)
        return x


class DilateTempAttentionStage(nn.Module):
    """ A basic Dilate Transformer layer for one stage.
    """
    def __init__(self, dim, num_heads, kernel_size, dilation,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, cpe_per_satge=False, cpe_per_block=False,
                 downsample=True, depth=3):

        super().__init__()
        # build blocks
        self.blocks = nn.ModuleList([
            DilateTempAttentionBlock(dim=dim, num_heads=num_heads,
                                     kernel_size=kernel_size, dilation=dilation,
                                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                     qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block)
            for i in range(depth)])

        # patch merging layer
        self.downsample = PatchMerging(dim, dim, cpe_per_satge) if downsample else nn.Identity()

    def forward(self, x, hx):
        for blk in self.blocks:
            x = torch.cat((x.unsqueeze(0), hx.unsqueeze(0)), dim=0)
            x = blk(x)
        x = self.downsample(x)
        return x



class FaxLSTMCell(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias, qk_scale,attn_drop, proj_drop, kernel_size, dilation):
        super().__init__()

        # self.temporal_fusion = TemporalFusion_lstm()
        # self.temp_attention = nn.Sequential(TempSpatialModule(dim=128, num_heads=4, qkv_bias=False, qk_scale=None,
        #                                         attn_drop=True, kernel_size=3, dilation=dilation[i]))
        self.temp_num = len(dilation)
        self.temp_attention = nn.ModuleList([DilateTempAttentionStage(dim=dim, num_heads=num_heads, kernel_size=kernel_size,
                                                       dilation=dilation[i], mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                       drop=0., attn_drop=attn_drop, drop_path=0.) for i in range(self.temp_num)])

    def pixel_shift(self, x, shift_x, shift_y):
        B, C, H, W = x.shape

        # 计算需要填充的像素数
        pad_left = max(shift_x, 0)
        pad_right = max(-shift_x, 0)
        pad_top = max(shift_y, 0)
        pad_bottom = max(-shift_y, 0)

        # 使用 replicate 模式填充图像边缘
        x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')

        # 从填充后的图像中裁剪出移动后的图像
        return x_padded[:, :,
               pad_top:pad_top + H,
               pad_left:pad_left + W]
    def forward(self, xt, hidden_states, id, inputs_len):
        B, C, H, W = xt.shape
        if hidden_states is None:
            # B, C, H, W = xt.shape
            # hx = torch.zeros(B, C, H, W).to(xt.device)
            # cx = torch.zeros(B, C, H, W).to(xt.device)
            hx = self.pixel_shift(xt.clone(),0, -2)
            cx = self.pixel_shift(xt.clone(),0, -2)
        else:
            hx, cx = hidden_states

        Ft = self.temp_attention[id](xt, hx)
        if id < inputs_len - 1:
            gate = torch.sigmoid(Ft)
            cell = torch.tanh(Ft)

            cy = gate * (cx + cell)
            hy = gate * torch.tanh(cy)
            hx = hy
            cx = cy
            return hx, (hx, cx)
        else:
            return Ft, (None, None)



class TemPatchEmbed(nn.Module):
    def __init__(self, in_chans, out_chns, embed_dim):
        super(TemPatchEmbed, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, out_chns, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_chns),
            nn.GELU(),
            nn.Conv2d(out_chns, out_chns * 2, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_chns * 2),
            nn.GELU(),
            nn.Conv2d(out_chns * 2, out_chns * 4, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_chns * 4),
            nn.GELU(),
            nn.Conv2d(out_chns * 4, embed_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),
        )
        return
    def forward(self, x):
        x = self.proj(x)
        return x

class FaxLSTMCellMoudle(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_scale,
                 attn_drop, proj_drop, kernel_size, dilation, embed_dim):
        super(FaxLSTMCellMoudle, self).__init__()
        self.cell = FaxLSTMCell(dim, num_heads, qkv_bias, qk_scale,attn_drop, proj_drop, kernel_size, dilation)
        self.patch_embed = TemPatchEmbed(in_chans=dim, out_chns=dim, embed_dim=dim)

    def forward(self, x, h, id, inputs_len):
        # x = self.patch_embed(x)

        hidden_states = []
        x = self.patch_embed(x)
        x, hidden_state = self.cell(x, h[0], id, inputs_len)
        hidden_states.append(hidden_state)

        # x = torch.sigmoid(x)

        return x, hidden_states

#-----------------------scope的重要性提取模块-------------------------------
class SpatialAttention_mtf(nn.Module):
    def __init__(self, kernel_size=35):
        super(SpatialAttention_mtf, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=2, bias=False)
        self.conv2 = nn.Conv2d(1, 128, kernel_size, padding=2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.Tan = nn.Tanh()

    def forward(self, curr, prev):
        curr_avg_out = torch.mean(curr, dim=1, keepdim=True)
        curr_max_out, _ = torch.max(curr, dim=1, keepdim=True)
        curr_merge = torch.cat([curr_avg_out, curr_max_out], dim=1)
        prev_avg_out = torch.mean(prev, dim=1, keepdim=True)
        prev_max_out, _ = torch.max(prev, dim=1, keepdim=True)
        prev_merge = torch.cat([prev_avg_out, prev_max_out], dim=1)
        # prev_avg_out = torch.mean(torch.sum(prev, dim=0, keepdim=True), dim=1, keepdim=True)
        # prev_max_out, _ = torch.max(torch.sum(prev, dim=0, keepdim=True), dim=1, keepdim=True)
        # prev_merge = torch.cat([prev_avg_out, prev_max_out], dim=1)
        merge = self.sigmoid(self.conv1(curr_merge + prev_merge))
        final_out = (1 - merge) * self.Tan(curr) + merge * prev

        return final_out

class model_forward_single_layer(nn.Module):
    def __init__(self, dim=128, num_heads=2, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[[3, 3], [3, 2], [2, 1]], embed_dim=128):
        super().__init__()
        self.temp_fuse = FaxLSTMCellMoudle(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 attn_drop=attn_drop, proj_drop=proj_drop, kernel_size=kernel_size, dilation=dilation, embed_dim=embed_dim)
        self.imp_att = SpatialAttention_mtf(kernel_size=5)

    def forward(self,inputs):
        outputs = []
        states = [None]

        inputs_len = inputs.shape[0]
        last_input = inputs[-1,:]

        for i in range(inputs_len):
            output, states = self.temp_fuse(inputs[i, :], states, i, inputs_len)
            outputs.append(output)


        # xt_fuse = model2(last_input, output)
        xt_fuse = self.imp_att(last_input, output)
        return xt_fuse + last_input


# def model_forward_single_layer(model1, model2, inputs):
#     outputs = []
#     states = [None]
#
#     inputs_len = inputs.shape[0]
#     last_input = inputs[-1,:]
#
#     for i in range(inputs_len):
#         output, states = model1(inputs[i, :], states, i, inputs_len)
#         outputs.append(output)
#
#
#     # xt_fuse = model2(last_input, output)
#     xt_fuse = model2(last_input, output)
#     return xt_fuse + last_input



def align_tensors(tensor_list):
    # 找出第一个维度的最大值
    max_b = max(tensor.shape[0] for tensor in tensor_list)

    aligned_tensors = []

    for tensor in tensor_list:
        if tensor.shape[0] < max_b:
            # 如果当前张量的第一个维度小于最大值，需要进行填充
            padding = torch.zeros(max_b - tensor.shape[0], *tensor.shape[1:], dtype=tensor.dtype,
                                  device=tensor.device)
            aligned_tensor = torch.cat([tensor, padding], dim=0)
        else:
            # 如果当前张量的第一个维度已经是最大值，不需要改变
            aligned_tensor = tensor

        aligned_tensors.append(aligned_tensor)

    return aligned_tensors

@register_model
def dilate_lstm(**kwargs):
    model1 = FaxLSTMCellMoudle(dim=128, num_heads=2, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[[3, 3], [3, 2], [2, 1]], embed_dim=128, **kwargs)
    return model1

@register_model
def temp_fuse(**kwargs):
    model2 = SpatialAttention_mtf(kernel_size=5, **kwargs)
    return model2

# @register_model
# def temp_fuse_uncertain(**kwargs):
#     model3 = UncertaintyAwareAttention(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, **kwargs)
#     return model3


if __name__ == "__main__":

    # x_list = []
    # for i in range(5):
    #     x = torch.rand([i+1, 4, 128, 32, 32])
    #     x_list.append(x)
    # arr = align_tensors(x_list)
#-----------------------------------------
    x_list = []
    for i in range(3):
        x = torch.rand([3, 1, 128, 32, 32])
        x_list.append(x)
    # t, B, C, H, W
    x_cat = torch.cat(x_list, dim=1).permute(1, 0, 2, 3, 4)
    speed = torch.rand([3, 3, 1])
    t, Bl, C, H, W = x_cat.shape
    model1 = dilate_lstm()
    model2 = temp_fuse()
    # model3 = temp_fuse_uncertain()
    model4 = SpatialAttention_mtf()

    y1 = model_forward_single_layer(model1, model2, x_cat)
    # y2 = model1(x_cat)
    # y = m(x_cat)
    print(y1.shape)
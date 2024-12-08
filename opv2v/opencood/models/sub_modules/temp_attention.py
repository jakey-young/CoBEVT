# -*-coding:utf-8-*-

import torch.nn as nn
import torch
from timm.models.registry import register_model
from torch import einsum
from einops import rearrange, repeat, reduce
import torch.nn.functional as F
import math
import numpy as np

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



class TempSpatialAttention(nn.Module):
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
        # x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x
class TempSpatialModule(nn.Module):
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
        self.q = nn.Conv2d(dim, dim, 1, bias=True)
        self.kv = nn.Conv2d(dim, 2*dim, 1, bias=qkv_bias)
        self.cov_3d = nn.Conv3d(128, 128, 3, 1, 1, 3)
        self.norm = nn.LayerNorm(dim)
        self.TSA = nn.ModuleList(
            [TempSpatialAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.temp_frame_num)])
        # self.TSA = TempSpatialAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x, hx):
        B, C, H, W= x.shape
        x0 = x.clone()
        # x = x.permute(0, 3, 1, 2)# B, C, H, W
        q = self.q(x0).reshape(B, self.temp_frame_num, C // self.temp_frame_num, H, W).permute(1, 0, 2, 3, 4)
        kv = self.kv(hx).reshape(B, 2, self.temp_frame_num, C // self.temp_frame_num, H, W).permute(2, 1, 0, 3, 4, 5)
        #num_dilation,3,B,C//num_dilation,H,W
        x = x.reshape(B, self.temp_frame_num, C // self.temp_frame_num, H, W).permute(1, 0, 3, 4, 2)
        # num_dilation, B, H, W, C//num_dilation
        for i in range(self.temp_frame_num):
            x[i] = self.TSA[i](q[i], kv[i][0], kv[i][1]) # B, H, W,C//num_dilation
        x = x.reshape(B, H, W, C) + x0.permute(0, 2, 3, 1)
        x = self.proj_drop(self.proj(self.norm((x)))) + x
        # x = self.proj(x)
        # x = self.proj_drop(x)
        return x.permute(0, 3, 1, 2)

#------------------------------t,t-1融合数据作为q----------------------------

class MFSA_ffd(nn.Module):
    def __init__(self, dim, attn_drop):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(attn_drop),
            nn.Linear(dim * 2, dim),
            nn.Dropout(attn_drop)
        )

    def forward(self, x):
        return self.net(self.norm(x)) + x


class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.cov_3d = nn.Conv3d(128, 128, 3, 1, 1, 3)
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # B, C//4, H, W
        B, H, W, d = q.shape
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
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class TempFuseDilateAttention(nn.Module):
    def __init__(self, dim=128, num_heads=2, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=1):
        super().__init__()
        self.temp_frame_num = num_heads
        head_dim = 32
        self.q = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, 2 * dim, 1, bias=qkv_bias)
        # self.layers =nn.ModuleList(nn.Conv3d((i+1)*dim, (i+2)*dim, (2,1,1), 1, 0, 1) for i in range(num_heads-1))
        self.layers = nn.ModuleList(nn.Conv3d(dim, dim, (2, 1, 1), 1, 0, 1) for i in range(num_heads - 1))
        # self.bev_fuse = nn.ModuleList([DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation)
        #                                for i in range(self.cav_num)])
        self.MSA = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.temp_frame_num)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)



    def forward(self, q, x):
        Bl, t, H, W, C = x.shape

        _, _, _, C_q = q.shape
        # x = x.permute(0, 3, 1, 2)# B, C, H, W
        # q = q.reshape(B, self.cav_num, C_q//self.cav_num, H, W).permute(1, 0, 2,  3, 4)
        # x.permute(0, 2, 1, 3, 4)

        kv = self.kv(rearrange(x, 'b t h w c-> (b t) c h w ')).reshape(Bl, 2, t, H, W, C).permute(2, 1, 0, 3, 4, 5)
        # num_dilation,3,B,C//num_dilation,H,W
        # x = rearrange(x, '(b l) c h w -> b l c h w', l=L).permute(1, 0, 3, 4, 2)
        # num_dilation, B, H, W, C//num_dilation
        # for i in range(self.cav_num):
        #     q = self.MSA[i](q, kv[i][0], kv[i][1])# B, H, W,C//num_dilation
        for i in range(self.temp_frame_num):
            q = self.MSA[i](q, kv[i][0], kv[i][1])  # B, H, W,C//num_dilation
        q = self.proj(q)
        q = self.proj_drop(q)
        return q


class MultiSclaeTempFuseAttention(nn.Module):
    def __init__(self, dim=128, num_heads=2, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.attn_layer = TempFuseDilateAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(dim)
        self.conv3d = nn.Conv3d(dim, dim, (2, 1, 1), 1, 0, 1)

    def forward(self, xt, hx):
        xt = xt.unsqueeze(2)
        hx = hx.unsqueeze(2)
        q_cat = torch.cat((hx, xt), 2)
        q = self.norm(rearrange(self.conv3d(q_cat),'b c t h w -> (b t) h w c'))
        kv = self.norm(q_cat.permute(0, 2, 3, 4, 1))
        Ft_output = self.attn_layer(q, kv)
        return Ft_output.permute(0, 3, 1, 2)
#----------------------------------------------------------------------------------------------------------------------
class FaxLSTMCell(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias, qk_scale,attn_drop, proj_drop, kernel_size, dilation):
        super().__init__()

        # self.temporal_fusion = TemporalFusion_lstm()
        # self.temp_attention = nn.Sequential(TempSpatialModule(dim=128, num_heads=4, qkv_bias=False, qk_scale=None,
        #                                         attn_drop=True, kernel_size=3, dilation=dilation[i]))
        self.temp_num = len(dilation)
        self.temp_attention = nn.ModuleList(
            [TempSpatialModule(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation[i])
             for i in range( self.temp_num)])

    def forward(self, xt, hidden_states, id, inputs_len):
        if hidden_states is None:
            B, C, H, W = xt.shape
            # hx = torch.zeros(B, C, H, W).to(xt.device)
            # cx = torch.zeros(B, C, H, W).to(xt.device)
            hx = xt.clone()
            cx = xt.clone()

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



class TemPoseEmbed(nn.Module):
    def __init__(self, embed_dim):
        super(TemPoseEmbed, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        return
    def forword(self, x):
        x = self.norm(x.flatten(2))

        return x

class FaxLSTMCellMoudle(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_scale,
                 attn_drop, proj_drop, kernel_size, dilation, embed_dim):
        super(FaxLSTMCellMoudle, self).__init__()
        self.cell = FaxLSTMCell(dim, num_heads, qkv_bias, qk_scale,attn_drop, proj_drop, kernel_size, dilation)
        self.patch_embed = TemPoseEmbed(embed_dim)

    def forward(self, x, h, id, inputs_len):
        # x = self.patch_embed(x)

        hidden_states = []

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



#-------------------------本文方法：不确定性时空重要性特征提取模块-----------------------------------
class AdaptiveWeightConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(AdaptiveWeightConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (4, 256, 32, 32)
        conv_out = self.conv(x)  # shape: (4, 128, 32, 32)

        # Channel attention
        channel_weight = self.channel_attention(conv_out)
        conv_out = conv_out * channel_weight

        # Spatial attention
        spatial_weight = self.spatial_attention(conv_out)
        conv_out = conv_out * spatial_weight

        return conv_out


class UncertaintyAwareAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_attn = AdaptiveWeightConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_uncertainty = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, real):
        attn = self.sigmoid(self.conv_attn(torch.cat((pred, real), dim=1)))
        uncertainty = self.sigmoid(self.conv_uncertainty(torch.abs(pred - real)))
        return attn * (1 - uncertainty)

def model_forward_single_layer(model1, model3, inputs):
    outputs = []
    states = [None]

    inputs_len = inputs.shape[0]
    last_input = inputs[-1,:]

    for i in range(inputs_len):
        output, states = model1(inputs[i, :], states, i, inputs_len)
        outputs.append(output)


    # xt_fuse = model2(last_input, output)
    xt_fuse = model3(last_input, output)
    return xt_fuse + last_input



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

@register_model
def temp_fuse_uncertain(**kwargs):
    model3 = UncertaintyAwareAttention(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, **kwargs)
    return model3


if __name__ == "__main__":

    # x_list = []
    # for i in range(5):
    #     x = torch.rand([i+1, 4, 128, 32, 32])
    #     x_list.append(x)
    # arr = align_tensors(x_list)
#-----------------------------------------
    x_list = []
    for i in range(3):
        x = torch.rand([4, 1, 128, 32, 32])
        x_list.append(x)
    # t, B, C, H, W
    x_cat = torch.cat(x_list, dim=1).permute(1, 0, 2, 3, 4)
    t, Bl, C, H, W = x_cat.shape
    model1 = dilate_lstm()
    model2 = temp_fuse()
    model3 = temp_fuse_uncertain()

    y1 = model_forward_single_layer(model1, model3, x_cat)
    # y2 = model1(x_cat)
    # y = m(x_cat)
    print(y1.shape)
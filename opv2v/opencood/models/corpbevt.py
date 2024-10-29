"""
Implementation of Brady Zhou's cross view transformer
"""
# import einops
# import numpy as np
# import torch.nn as nn
# import torch
# from einops import rearrange
# from opencood.models.sub_modules.fax_modules import FAXModule
# from opencood.models.backbones.resnet_ms import ResnetEncoder
# from opencood.models.sub_modules.naive_decoder import NaiveDecoder
# from opencood.models.sub_modules.bev_seg_head import BevSegHead
# from opencood.models.sub_modules.naive_compress import NaiveCompressor
# from opencood.models.fusion_modules.swap_fusion_modules import \
#     SwapFusionEncoder
# from opencood.models.sub_modules.fuse_utils import regroup
# from opencood.models.sub_modules.torch_transformation_utils import \
#     get_transformation_matrix, warp_affine, get_roi_and_cav_mask, \
#     get_discretized_transformation_matrix
# from opencood.models.sub_modules.temp_attention_v4 import MSTIA
# import torch.nn.functional as F
# import math



#=================================V7==============================================
"""
Implementation of Brady Zhou's cross view transformer
"""
import einops
import numpy as np
import torch.nn as nn
import torch
from einops import rearrange
from opencood.models.sub_modules.fax_modules import FAXModule
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fusion_modules.swap_fusion_modules import \
    SwapFusionEncoder
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    get_transformation_matrix, warp_affine, get_roi_and_cav_mask, \
    get_discretized_transformation_matrix
from opencood.models.sub_modules.temp_attention_v7 import MSTIA



# from opencood.data_utils.augmentor.bev_embedding_augmentor import NoiseCombiner, SignalToNoise, ZeroOut, FullZeroOut


class STTF(nn.Module):
    def __init__(self, args):
        super(STTF, self).__init__()
        self.discrete_ratio = args['resolution']
        self.downsample_rate = args['downsample_rate']

    def forward(self, x, spatial_correction_matrix):
        """
        Transform the bev features to ego space.

        Parameters
        ----------
        x : torch.Tensor
            B L C H W
        spatial_correction_matrix : torch.Tensor
            Transformation matrix to ego

        Returns
        -------
        The bev feature same shape as x but with transformation
        """
        dist_correction_matrix = get_discretized_transformation_matrix(
            spatial_correction_matrix, self.discrete_ratio,
            self.downsample_rate)

        # transpose and flip to make the transformation correct
        x = rearrange(x, 'b l c h w  -> b l c w h')
        x = torch.flip(x, dims=(4,))
        # Only compensate non-ego vehicles
        B, L, C, H, W = x.shape

        T = get_transformation_matrix(
            dist_correction_matrix[:, :, :, :].reshape(-1, 2, 3), (H, W))
        cav_features = warp_affine(x[:, :, :, :, :].reshape(-1, C, H, W), T,
                                   (H, W))
        cav_features = cav_features.reshape(B, -1, C, H, W)

        # flip and transpose back
        x = cav_features
        x = torch.flip(x, dims=(4,))
        x = rearrange(x, 'b l c w h -> b l h w c')

        return x

class CorpBEVT(nn.Module):
    def __init__(self, config):
        super(CorpBEVT, self).__init__()
        self.max_cav = config['max_cav']
        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # cvm params
        fax_params = config['fax']
        fax_params['backbone_output_shape'] = self.encoder.output_shapes
        self.fax = FAXModule(fax_params)

        # self.temporal_fusion = model_forward_single_layer(dim=128, num_heads=2, qkv_bias=True, qk_scale=None,
        #          attn_drop=0., proj_drop=0., kernel_size=3, dilation=[[3, 3], [3, 2], [2, 1]], embed_dim=128)
        # self.temporal_fusion = MSTIA(in_channels=128, num_scales=3, num_heads=8, dropout=0.1, max_window_size=3, threshold=0.1)
        self.temporal_fusion =MSTIA(in_channels=128, image_size=32, patch_size=4,
                        depth=3, num_heads=8, dim_head=64, attn_dropout=0.,
                        ff_dropout=0., rotary_emb=True,  max_window_size=3, threshold=0.1)

        if config['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(128, config['compression'])
        else:
            self.compression = False

        # spatial feature transform module
        self.downsample_rate = config['sttf']['downsample_rate']
        self.discrete_ratio = config['sttf']['resolution']
        self.use_roi_mask = config['sttf']['use_roi_mask']
        self.sttf = STTF(config['sttf'])

        # spatial fusion
        self.fusion_net = SwapFusionEncoder(config['fax_fusion'])

        # decoder params
        decoder_params = config['decoder']
        # decoder for dynamic and static differet
        self.decoder = NaiveDecoder(decoder_params)

        self.target = config['target']
        self.seg_head = BevSegHead(self.target,
                                   config['seg_head_dim'],
                                   config['output_class'])

    def align_tensors(self, tensor_list):
        # 找出第一个维度的最大值
        max_b = max(tensor.shape[0] for tensor in tensor_list)

        # 创建一个新的列表来存储对齐后的张量
        aligned_tensors = []

        for tensor in tensor_list:
            if tensor.shape[0] < max_b:
                # 如果当前张量的第一个维度小于最大值，我们需要进行填充
                padding = torch.zeros(max_b - tensor.shape[0], *tensor.shape[1:], dtype=tensor.dtype,
                                      device=tensor.device)
                aligned_tensor = torch.cat([tensor, padding], dim=0)
            else:
                # 如果当前张量的第一个维度已经是最大值，不需要改变
                aligned_tensor = tensor

            aligned_tensors.append(aligned_tensor.unsqueeze(0))

        return aligned_tensors

    def forward(self, batch_dict_list):
        # x, transformation_matrix, record_len= self.transform_feature(batch_dict)
        single_bev = []

        # record_len = batch_dict_list[0]['record_len']
        # transformation_matrix = batch_dict_list[0]['transformation_matrix']

        for i in range(len(batch_dict_list)):
            record_len = []
            transformation_matrix = []
            for id, batch_dict in batch_dict_list[i].items():
                x = batch_dict['inputs'] # (b,1,4,512,512,3)(connected_car,batch,cam_num,H,W,C)
                b, t, m, _, _, _ = x.shape

                # shape: (B, max_cav, 4, 4)
                record_len.append(batch_dict['record_len'])
                transformation_matrix.append(batch_dict['transformation_matrix'])

                x = self.encoder(x)
                batch_dict.update({'features': x})
                x = self.fax(batch_dict)
                # x = torch.rand([b, t, 128, 32, 32]).to(torch.device('cuda'))
                record_len = record_len[0]
                transformation_matrix = transformation_matrix[0]

                # B*L, C, H, W
                x = x.squeeze(1)

                # compressor
                if self.compression:
                    x = self.naive_compressor(x)

                # Reformat to (B, max_cav, C, H, W)
                x, mask = regroup(x, record_len, self.max_cav)
                # perform feature spatial transformation,  B, max_cav, H, W, C
                x = self.sttf(x, transformation_matrix) # 这个模块利用了坐标转换矩阵,实现将BEV特征从世界坐标系转換到自身坐标系,应该是为了后续在自身坐标系下分析处理BEV特征
                com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
                    3) if not self.use_roi_mask \
                    else get_roi_and_cav_mask(x.shape,
                                              mask,
                                              transformation_matrix,
                                              self.discrete_ratio,
                                              self.downsample_rate)

                # fuse all agents together to get a single bev map, b h w c
                x = rearrange(x, 'b l h w c -> b l c h w')
                x = self.fusion_net(x, com_mask)
                # x = torch.randn([b, 128, 32, 32]).to(torch.device('cuda'))
                single_bev.append(x)
        single_bev = self.align_tensors(single_bev)
        single_bev = single_bev[::-1]    # [t,t-1,t-2] -> [t-2,t-1,t]
        x = self.temporal_fusion(torch.cat(single_bev, dim=0).permute(1,0,2,3,4))
        x = x.unsqueeze(1)

        # dynamic head
        x = self.decoder(x)
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        b = x.shape[0]
        output_dict = self.seg_head(x, b, 1)

        return output_dict

#=================================================================================================================



# class STTF(nn.Module):
#     def __init__(self, args):
#         super(STTF, self).__init__()
#         self.discrete_ratio = args['resolution']
#         self.downsample_rate = args['downsample_rate']
#
#     def forward(self, x, spatial_correction_matrix):
#         """
#         Transform the bev features to ego space.
#
#         Parameters
#         ----------
#         x : torch.Tensor
#             B L C H W
#         spatial_correction_matrix : torch.Tensor
#             Transformation matrix to ego
#
#         Returns
#         -------
#         The bev feature same shape as x but with transformation
#         """
#         dist_correction_matrix = get_discretized_transformation_matrix(
#             spatial_correction_matrix, self.discrete_ratio,
#             self.downsample_rate)
#
#         # transpose and flip to make the transformation correct
#         x = rearrange(x, 'b l c h w  -> b l c w h')
#         x = torch.flip(x, dims=(4,))
#         # Only compensate non-ego vehicles
#         B, L, C, H, W = x.shape
#
#         T = get_transformation_matrix(
#             dist_correction_matrix[:, :, :, :].reshape(-1, 2, 3), (H, W))
#         cav_features = warp_affine(x[:, :, :, :, :].reshape(-1, C, H, W), T,
#                                    (H, W))
#         cav_features = cav_features.reshape(B, -1, C, H, W)
#
#         # flip and transpose back
#         x = cav_features
#         x = torch.flip(x, dims=(4,))
#         x = rearrange(x, 'b l c w h -> b l h w c')
#
#         return x
#
#
# class CorpBEVT(nn.Module):
#     def __init__(self, config):
#         super(CorpBEVT, self).__init__()
#         self.max_cav = config['max_cav']
#         # encoder params
#         self.encoder = ResnetEncoder(config['encoder'])
#
#         # cvm params
#         fax_params = config['fax']
#         fax_params['backbone_output_shape'] = self.encoder.output_shapes
#         self.fax = FAXModule(fax_params)
#
#         if config['compression'] > 0:
#             self.compression = True
#             self.naive_compressor = NaiveCompressor(128, config['compression'])
#         else:
#             self.compression = False
#
#         # spatial feature transform module
#         self.downsample_rate = config['sttf']['downsample_rate']
#         self.discrete_ratio = config['sttf']['resolution']
#         self.use_roi_mask = config['sttf']['use_roi_mask']
#         self.sttf = STTF(config['sttf'])
#
#         # spatial fusion
#         self.fusion_net = SwapFusionEncoder(config['fax_fusion'])
#
#         # decoder params
#         decoder_params = config['decoder']
#         # decoder for dynamic and static differet
#         self.decoder = NaiveDecoder(decoder_params)
#
#         self.target = config['target']
#         self.seg_head = BevSegHead(self.target,
#                                    config['seg_head_dim'],
#                                    config['output_class'])
#
#     def forward(self, batch_dict):
#         x = batch_dict['inputs'] # (b,1,4,512,512,3)(connected_car,batch,cam_num,H,W,C)
#         b, l, m, _, _, _ = x.shape
#
#         # shape: (B, max_cav, 4, 4)
#         transformation_matrix = batch_dict['transformation_matrix'] # (1,5,4,4)
#         record_len = batch_dict['record_len']
#
#         x = self.encoder(x)
#         batch_dict.update({'features': x})
#         x = self.fax(batch_dict)
#
#         # B*L, C, H, W
#         x = x.squeeze(1)
#
#         # compressor
#         if self.compression:
#             x = self.naive_compressor(x)
#
#         # Reformat to (B, max_cav, C, H, W)
#         x, mask = regroup(x, record_len, self.max_cav)
#         # perform feature spatial transformation,  B, max_cav, H, W, C
#         x = self.sttf(x, transformation_matrix) # 这个模块利用了坐标转换矩阵,实现将BEV特征从世界坐标系转換到自身坐标系,应该是为了后续在自身坐标系下分析处理BEV特征
#         com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
#             3) if not self.use_roi_mask \
#             else get_roi_and_cav_mask(x.shape,
#                                       mask,
#                                       transformation_matrix,
#                                       self.discrete_ratio,
#                                       self.downsample_rate)
#
#         # fuse all agents together to get a single bev map, b h w c
#         x = rearrange(x, 'b l h w c -> b l c h w')
#         x = self.fusion_net(x, com_mask)
#         x = x.unsqueeze(1)
#
#         # dynamic head
#         x = self.decoder(x)
#         x = rearrange(x, 'b l c h w -> (b l) c h w')
#         b = x.shape[0]
#         output_dict = self.seg_head(x, b, 1)
#
#         return output_dict

#===================================================魔改========================================================
# class SwinLSTMCell(nn.Module):
#
#     def __init__(self, dim, input_resolution, num_heads, window_size, depth,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm):
#         super(SwinLSTMCell, self).__init__()
#
#
#
#     def forward(self, xt, hidden_states):
#         if hidden_states is None:
#             B, L, C = xt.shape
#             hx = torch.zeros(B, L, C).to(xt.device)
#             cx = torch.zeros(B, L, C).to(xt.device)
#
#         else:
#             hx, cx = hidden_states
#
#         Ft = self.Swin(xt, hx)
#
#         gate = torch.sigmoid(Ft)
#         cell = torch.tanh(Ft)
#
#         cy = gate * (cx + cell)
#         hy = gate * torch.tanh(cy)
#         hx = hy
#         cx = cy
#
#         return hx, (hx, cx)
#
# class SpatialAttention_mtf(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention_mtf, self).__init__()
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.Tan = nn.Tanh()
#
#     def forward(self, curr, prev):
#         curr_avg_out = torch.mean(curr, dim=1, keepdim=True)
#         curr_max_out, _ = torch.max(curr, dim=1, keepdim=True)
#         curr_merge = torch.cat([curr_avg_out, curr_max_out], dim=1)
#         prev_avg_out = torch.mean(prev, dim=1, keepdim=True)
#         prev_max_out, _ = torch.max(prev, dim=1, keepdim=True)
#         prev_merge = torch.cat([prev_avg_out, prev_max_out], dim=1)
#         # prev_avg_out = torch.mean(torch.sum(prev, dim=0, keepdim=True), dim=1, keepdim=True)
#         # prev_max_out, _ = torch.max(torch.sum(prev, dim=0, keepdim=True), dim=1, keepdim=True)
#         # prev_merge = torch.cat([prev_avg_out, prev_max_out], dim=1)
#         merge = self.sigmoid(self.conv1(curr_merge + prev_merge))
#         final_out = (1 - merge) * self.Tan(curr) + merge * prev
#
#         return final_out
#
#
# class SyncLSTM(nn.Module):
#     def __init__(self, channel_size=128, spatial_size=32, compressed_size=64, height=32, width=32):
#         super(SyncLSTM, self).__init__()
#         self.spatial_size = spatial_size
#         self.channel_size = channel_size
#         self.compressed_size = compressed_size
#         self.lstmcell = MotionLSTM(32, self.compressed_size, height=height, width=width)
#         self.init_c = nn.parameter.Parameter(torch.rand(self.compressed_size, height, width))
#
#         self.ratio = int(math.sqrt(channel_size / compressed_size))
#         self.conv_pre_1 = nn.Conv2d(self.channel_size, self.ratio * self.compressed_size, kernel_size=3, stride=1,
#                                     padding=1)
#         self.conv_pre_2 = nn.Conv2d(self.ratio * self.compressed_size, self.compressed_size, kernel_size=3, stride=1,
#                                     padding=1)
#         self.bn_pre_1 = nn.BatchNorm2d(self.ratio * self.compressed_size)
#         self.bn_pre_2 = nn.BatchNorm2d(self.compressed_size)
#         self.conv_after_1 = nn.Conv2d(self.compressed_size, self.ratio * self.compressed_size, kernel_size=3, stride=1,
#                                       padding=1)
#         self.conv_after_2 = nn.Conv2d(self.ratio * self.compressed_size, self.channel_size, kernel_size=3, stride=1,
#                                       padding=1)
#         self.bn_after_1 = nn.BatchNorm2d(self.ratio * self.compressed_size)
#         self.bn_after_2 = nn.BatchNorm2d(self.channel_size)
#
#     def forward(self, x_raw):
#         frame_num, C, H, W = x_raw.shape
#         if self.compressed_size != self.channel_size:
#             x = F.relu(self.bn_pre_1(self.conv_pre_1(x_raw.view(-1, C, H, W))))
#             x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))
#             x = x.view(frame_num, self.compressed_size, H, W)
#         else:
#             x = x_raw
#
#         h = x[-1, :].unsqueeze(0)
#         c = self.init_c
#         for i in range(frame_num - 2, -1, -1):
#             h, c = self.lstmcell(x[i, :].unsqueeze(0), (h, c))
#         res = h
#         if self.compressed_size != self.channel_size:
#             res = F.relu(self.bn_after_1(self.conv_after_1(res)))
#             res = F.relu(self.bn_after_2(self.conv_after_2(res)))
#         else:
#             res = res
#         return res
#
#
# class MotionLSTM(nn.Module):
#     def __init__(self, spatial_size, input_channel_size, hidden_size=0, height=32, width=32):
#         super().__init__()
#         self.input_channel_size = input_channel_size
#         self.hidden_size = hidden_size
#         self.spatial_size = spatial_size
#
#         self.U_i = STPN_MotionLSTM(height_feat_size=self.input_channel_size)
#         self.V_i = STPN_MotionLSTM(height_feat_size=self.input_channel_size)
#         self.b_i = nn.Parameter(torch.Tensor(1, self.input_channel_size, height, width))
#
#         self.U_f = STPN_MotionLSTM(height_feat_size=self.input_channel_size)
#         self.V_f = STPN_MotionLSTM(height_feat_size=self.input_channel_size)
#         self.b_f = nn.Parameter(torch.Tensor(1, self.input_channel_size, height, width))
#
#         self.U_c = STPN_MotionLSTM(height_feat_size=self.input_channel_size)
#         self.V_c = STPN_MotionLSTM(height_feat_size=self.input_channel_size)
#         self.b_c = nn.Parameter(torch.Tensor(1, self.input_channel_size, height, width))
#
#         self.U_o = STPN_MotionLSTM(height_feat_size=self.input_channel_size)
#         self.V_o = STPN_MotionLSTM(height_feat_size=self.input_channel_size)
#         self.b_o = nn.Parameter(torch.Tensor(1, self.input_channel_size, height, width))
#
#     def forward(self, x, init_states=None):
#         h, c = init_states
#         i = torch.sigmoid(self.U_i(x) + self.V_i(h) + self.b_i)
#         f = torch.sigmoid(self.U_f(x) + self.V_f(h) + self.b_f)
#         g = torch.tanh(self.U_c(x) + self.V_c(h) + self.b_c)
#         o = torch.sigmoid(self.U_o(x) + self.V_o(x) + self.b_o)
#         c_out = f * c + i * g
#         h_out = o * torch.tanh(c_out)
#
#         return (h_out, c_out)
#
# class STPN_MotionLSTM(nn.Module):
#     def __init__(self, height_feat_size = 16):
#         super(STPN_MotionLSTM, self).__init__()
#
#         self.conv1_1 = nn.Conv2d(height_feat_size, 2*height_feat_size, kernel_size=3, stride=2, padding=1)
#         self.conv1_2 = nn.Conv2d(2*height_feat_size, 2*height_feat_size, kernel_size=3, stride=1, padding=1)
#
#         self.conv2_1 = nn.Conv2d(2*height_feat_size, 4*height_feat_size, kernel_size=3, stride=2, padding=1)
#         self.conv2_2 = nn.Conv2d(4*height_feat_size, 4*height_feat_size, kernel_size=3, stride=1, padding=1)
#
#         self.conv7_1 = nn.Conv2d(6*height_feat_size, 2*height_feat_size, kernel_size=3, stride=1, padding=1)
#         self.conv7_2 = nn.Conv2d(2*height_feat_size, 2*height_feat_size, kernel_size=3, stride=1, padding=1)
#
#         self.conv8_1 = nn.Conv2d(3*height_feat_size , height_feat_size, kernel_size=3, stride=1, padding=1)
#         self.conv8_2 = nn.Conv2d(height_feat_size, height_feat_size, kernel_size=3, stride=1, padding=1)
#
#         self.bn1_1 = nn.BatchNorm2d(2*height_feat_size)
#         self.bn1_2 = nn.BatchNorm2d(2*height_feat_size)
#
#         self.bn2_1 = nn.BatchNorm2d(4*height_feat_size)
#         self.bn2_2 = nn.BatchNorm2d(4*height_feat_size)
#
#         self.bn7_1 = nn.BatchNorm2d(2*height_feat_size)
#         self.bn7_2 = nn.BatchNorm2d(2*height_feat_size)
#
#         self.bn8_1 = nn.BatchNorm2d(1*height_feat_size)
#         self.bn8_2 = nn.BatchNorm2d(1*height_feat_size)
#
#     def forward(self, x):
#         x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
#         x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
#         x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
#         x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
#         x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))
#         x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_2, scale_factor=(2, 2)), x_1), dim=1))))
#         x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))
#         x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
#         res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))
#
#         return res_x
#
# class TemporalFusion_lstm(nn.Module):
#     def __init__(self):
#         super(TemporalFusion_lstm, self).__init__()
#
#         self.mtf_attention = SpatialAttention_mtf()
#         self.sync_lstm = SyncLSTM(channel_size=128, height=32, width=32)
#
#     def forward(self, origin_input):
#         x_cav = origin_input[0][1:,:] # 当前时刻其他车数据
#         x_curr_test = origin_input[0][0:1,:]
#         x_curr = origin_input[0][0:1,:] # 当前时刻自车数据
#         x_prev = origin_input[1]
#
#         x_prev_cat = self.mtf_attention(x_curr, x_prev)
#         x_raw = torch.cat([x_curr, x_prev_cat], dim=0)
#
#         x_fuse = self.sync_lstm(x_raw)
#
#         x_fuse = torch.cat([x_fuse, x_cav], dim=0)
#
#         return x_fuse
#
# class STTF(nn.Module):
#     def __init__(self, args):
#         super(STTF, self).__init__()
#         self.discrete_ratio = args['resolution']
#         self.downsample_rate = args['downsample_rate']
#
#     def forward(self, x, spatial_correction_matrix):
#         """
#         Transform the bev features to ego space.
#
#         Parameters
#         ----------
#         x : torch.Tensor
#             B L C H W
#         spatial_correction_matrix : torch.Tensor
#             Transformation matrix to ego
#
#         Returns
#         -------
#         The bev feature same shape as x but with transformation
#         """
#         dist_correction_matrix = get_discretized_transformation_matrix(
#             spatial_correction_matrix, self.discrete_ratio,
#             self.downsample_rate)
#
#         # transpose and flip to make the transformation correct
#         x = rearrange(x, 'b l c h w  -> b l c w h')
#         x = torch.flip(x, dims=(4,))
#         # Only compensate non-ego vehicles
#         B, L, C, H, W = x.shape
#
#         T = get_transformation_matrix(
#             dist_correction_matrix[:, :, :, :].reshape(-1, 2, 3), (H, W))
#         cav_features = warp_affine(x[:, :, :, :, :].reshape(-1, C, H, W), T,
#                                    (H, W))
#         cav_features = cav_features.reshape(B, -1, C, H, W)
#
#         # flip and transpose back
#         x = cav_features
#         x = torch.flip(x, dims=(4,))
#         x = rearrange(x, 'b l c w h -> b l h w c')
#
#         return x
#
# class CorpBEVT(nn.Module):
#     def __init__(self, config):
#         super(CorpBEVT, self).__init__()
#         self.max_cav = config['max_cav']
#         # encoder params
#         self.encoder = ResnetEncoder(config['encoder'])
#
#         # cvm params
#         fax_params = config['fax']
#         fax_params['backbone_output_shape'] = self.encoder.output_shapes
#         self.fax = FAXModule(fax_params)
#
#         self.temporal_fusion = TemporalFusion_lstm()
#
#         if config['compression'] > 0:
#             self.compression = True
#             self.naive_compressor = NaiveCompressor(128, config['compression'])
#         else:
#             self.compression = False
#
#         # spatial feature transform module
#         self.downsample_rate = config['sttf']['downsample_rate']
#         self.discrete_ratio = config['sttf']['resolution']
#         self.use_roi_mask = config['sttf']['use_roi_mask']
#         self.sttf = STTF(config['sttf'])
#
#         # spatial fusion
#         self.fusion_net = SwapFusionEncoder(config['fax_fusion'])
#
#         # decoder params
#         decoder_params = config['decoder']
#         # decoder for dynamic and static differet
#         self.decoder = NaiveDecoder(decoder_params)
#
#         self.target = config['target']
#         self.seg_head = BevSegHead(self.target,
#                                    config['seg_head_dim'],
#                                    config['output_class'])
#
#
#     def forward(self, batch_dict_list):
#         # x, transformation_matrix, record_len= self.transform_feature(batch_dict)
#         single_bev = []
#         record_len = []
#         transformation_matrix = []
#         # record_len = batch_dict_list[0]['record_len']
#         # transformation_matrix = batch_dict_list[0]['transformation_matrix']
#
#         for i in range(len(batch_dict_list)):
#             for id, batch_dict in batch_dict_list[i].items():
#                 x = batch_dict['inputs'] # (b,1,4,512,512,3)(connected_car,batch,cam_num,H,W,C)
#                 b, t, m, _, _, _ = x.shape
#
#                 # shape: (B, max_cav, 4, 4)
#                 record_len.append(batch_dict['record_len'])
#                 transformation_matrix.append(batch_dict['transformation_matrix'])
#
#                 x = self.encoder(x)
#                 batch_dict.update({'features': x})
#                 x = self.fax(batch_dict)
#                 single_bev.append(rearrange(x,'b t c h w -> (b t) c h w'))
#         x = self.temporal_fusion(single_bev)
#         record_len = record_len[0]
#         transformation_matrix = transformation_matrix[0]
#
#         # B*L, C, H, W
#         x = x.squeeze(1)
#
#         # compressor
#         if self.compression:
#             x = self.naive_compressor(x)
#
#         # Reformat to (B, max_cav, C, H, W)
#         x, mask = regroup(x, record_len, self.max_cav)
#         # perform feature spatial transformation,  B, max_cav, H, W, C
#         x = self.sttf(x, transformation_matrix) # 这个模块利用了坐标转换矩阵,实现将BEV特征从世界坐标系转換到自身坐标系,应该是为了后续在自身坐标系下分析处理BEV特征
#         com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
#             3) if not self.use_roi_mask \
#             else get_roi_and_cav_mask(x.shape,
#                                       mask,
#                                       transformation_matrix,
#                                       self.discrete_ratio,
#                                       self.downsample_rate)
#
#         # fuse all agents together to get a single bev map, b h w c
#         x = rearrange(x, 'b l h w c -> b l c h w')
#         x = self.fusion_net(x, com_mask)
#         x = x.unsqueeze(1)
#
#         # dynamic head
#         x = self.decoder(x)
#         x = rearrange(x, 'b l c h w -> (b l) c h w')
#         b = x.shape[0]
#         output_dict = self.seg_head(x, b, 1)
#
#         return output_dict

# =============================== v4 ==================================
# class STTF(nn.Module):
#     def __init__(self, args):
#         super(STTF, self).__init__()
#         self.discrete_ratio = args['resolution']
#         self.downsample_rate = args['downsample_rate']
#
#     def forward(self, x, spatial_correction_matrix):
#         """
#         Transform the bev features to ego space.
#
#         Parameters
#         ----------
#         x : torch.Tensor
#             B L C H W
#         spatial_correction_matrix : torch.Tensor
#             Transformation matrix to ego
#
#         Returns
#         -------
#         The bev feature same shape as x but with transformation
#         """
#         dist_correction_matrix = get_discretized_transformation_matrix(
#             spatial_correction_matrix, self.discrete_ratio,
#             self.downsample_rate)
#
#         # transpose and flip to make the transformation correct
#         x = rearrange(x, 'b l c h w  -> b l c w h')
#         x = torch.flip(x, dims=(4,))
#         # Only compensate non-ego vehicles
#         B, L, C, H, W = x.shape
#
#         T = get_transformation_matrix(
#             dist_correction_matrix[:, :, :, :].reshape(-1, 2, 3), (H, W))
#         cav_features = warp_affine(x[:, :, :, :, :].reshape(-1, C, H, W), T,
#                                    (H, W))
#         cav_features = cav_features.reshape(B, -1, C, H, W)
#
#         # flip and transpose back
#         x = cav_features
#         x = torch.flip(x, dims=(4,))
#         x = rearrange(x, 'b l c w h -> b l h w c')
#
#         return x
#
# class CorpBEVT(nn.Module):
#     def __init__(self, config):
#         super(CorpBEVT, self).__init__()
#         self.max_cav = config['max_cav']
#         # encoder params
#         self.encoder = ResnetEncoder(config['encoder'])
#
#         # cvm params
#         fax_params = config['fax']
#         fax_params['backbone_output_shape'] = self.encoder.output_shapes
#         self.fax = FAXModule(fax_params)
#
#         # self.temporal_fusion = model_forward_single_layer(dim=128, num_heads=2, qkv_bias=True, qk_scale=None,
#         #          attn_drop=0., proj_drop=0., kernel_size=3, dilation=[[3, 3], [3, 2], [2, 1]], embed_dim=128)
#         self.temporal_fusion = MSTIA(in_channels=128, num_scales=3, num_heads=8, dropout=0.1, max_window_size=3, threshold=0.1)
#         if config['compression'] > 0:
#             self.compression = True
#             self.naive_compressor = NaiveCompressor(128, config['compression'])
#         else:
#             self.compression = False
#
#         # spatial feature transform module
#         self.downsample_rate = config['sttf']['downsample_rate']
#         self.discrete_ratio = config['sttf']['resolution']
#         self.use_roi_mask = config['sttf']['use_roi_mask']
#         self.sttf = STTF(config['sttf'])
#
#         # spatial fusion
#         self.fusion_net = SwapFusionEncoder(config['fax_fusion'])
#
#         # decoder params
#         decoder_params = config['decoder']
#         # decoder for dynamic and static differet
#         self.decoder = NaiveDecoder(decoder_params)
#
#         self.target = config['target']
#         self.seg_head = BevSegHead(self.target,
#                                    config['seg_head_dim'],
#                                    config['output_class'])
#
#     def align_tensors(self, tensor_list):
#         # 找出第一个维度的最大值
#         max_b = max(tensor.shape[0] for tensor in tensor_list)
#
#         # 创建一个新的列表来存储对齐后的张量
#         aligned_tensors = []
#
#         for tensor in tensor_list:
#             if tensor.shape[0] < max_b:
#                 # 如果当前张量的第一个维度小于最大值，我们需要进行填充
#                 padding = torch.zeros(max_b - tensor.shape[0], *tensor.shape[1:], dtype=tensor.dtype,
#                                       device=tensor.device)
#                 aligned_tensor = torch.cat([tensor, padding], dim=0)
#             else:
#                 # 如果当前张量的第一个维度已经是最大值，不需要改变
#                 aligned_tensor = tensor
#
#             aligned_tensors.append(aligned_tensor.unsqueeze(0))
#
#         return aligned_tensors
#
#     def forward(self, batch_dict_list):
#         x, transformation_matrix, record_len= self.transform_feature(batch_dict)
#         single_bev = []
#
#         # record_len = batch_dict_list[0]['record_len']
#         # transformation_matrix = batch_dict_list[0]['transformation_matrix']
#
#         for i in range(len(batch_dict_list)):
#             record_len = []
#             transformation_matrix = []
#             for id, batch_dict in batch_dict_list[i].items():
#                 x = batch_dict['inputs'] # (b,1,4,512,512,3)(connected_car,batch,cam_num,H,W,C)
#                 b, t, m, _, _, _ = x.shape
#
#                 # shape: (B, max_cav, 4, 4)
#                 record_len.append(batch_dict['record_len'])
#                 transformation_matrix.append(batch_dict['transformation_matrix'])
#
#                 x = self.encoder(x)
#                 batch_dict.update({'features': x})
#                 # x = self.fax(batch_dict)
#                 x = torch.rand([b, t, 128, 32, 32]).to(torch.device('cuda'))
#                 record_len = record_len[0]
#                 transformation_matrix = transformation_matrix[0]
#
#                 # B*L, C, H, W
#                 x = x.squeeze(1)
#
#                 # compressor
#                 if self.compression:
#                     x = self.naive_compressor(x)
#
#                 # Reformat to (B, max_cav, C, H, W)
#                 x, mask = regroup(x, record_len, self.max_cav)
#                 # perform feature spatial transformation,  B, max_cav, H, W, C
#                 x = self.sttf(x, transformation_matrix) # 这个模块利用了坐标转换矩阵,实现将BEV特征从世界坐标系转換到自身坐标系,应该是为了后续在自身坐标系下分析处理BEV特征
#                 com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
#                     3) if not self.use_roi_mask \
#                     else get_roi_and_cav_mask(x.shape,
#                                               mask,
#                                               transformation_matrix,
#                                               self.discrete_ratio,
#                                               self.downsample_rate)
#
#                 # fuse all agents together to get a single bev map, b h w c
#                 x = rearrange(x, 'b l h w c -> b l c h w')
#                 # x = self.fusion_net(x, com_mask)
#                 x = torch.randn([b, 128, 32, 32]).to(torch.device('cuda'))
#                 single_bev.append(x)
#         single_bev = self.align_tensors(single_bev)
#         single_bev = single_bev[::-1]    # [t,t-1,t-2] -> [t-2,t-1,t]
#         temp_features = torch.cat(single_bev, dim=0).permute(1,0,2,3,4)
#         temp_features = torch.rand([1, 3, 128, 32, 32])
#         x, selected_window, window_logits = self.temporal_fusion(temp_features)
#         x = x.unsqueeze(1)
#
#         # dynamic head
#         x = self.decoder(x)
#         x = rearrange(x, 'b l c h w -> (b l) c h w')
#         b = x.shape[0]
#         output_dict = self.seg_head(x, b, 1)
#
#         return output_dict, temp_features, selected_window, window_logits
# -*-coding:utf-8-*-

import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat, reduce
from opencood.models.backbones.resnet_ms import ResnetEncoder
import torch.nn.functional as F
class CrossWinAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, rel_pos_emb=False, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head
        self.rel_pos_emb = rel_pos_emb

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)

    def add_rel_pos_emb(self, x):
        return x

    def forward(self, q, k, v, skip=None):
        """
        q: (b n X Y W1 W2 d)
        k: (b n x y w1 w2 d)
        v: (b n x y w1 w2 d)
        return: (b X Y W1 W2 d)
        """
        assert k.shape == v.shape
        _, view_size, q_height, q_width, q_win_height, q_win_width, _ = q.shape
        _, _, kv_height, kv_width, _, _, _ = k.shape
        assert q_height * q_width == kv_height * kv_width

        # flattening
        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (X Y) (n W1 W2) (heads dim_head)
        k = self.to_k(k)                                # b (X Y) (n w1 w2) (heads dim_head)
        v = self.to_v(v)                                # b (X Y) (n w1 w2) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras计算相似度分数
        dot = self.scale * torch.einsum('b l Q d, b l K d -> b l Q K', q, k)  # b (X Y) (n W1 W2) (n w1 w2)
        # dot = rearrange(dot, 'b l n Q K -> b l Q (n K)')  # b (X Y) (W1 W2) (n w1 w2)

        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)  # b (X Y) (n W1 W2) d
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)
        a = rearrange(a, ' b (x y) (n w1 w2) d -> b n x y w1 w2 d',
            x=q_height, y=q_width, w1=q_win_height, w2=q_win_width)

        # Combine multiple heads
        z = self.proj(a)

        # reduce n: (b n X Y W1 W2 d) -> (b X Y W1 W2 d)
        z = z.mean(1)  # for sequential usage, we cannot reduce it!

        # Optional skip connection
        if skip is not None:
            z = z + skip
        return z
class HFAX_CA(nn.Module):
    def __init__(self, feat_height, feat_width, feat_dim, dim, scales, q_win_sizes, kv_win_sizes, heads, dim_head,
                 qkv_bias=True):
        super().__init__()
        self.scales = scales

        self.attns = nn.ModuleDict()
        for scale, q_win_size, kv_win_size, head, dim_h in zip(scales, q_win_sizes, kv_win_sizes, heads, dim_head):
            self.attns[scale] = CrossWinAttention(dim, head, dim_h, qkv_bias)
            self.register_buffer(f'{scale}_q_win_size', torch.tensor(q_win_size))
            self.register_buffer(f'{scale}_kv_win_size', torch.tensor(kv_win_size))

        self.projs = nn.ModuleDict({
            f'{scale_1}_{scale_2}': nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(2 * dim, dim)
            )
            for scale_1 in scales for scale_2 in scales if scale_1 != scale_2
        })

        self.prenorm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def pad_divisble(self, x, win_h, win_w):
        """Pad the feature map to be divisible by window size."""
        _, _, _, h, w = x.shape
        pad_h = (h + win_h - 1) // win_h * win_h - h
        pad_w = (w + win_w - 1) // win_w * win_w - w
        return F.pad(x, (0, pad_w, 0, pad_h), value=0)

    def forward(self, x_dict, bev_dict, img_feat_dict):
        out_dict = {}
        for scale in self.scales:
            # x_dict[scale]: b d H W -> b n H W
            q = repeat(x_dict[scale], 'b d H W -> b n H W', n=bev_dict[scale].shape[1])

            # Pad feature map to be divisible by window size
            q = self.pad_divisble(q, self.attns[f'{scale}_q_win_size'][0], self.attns[f'{scale}_q_win_size'][1])
            kv = self.pad_divisble(img_feat_dict[scale], self.attns[f'{scale}_kv_win_size'][0],
                                   self.attns[f'{scale}_kv_win_size'][1])

            # Perform local cross-attention
            out_dict[scale] = self.attns[scale](
                rearrange(q, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.attns[f'{scale}_q_win_size'][0],
                          w2=self.attns[f'{scale}_q_win_size'][1]),
                rearrange(kv, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.attns[f'{scale}_kv_win_size'][0],
                          w2=self.attns[f'{scale}_kv_win_size'][1]),
                rearrange(kv, 'b n d (x w1) (y w2) -> b n x y w1 w2 d', w1=self.attns[f'{scale}_kv_win_size'][0],
                          w2=self.attns[f'{scale}_kv_win_size'][1]),
                skip=x_dict[scale].unsqueeze(1)
            )
            out_dict[scale] = rearrange(out_dict[scale], 'b n x y w1 w2 d -> b n d (x w1) (y w2)',
                                        w1=self.attns[f'{scale}_q_win_size'][0],
                                        w2=self.attns[f'{scale}_q_win_size'][1])
            out_dict[scale] = out_dict[scale].mean(1)  # Average over n

        for scale_1 in self.scales:
            for scale_2 in self.scales:
                if scale_1 != scale_2:
                    out_dict[scale_1] = self.projs[f'{scale_1}_{scale_2}'](
                        torch.cat([out_dict[scale_1],
                                   F.interpolate(out_dict[scale_2], size=out_dict[scale_1].shape[-2:], mode='bilinear',
                                                 align_corners=True)], dim=1)
                    ) + out_dict[scale_1]

        for scale in self.scales:
            out_dict[scale] = out_dict[scale] + self.mlp(self.prenorm(out_dict[scale]))

        return {scale: rearrange(out_dict[scale], 'b d H W -> b H W d') for scale in self.scales}

class SinBEVTEncoder(nn.Module):
    def __init__(self, img_encoder, bev_encoder, hfax_ca):
        super().__init__()
        self.img_encoder = img_encoder
        self.bev_encoder = bev_encoder
        self.hfax_ca = hfax_ca

    def forward(self, img, I_inv, E_inv, bev_embed):
        img_feat_dict = img
        bev_dict = {'1_8':bev_embed,'1_4':bev_embed,'1_2':bev_embed}

        x_dict = {}
        for scale in self.hfax_ca.scales:
            x_dict[scale] = bev_dict[scale].mean(0)  # Average over n

        x_dict = self.hfax_ca(x_dict, bev_dict, img_feat_dict)

        return x_dict


class ImageEncoder(nn.Module):
    def __init__(self, backbone, neck, img_feat_height, img_feat_width, img_feat_dim):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.img_feat_height = img_feat_height
        self.img_feat_width = img_feat_width
        self.img_feat_dim = img_feat_dim

    def forward(self, x):
        # x: (B, N, C, H, W)
        B, N, C, H, W = x.shape
        x = x.reshape(B * N, C, H, W)

        # Backbone
        x = self.backbone(x)

        # Neck
        x = self.neck(x)

        _, C_, H_, W_ = x.shape
        assert H_ == self.img_feat_height and W_ == self.img_feat_width and C_ == self.img_feat_dim

        x = x.reshape(B, N, self.img_feat_dim, self.img_feat_height, self.img_feat_width)

        return x


class BEVEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, ds_layer_num):
        super().__init__()
        self.ds_layer_num = ds_layer_num

        self.layers = nn.ModuleList()
        for _ in range(ds_layer_num):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ))
            in_dim = out_dim

    def forward(self, x):
        # x: (B, C, H, W)
        out_dict = {}
        for i in range(self.ds_layer_num):
            x = self.layers[i](x)
            out_dict[f'1_{2 ** (i + 1)}'] = x
        return out_dict

if __name__ == '__main__':
    # Define hyperparameters
    config = {
        'encoder': {'num_layers': 34, 'pretrained': True, 'image_width': 512, 'image_height': 512, 'id_pick': [1, 2, 3]},
        'dim': [128, 128, 128],
        'middle': [2, 2, 2],
        'bev_embedding': {
            'sigma': 1.0,
            'bev_height': 256,
            'bev_width': 256,
            'h_meters': 100,
            'w_meters': 100,
            'offset': 0.0,
            'upsample_scales': [2, 4, 8]
        },
        'cross_view': {
            'image_height': 512,
            'image_width': 512,
            'no_image_features': False,
            'skip': True,
            'heads': [4, 4, 4],
            'dim_head': [32, 32, 32],
            'qkv_bias': True
        },
        'cross_view_swap': {
            'rel_pos_emb': False,
            'q_win_size': [[16, 16], [16, 16], [32, 32]],
            'feat_win_size': [[8, 8], [8, 8], [16, 16]],
            'bev_embedding_flag': [True, False, False]
        },
        'self_attn': {
            'dim_head': 32,
            'dropout': 0.1,
            'window_size': 32
        },
        'backbone_output_shape': [
            torch.Size([1, 1, 1, 128, 64, 64]),
            torch.Size([1, 1, 1, 256, 32, 32]),
            torch.Size([1, 1, 1, 512, 16, 16])
        ]
    }
    feat_height = feat_width = 32
    feat_dim = 256
    dim = 3
    scales = ['1_8', '1_4', '1_2']
    q_win_sizes = [(8, 8), (4, 4), (2, 2)]
    kv_win_sizes = [(4, 4), (2, 2), (1, 1)]
    heads = [4, 4, 4]
    dim_head = [64, 64, 64]

    # Create modules
    img_encoder = ResnetEncoder(config['encoder'])
    # img_encoder = ImageEncoder()
    bev_encoder = BEVEncoder(feat_height, feat_width, dim)
    hfax_ca = HFAX_CA(feat_height, feat_width, feat_dim, dim, scales, q_win_sizes, kv_win_sizes, heads, dim_head)

    image = torch.load('/home/why/YJQ/CoBEVT/opv2v/opencood/tools/临时变量/image')
    bev = torch.load('/home/why/YJQ/CoBEVT/opv2v/opencood/tools/临时变量/bev')
    # Create SinBEVT encoder
    sinbevt_encoder = SinBEVTEncoder(img_encoder, bev_encoder, hfax_ca)

    # Forward pass
    img = torch.rand(2, 5, 3, 256, 256) # batch_size, num_cams, channels, height, width
    I_inv = torch.rand(2, 5, 3, 3) # batch_size, num_cams, 3, 3
    E_inv = torch.rand(2, 5, 4, 4) # batch_size, num_cams, 4, 4
    bev_embed = torch.rand(2, 3, feat_height, feat_width) # batch_size, 3, height, width

    x_dict = sinbevt_encoder(image, I_inv, E_inv, bev)
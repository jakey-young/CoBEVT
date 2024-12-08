# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.registry import register_model
from opencood.tools.heat_map import visualize_attention_weights
from opencood.tools.heat_map import visualize_attention_pattern, visualize_attention_pattern_fused


def compute_cross_vehicle_similarity(x, sim_mlp):
    """
    计算跨车辆的特征相似度
    Args:
        x: (b, l, h, w, c) - batch, num_vehicles, height, width, channels
        sim_mlp: 相似度映射网络
    Returns:
        similarity: (b, num_heads, l*h*w, l*h*w)
    """
    # x = rearrange(x,'(b l) (h w) c -> b l h w c', b = 1, h=32)
    num_heads = 8
    b_l, hw, c = x.shape
    # 假设第一辆车(索引0)为自车
    b = 1  # 由于输入是stack后的批次
    l = b_l // b

    # 1. 重塑输入以分离车辆维度
    x = x.view(b, l, hw, c)  # (b, l, h*w, c)

    # 2. 提取自车特征
    ego_features = x[:, 0:1]  # (b, 1, h*w, c)

    # 3. 使用sim_mlp转换特征
    ego_features = sim_mlp(ego_features.view(b, hw, c))  # (b, h*w, num_heads)
    other_features = sim_mlp(x.view(b * l, hw, c))  # (b*l, h*w, num_heads)

    # 4. 重塑other_features以便计算
    other_features = other_features.view(b, l, hw, num_heads)

    # 5. 计算自车与所有车辆的相似度
    ego_features = ego_features.unsqueeze(1).expand(-1, l, -1, -1)  # (b, l, h*w, num_heads)
    similarity = torch.matmul(ego_features, other_features.transpose(-2, -1))  # (b, l, h*w, h*w)

    # 6. 调整维度顺序并复制到所有车辆
    similarity = similarity.view(b * l, 1, hw, hw)  # (b*l, 1, h*w, h*w)
    similarity = similarity.expand(-1, num_heads, -1, -1)  # (b*l, num_heads, h*w, h*w)
    return similarity



class RefinedFusionAttention(nn.Module):
    """
    在local attention和global attention之间增加的精细化融合模块
    """

    def __init__(self, dim, num_heads=8, head_dim=32, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        # 特征变换
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)

        # 空间感知编码
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, 32),  # 4 = (x1,y1,x2,y2) 相对位置
            nn.ReLU(),
            nn.Linear(32, num_heads)
        )

        # 特征相似度编码
        self.sim_mlp = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_heads)
        )

        # 输出映射
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        # Gating机制
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.idx = 0

    def get_relative_positions(self, h, w):
        """计算网格中所有位置对之间的相对位置"""
        coords = torch.stack(torch.meshgrid(
            torch.arange(h),
            torch.arange(w)
        )).flatten(1).transpose(0, 1)  # (h*w, 2)

        rel_pos = coords.unsqueeze(1) - coords.unsqueeze(0)  # (h*w, h*w, 2)
        # 加入距离信息
        dist = torch.norm(rel_pos.float(), dim=-1, keepdim=True)
        rel_pos = torch.cat([rel_pos, dist], dim=-1)  # (h*w, h*w, 3)
        return rel_pos

    def forward(self, x):
        """
        x: (b, l, h, w, c) - batch, num_agents, height, width, channels
        """
        self.idx = self.idx+1
        save_dir = '/home/why/workspace/heat_map/'

        b, l, h, w, c = x.shape

        # 重塑输入
        x = x.view(b * l, h * w, c)

        # 1. 计算Q,K,V
        q = self.to_q(x).view(b * l, h * w, self.num_heads, self.head_dim)
        k = self.to_k(x).view(b * l, h * w, self.num_heads, self.head_dim)
        v = self.to_v(x).view(b * l, h * w, self.num_heads, self.head_dim)

        # 转置以便进行注意力计算
        q = q.permute(0, 2, 1, 3)  # (b*l, heads, h*w, head_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 2. 计算attention scores
        attn_scores_o = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 3. 计算空间位置权重
        rel_pos = self.get_relative_positions(h, w).to(x.device)
        pos_weights = self.pos_mlp(rel_pos)  # (h*w, h*w, heads)
        pos_weights = pos_weights.permute(2, 0, 1).unsqueeze(0)  # (1, heads, h*w, h*w)


        # 4. 计算特征相似度权重
        feature_sim = torch.matmul(self.sim_mlp(x), self.sim_mlp(x).transpose(-2, -1))
        feature_sim = feature_sim.unsqueeze(1)
        feature_sim = feature_sim.expand(-1,self.num_heads,-1,-1)

        # feature_sim = compute_cross_vehicle_similarity(x, self.sim_mlp)

        # 5. 组合不同的attention权重
        attn_scores = attn_scores_o + pos_weights + feature_sim
        # attn_scores = attn_scores_o + feature_sim
        attn_probs = F.softmax(attn_scores, dim=-1)


        # visualize_attention_weights(self.idx,attn_scores, pos_weights, feature_sim, save_path=save_dir )
        # center_pos = (h//2, w//2)
        # visualize_attention_pattern(self.idx,
        #     attn_scores_o, pos_weights, feature_sim,
        #     center_pos, h, w,
        #     save_path=save_dir
        # )

        visualize_attention_pattern_fused(self.idx,
            attn_scores_o,
            pos_weights,
            feature_sim,
            query_pos=(16, 16),  # 图像中心位置
            h=32, w=32,
            batch_idx=0,
            save_path=save_dir,
            normalize=False  # 设置为False可以先看到原始分数
        )

        # 6. 计算输出
        out = torch.matmul(attn_probs, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(b * l, h * w, -1)
        out = self.to_out(out)

        # 7. 使用gating机制控制特征更新
        gate_weights = self.gate(torch.cat([x, out], dim=-1))
        out = gate_weights * out + (1 - gate_weights) * x

        # 恢复原始形状
        out = out.view(b, l, h, w, c)
        return out



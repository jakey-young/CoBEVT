# -*-coding:utf-8-*-

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import numpy as np
def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor



def visualize_attention_weights(idx, content_attn, spatial_attn, similarity_attn, save_path='/home/why/workspace/heat_map'):
    head_idx = 0
    content_attn = to_numpy(content_attn)
    spatial_attn = to_numpy(spatial_attn)
    similarity_attn = to_numpy(similarity_attn)

    content_map = content_attn[0, head_idx]  # (h*w, h*w)
    spatial_map = spatial_attn[0, head_idx]  # (h*w, h*w)
    similarity_map = similarity_attn[0, head_idx]  # (h*w, h*w)

    # 创建图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 设置colormap和归一化
    vmin = min(content_map.min(), spatial_map.min(), similarity_map.min())
    vmax = max(content_map.max(), spatial_map.max(), similarity_map.max())

    # 1. 内容注意力热力图
    sns.heatmap(content_map, ax=axes[0], cmap='YlOrRd',
                vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Content Attention\nHead {head_idx}')

    # 2. 空间注意力热力图
    sns.heatmap(spatial_map, ax=axes[1], cmap='YlGnBu',
                vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Spatial Attention\nHead {head_idx}')

    # 3. 相似度注意力热力图
    sns.heatmap(similarity_map, ax=axes[2], cmap='RdPu',
                vmin=vmin, vmax=vmax)
    axes[2].set_title(f'Similarity Attention\nHead {head_idx}')

    # 4. 组合注意力热力图
    combined_map = content_map + spatial_map + similarity_map
    sns.heatmap(combined_map, ax=axes[3], cmap='viridis')
    axes[3].set_title(f'Combined Attention\nHead {head_idx}')

    # 添加总标题
    plt.suptitle(f'Attention Weights Visualization (Batch {head_idx})', y=1.05)

    plt.tight_layout()

    idx = str(idx)
    base_dir = save_path
    base_name = 'attention'
    filename = f"{base_name}{idx}{'.png'}"
    if save_path:
        # plt.savefig(save_path, bbox_inches='tight', dpi=300)
        save_path = os.path.join(base_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    # plt.show()


def visualize_attention_pattern(idx_save, content_attn, spatial_attn, similarity_attn,
                                query_pos, h, w, batch_idx=0, head_idx=4, save_path=None):
    """
    可视化特定查询位置的注意力模式
    Args:
        query_pos: (x, y) 查询位置的坐标
        h, w: 特征图的高度和宽度
    """

    # 转换为numpy
    content_attn = content_attn.detach().cpu().numpy()
    spatial_attn = spatial_attn.detach().cpu().numpy()
    similarity_attn = similarity_attn.detach().cpu().numpy()

    # 计算一维索引
    query_idx = query_pos[0] * w + query_pos[1]

    # 获取该位置的注意力分布
    # content_pattern = content_attn[batch_idx, head_idx, query_idx].reshape(h, w)
    # spatial_pattern = spatial_attn[batch_idx, head_idx, query_idx].reshape(h, w)
    # similarity_pattern = similarity_attn[batch_idx, head_idx, query_idx].reshape(h, w)
    # content_pattern = np.mean(content_attn[batch_idx, :, query_idx], axis=0).reshape(h, w)
    # spatial_pattern = np.mean(spatial_attn[batch_idx, :, query_idx], axis=0).reshape(h, w)
    # similarity_pattern = np.mean(similarity_attn[batch_idx, :, query_idx], axis=0).reshape(h, w)
    content_pattern = np.mean(content_attn[0, :, query_idx], axis=0).reshape(h, w)
    spatial_pattern = np.mean(content_attn[1, :, query_idx], axis=0).reshape(h, w)
    similarity_pattern = np.mean(content_attn[2, :, query_idx], axis=0).reshape(h, w)
    # 创建图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 设置colormap和归一化
    vmin = min(content_pattern.min(), spatial_pattern.min(), similarity_pattern.min())
    vmax = max(content_pattern.max(), spatial_pattern.max(), similarity_pattern.max())

    # 绘制热力图
    maps = [content_pattern, spatial_pattern, similarity_pattern]
    titles = ['Content', 'Spatial', 'Similarity']
    cmaps = ['YlOrRd', 'YlGnBu', 'RdPu']

    for idx, (attn_map, title, cmap) in enumerate(zip(maps, titles, cmaps)):
        im = axes[idx].imshow(attn_map, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[idx].set_title(f'{title} Attention\nQuery Position: {query_pos}')
        plt.colorbar(im, ax=axes[idx])

        # 标记查询位置
        axes[idx].plot(query_pos[1], query_pos[0], 'r*', markersize=10)

    # 绘制组合注意力
    combined_pattern = sum(maps)
    im = axes[3].imshow(combined_pattern, cmap='viridis')
    axes[3].set_title(f'Combined Attention\nQuery Position: {query_pos}')
    plt.colorbar(im, ax=axes[3])
    # axes[3].plot(query_pos[1], query_pos[0], 'r*', markersize=10)

    plt.suptitle(f'Attention Pattern Visualization\nBatch {batch_idx}, Head {head_idx}', y=1.05)
    plt.tight_layout()

    if save_path:
        idx_save = str(idx_save)
        base_dir = save_path
        base_name = 'center'
        filename = f"{base_name}{idx_save}{'.png'}"
        if save_path:
            # plt.savefig(save_path, bbox_inches='tight', dpi=300)
            save_path = os.path.join(base_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    # plt.show()


def visualize_attention_pattern_fused(idx_save,content_attn, spatial_attn, similarity_attn,
                                      query_pos, h, w, batch_idx=0, save_path=None, normalize=True):
    """
    可视化特定查询位置的注意力模式，同时显示原始分数和归一化后的权重
    Args:
        normalize: 是否在第一次显示时就使用softmax归一化
    """
    # 转换为numpy
    content_attn = content_attn.detach().cpu().numpy()
    spatial_attn = spatial_attn.detach().cpu().numpy()
    similarity_attn = similarity_attn.detach().cpu().numpy()

    # 计算一维索引
    query_idx = query_pos[0] * w + query_pos[1]

    # 融合多头注意力
    content_pattern_ego = np.mean(content_attn[batch_idx, :, query_idx], axis=0).reshape(h, w)
    # content_pattern_cav1 = np.mean(content_attn[batch_idx+1, :, query_idx], axis=0).reshape(h, w)
    spatial_pattern_ego = np.mean(spatial_attn[batch_idx, :, query_idx], axis=0).reshape(h, w)
    similarity_pattern_ego = np.mean(similarity_attn[batch_idx, :, query_idx], axis=0).reshape(h, w)
    similarity_pattern_cav1 = np.mean(similarity_attn[batch_idx+1, :, query_idx], axis=0).reshape(h, w)

    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    # 创建图表：2行4列
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))

    # 准备数据
    patterns = [content_pattern_ego, spatial_pattern_ego, similarity_pattern_ego, similarity_pattern_cav1]
    titles = ['Content_ego', 'Spatial_ego', 'Similarity_ego','similarity_pattern_cav1']
    # titles = ['Similarity_0', 'Similarity_1', 'Similarity_2']

    # 第一行：显示原始分数
    for idx, (pattern, title) in enumerate(zip(patterns, titles)):
        if normalize:
            pattern = softmax(pattern)

        im = axes[0, idx].imshow(pattern, cmap='RdBu_r')
        axes[0, idx].set_title(f'Raw {title} Pattern')
        # axes[0, idx].plot(query_pos[1], query_pos[0], 'r*', markersize=10)
        plt.colorbar(im, ax=axes[0, idx])

        # 打印统计信息
        # print(f"\nRaw {title} Pattern Stats:")
        # print(f"Min: {pattern.min():.3f}, Max: {pattern.max():.3f}")
        # print(f"Mean: {pattern.mean():.3f}, Std: {pattern.std():.3f}")

    # 显示原始组合模式
    combined_raw = sum(patterns)
    im = axes[0, 4].imshow(combined_raw, cmap='RdBu_r')
    axes[0, 4].set_title('Raw Combined Pattern')
    # axes[0, 4].plot(query_pos[1], query_pos[0], 'r*', markersize=10)
    plt.colorbar(im, ax=axes[0, 4])





    # 第二行：显示归一化后的权重
    normalized_patterns = [softmax(p) for p in patterns]

    for idx, (pattern, title) in enumerate(zip(normalized_patterns, titles)):
        im = axes[1, idx].imshow(pattern, cmap='viridis')
        axes[1, idx].set_title(f'Normalized {title} Pattern')
        # axes[1, idx].plot(query_pos[1], query_pos[0], 'r*', markersize=10)
        plt.colorbar(im, ax=axes[1, idx])

        # 打印归一化后的统计信息
        # print(f"\nNormalized {title} Pattern Stats:")
        # print(f"Min: {pattern.min():.3f}, Max: {pattern.max():.3f}")
        # print(f"Mean: {pattern.mean():.3f}, Std: {pattern.std():.3f}")

    # 显示归一化后的组合模式
    combined_norm = softmax(combined_raw)
    im = axes[1, 4].imshow(combined_norm, cmap='viridis')
    axes[1, 4].set_title('Normalized Combined Pattern')
    # axes[1, 4].plot(query_pos[1], query_pos[0], 'r*', markersize=10)
    plt.colorbar(im, ax=axes[1, 4])

    plt.suptitle(
        f'Attention Pattern Visualization (Batch {batch_idx}, Query Pos {query_pos})\n'
        'Top: Raw Scores, Bottom: Normalized Weights',
        y=1.02
    )

    plt.tight_layout()

    idx = str(idx_save)
    base_dir = save_path
    base_name = 'attention'
    filename = f"{base_name}{idx}{'.png'}"
    if save_path:
        # plt.savefig(save_path, bbox_inches='tight', dpi=300)
        save_path = os.path.join(base_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    # plt.show()

    # 输出注意力分布的一些关键特征
    # def analyze_pattern(pattern, name):
    #     """分析注意力模式的特征"""
    #     # 找到最大注意力的位置
    #     max_pos = np.unravel_index(np.argmax(pattern), pattern.shape)
    #     # 计算与查询位置的距离
    #     distance = np.sqrt((max_pos[0] - query_pos[0]) ** 2 + (max_pos[1] - query_pos[1]) ** 2)
    #
    #     print(f"\n{name} Pattern Analysis:")
    #     print(f"Max attention position: {max_pos}")
    #     print(f"Distance to query position: {distance:.2f}")
    #     print(f"Attention at query position: {pattern[query_pos[0], query_pos[1]]:.3f}")
    #
    #     # 计算注意力的空间分布
    #     attention_mass = np.sum(pattern > np.mean(pattern))
    #     print(f"Number of high-attention points: {attention_mass}")
    #
    # print("\nPattern Analysis:")
    # for pattern, name in zip(normalized_patterns, titles):
    #     analyze_pattern(pattern, name)
    # analyze_pattern(combined_norm, "Combined")
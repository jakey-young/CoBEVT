# -*-coding:utf-8-*-
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def generate_center_weighted_heatmap(h=32, w=32):
    # 生成网格坐标
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)

    # 计算到中心的距离
    center_x, center_y = 0, 0
    dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

    # 生成高斯衰减的权重
    sigma = 0.15 # 控制衰减速度
    weights = np.exp(-dist ** 2 / (2 * sigma ** 2))

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(weights, cmap='RdBu_r')
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    plt.axis('off')
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Position Encoding Weights')
    plt.savefig('/home/why/workspace/heat_map/pos_emb.svg',format='svg',bbox_inches='tight',dpi=300)
    plt.show()

    return weights


# 生成另一个版本的热力图
def generate_modified_heatmap(h=32, w=32):
    # 生成网格坐标
    x = np.linspace(-2, 2, w)
    y = np.linspace(-2, 2, h)
    xx, yy = np.meshgrid(x, y)

    # 计算复合权重
    dist = np.sqrt(xx ** 2 + yy ** 2)
    weights = 1 / (1 + dist ** 2)  # 使用平方反比衰减

    # 归一化权重到[0,1]
    weights = (weights - weights.min()) / (weights.max() - weights.min())

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, cmap='viridis')
    plt.title('Modified Position Encoding Weights')
    plt.show()

    return weights

if __name__ == "__main__":
# 调用函数生成两种热力图
    weights1 = generate_center_weighted_heatmap()
#     weights2 = generate_modified_heatmap()
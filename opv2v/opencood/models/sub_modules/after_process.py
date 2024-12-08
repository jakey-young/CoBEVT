import numpy as np
from scipy import ndimage
import torch


def optimize_vehicle_segmentation(dynamic_map):
    """
    优化车辆分割图，分离相邻车辆

    参数:
    dynamic_map: torch.Tensor, shape=(1, 256, 256)
                二值分割图，1表示车辆，0表示背景

    返回:
    torch.Tensor: 优化后的分割图，shape=(1, 256, 256)
    """
    # 转换输入tensor为numpy数组
    seg_map = dynamic_map.squeeze(0).cpu().numpy()

    # 确保输入是二值图像
    seg_map = (seg_map > 0.5).astype(np.uint8)

    # 找到所有连通区域
    labeled_array, num_features = ndimage.label(seg_map)

    # 创建输出图像
    result = np.zeros_like(seg_map, dtype=np.uint8)

    # 车辆尺寸范围
    MIN_LENGTH, MAX_LENGTH = 9, 14
    MIN_WIDTH, MAX_WIDTH = 5, 6

    # 处理每个连通区域
    for label in range(1, num_features + 1):
        # 获取当前区域
        region_mask = (labeled_array == label)

        # 计算区域的边界框
        rows, cols = np.where(region_mask)
        if len(rows) == 0:
            continue

        height = rows.max() - rows.min() + 1
        width = cols.max() - cols.min() + 1

        # 如果区域尺寸在单个车辆范围内，直接保留
        if (MIN_LENGTH <= height <= MAX_LENGTH and
                MIN_WIDTH <= width <= MAX_WIDTH):
            result[region_mask] = 1
            continue

        # 对于过大的区域，进行分离处理
        # 使用距离变换找到车辆中心点
        dist_transform = ndimage.distance_transform_edt(region_mask)

        # 获取局部最大值作为车辆中心
        local_max = (dist_transform > 0.7 * dist_transform.max())

        # 使用膨胀和腐蚀操作分离车辆
        separated_vehicles = ndimage.binary_opening(region_mask,
                                                    structure=np.ones((MIN_WIDTH, MIN_WIDTH)))

        # 将分离后的区域添加到结果中
        result |= separated_vehicles.astype(np.uint8)

    # 转换回PyTorch tensor并恢复维度
    result_tensor = torch.from_numpy(result).float().unsqueeze(0)

    return result_tensor
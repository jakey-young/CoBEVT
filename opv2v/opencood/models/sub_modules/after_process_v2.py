import numpy as np
from scipy import ndimage
import torch
from skimage import measure
import cv2


def optimize_vehicle_segmentation(dynamic_map):
    """
    优化车辆分割图，使用固定尺寸(9,5)分离相邻车辆

    参数:
    dynamic_map: torch.Tensor, shape=(1, 256, 256)
                二值分割图，1表示车辆，0表示背景

    返回:
    torch.Tensor: 优化后的分割图，shape=(1, 256, 256)
    """
    # 转换输入tensor为numpy数组
    seg_map = dynamic_map.squeeze(0).cpu().numpy()

    # 确保输入是二值图像uint8类型
    seg_map = (seg_map > 0.5).astype(np.uint8)

    # 找到所有连通区域
    labeled_array, num_features = ndimage.label(seg_map)

    # 获取每个连通区域的属性
    regions = measure.regionprops(labeled_array)

    # 创建输出图像（确保是uint8类型）
    result = np.zeros_like(seg_map, dtype=np.uint8)

    # 固定车辆尺寸
    VEHICLE_LENGTH = 9
    VEHICLE_WIDTH = 5
    VEHICLE_AREA = VEHICLE_LENGTH * VEHICLE_WIDTH

    def estimate_vehicle_count(area):
        """估计区域内车辆数量"""
        return max(1, round(area / VEHICLE_AREA))

    def get_region_orientation(region_mask):
        """计算区域的主方向"""
        rows, cols = np.where(region_mask)
        if len(rows) < 2:  # 防止只有一个点的情况
            return 0

        # 计算协方差矩阵的主方向
        cov = np.cov(np.vstack([rows, cols]))
        eigenvals, eigenvects = np.linalg.eig(cov)
        # 确保返回的是实数
        angle = np.arctan2(eigenvects[0, 0].real, eigenvects[1, 0].real)
        return angle

    def create_vehicle_mask(center, angle, shape):
        """创建单个车辆的掩码"""
        mask = np.zeros(shape, dtype=np.uint8)
        row, col = center

        # 计算旋转后的矩形四个角点
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)

        # 定义矩形的半长和半宽
        half_length = VEHICLE_LENGTH // 2
        half_width = VEHICLE_WIDTH // 2

        # 计算四个角点（考虑旋转）
        corners = []
        for dx, dy in [(-half_length, -half_width),
                       (-half_length, half_width),
                       (half_length, half_width),
                       (half_length, -half_width)]:
            x = dx * cos_theta - dy * sin_theta + col
            y = dx * sin_theta + dy * cos_theta + row
            corners.append((int(x), int(y)))

        # 使用cv2.fillPoly填充旋转后的矩形
        corners = np.array(corners)
        cv2_corners = corners.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [cv2_corners.astype(np.int32)], 1)

        return mask

    def split_region(region_mask, estimated_count):
        """根据估计的车辆数量分割区域"""
        # 确保region_mask是uint8类型
        region_mask = region_mask.astype(np.uint8)

        # 获取区域的轮廓点
        rows, cols = np.where(region_mask)
        if len(rows) == 0:
            return np.zeros_like(region_mask, dtype=np.uint8)

        # 计算区域的主方向
        angle = get_region_orientation(region_mask)

        # 找到区域的中心
        center_row = (rows.max() + rows.min()) // 2
        center_col = (cols.max() + cols.min()) // 2

        # 计算区域长度
        length = ((rows.max() - rows.min()) ** 2 +
                  (cols.max() - cols.min()) ** 2) ** 0.5

        # 在主方向上均匀分布车辆
        vehicle_masks = []
        if estimated_count == 1:
            # 单个车辆，直接在中心创建
            vehicle_mask = create_vehicle_mask(
                (center_row, center_col),
                angle,
                region_mask.shape
            )
            # 确保使用uint8类型进行与运算
            vehicle_masks.append(vehicle_mask & region_mask)
        else:
            # 多个车辆，沿主方向分布
            step = length / (estimated_count + 1)  # +1使分布更均匀
            for i in range(estimated_count):
                offset = (i - (estimated_count - 1) / 2) * step
                row = int(center_row + offset * np.sin(angle))
                col = int(center_col + offset * np.cos(angle))

                vehicle_mask = create_vehicle_mask(
                    (row, col),
                    angle,
                    region_mask.shape
                )
                # 确保使用uint8类型进行与运算
                vehicle_masks.append(vehicle_mask & region_mask)

        # 合并所有车辆掩码（使用uint8类型）
        final_mask = np.zeros_like(region_mask, dtype=np.uint8)
        for mask in vehicle_masks:
            final_mask = cv2.bitwise_or(final_mask, mask.astype(np.uint8))

        return final_mask

    # 处理每个连通区域
    for region in regions:
        area = region.area
        region_mask = (labeled_array == region.label).astype(np.uint8)

        # 如果面积接近单个车辆，直接处理
        if 0.8 * VEHICLE_AREA <= area <= 1.2 * VEHICLE_AREA:
            result |= region_mask
            continue

        # 估计区域内车辆数量并分割
        estimated_count = estimate_vehicle_count(area)
        split_mask = split_region(region_mask, estimated_count)
        result = cv2.bitwise_or(result, split_mask)

    # 转换回PyTorch tensor并恢复维度
    result_tensor = torch.from_numpy(result).float().unsqueeze(0)

    return result_tensor
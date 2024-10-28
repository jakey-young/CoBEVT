import numpy as np
import torch
from opencood.utils import common_utils


def random_flip_along_x(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    return gt_boxes, points


def random_flip_along_y(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0],
                                       rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :],
                                                np.array([noise_rotation]))[0]

    gt_boxes[:, 0:3] = \
        common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3],
                                           np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation

    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[
            np.newaxis, :, :],
            np.array([noise_rotation]))[0][:, 0:2]

    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale

    return gt_boxes, points


def signal_to_noise(bev_embedding: torch.Tensor, snr: float = 0.5):
    """
    Add signal to noise to bev embedding.
    Args:
        bev_embedding: (B, C, H, W) - Batch of BEV embeddings
        snr: Desired signal-to-noise ratio
    Returns:
        noisy_bev: (B, C, H, W) - Noisy BEV embeddings
    """
    # Calculate signal power
    signal_power = torch.mean(bev_embedding ** 2, dim=(1, 2, 3), keepdim=True).squeeze()

    # Calculate noise power
    noise_power = signal_power / snr

    # Generate noise with desired power
    noise = torch.normal(mean=0, std=torch.sqrt(noise_power).view(-1, 1, 1, 1).repeat(1, *bev_embedding.shape[1:])).to(
        device=bev_embedding.device)

    # Add noise and clip
    noisy_bev = bev_embedding + noise
    noisy_bev = torch.clamp(noisy_bev, bev_embedding.min(), bev_embedding.max())  # Assuming 0-255 range

    return noisy_bev


def zero_out(bev_embedding: torch.Tensor, probability: float = 0.1):
    """
    Zero out a random portion of the BEV embedding.
    Args:
        bev_embedding: (B, C, H, W) - Batch of BEV embeddings
        probability: Probability of zeroing out a pixel
    Returns:
        noisy_bev: (B, C, H, W) - Noisy BEV embeddings
    """
    # Generate random mask
    mask = torch.rand(bev_embedding.shape) < probability
    mask = mask.to(bev_embedding.device)

    # Zero out masked pixels
    noisy_bev = bev_embedding * ~mask

    return noisy_bev


def full_zero_out(bev_embedding: torch.Tensor):
    """
    Zero out the entire BEV embedding.
    Args:
        bev_embedding: (B, C, H, W) - Batch of BEV embeddings
    Returns:
        noisy_bev: (B, C, H, W) - Noisy BEV embeddings
    """

    noisy_bev = torch.zeros_like(bev_embedding)

    return noisy_bev
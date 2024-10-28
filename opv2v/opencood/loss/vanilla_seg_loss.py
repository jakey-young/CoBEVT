import torch
import torch.nn as nn

from einops import rearrange
import torch.nn.functional as F

class WindowSelectionLoss(nn.Module):
    def __init__(self, sim_type='cosine'):
        super().__init__()
        self.sim_type = sim_type

    def compute_dissimilarity(self, x1, x2):
        if self.sim_type == 'cosine':
            x1_flat = x1.view(x1.size(0), -1)
            x2_flat = x2.view(x2.size(0), -1)
            return 1 - F.cosine_similarity(x1_flat, x2_flat)
        elif self.sim_type == 'l2':
            return torch.norm(x1 - x2, dim=(1, 2, 3))

    def forward(self, features, window_logits, selected_window):
        B, T, C, H, W = features.shape

        # 计算完整序列的差异（真值）
        full_seq_dissim = 0
        for t in range(1, T):
            full_seq_dissim += self.compute_dissimilarity(features[:, t], features[:, t - 1])
        full_seq_dissim /= (T - 1)

        # 计算选定窗口的差异
        window_dissim = 0
        for b in range(B):
            window_size = int(selected_window[b].item())
            for t in range(T - window_size, T):
                window_dissim += self.compute_dissimilarity(features[b, t], features[b, t - 1])
            window_dissim /= window_size

        # 计算差异的逆相关性
        divergence = 1 / (1 + torch.exp(-(full_seq_dissim - window_dissim)))

        # 窗口选择的交叉熵损失
        ce_loss = F.cross_entropy(window_logits, (selected_window - 1).long())

        # 总损失
        loss = -torch.log(divergence + 1e-6) + ce_loss

        return loss.mean()

class VanillaSegLoss(nn.Module):
    def __init__(self, args):
        super(VanillaSegLoss, self).__init__()

        self.d_weights = args['d_weights']
        self.s_weights = args['s_weights']
        self.l_weights = 50 if 'l_weights' not in args else args['l_weights']

        self.d_coe = args['d_coe']
        self.s_coe = args['s_coe']
        self.target = args['target']

        self.loss_func_static = \
            nn.CrossEntropyLoss(
                weight=torch.Tensor([1., self.s_weights, self.l_weights]).cuda())
        self.loss_func_dynamic = \
            nn.CrossEntropyLoss(
                weight=torch.Tensor([1., self.d_weights]).cuda())

        self.window_loss = WindowSelectionLoss()

        self.loss_dict = {}

    def forward(self, output_dict, gt_dict, temp_features, selected_window, window_logits):
        """
        Perform loss function on the prediction.

        Parameters
        ----------
        output_dict : dict
            The dictionary contains the prediction.

        gt_dict : dict
            The dictionary contains the groundtruth.

        Returns
        -------
        Loss dictionary.
        """
        # 自适应窗口损失
        adp_time_window_loss = self.window_loss(temp_features, selected_window, window_logits)

        static_pred = output_dict['static_seg']
        dynamic_pred = output_dict['dynamic_seg']

        static_loss = torch.tensor(0, device=static_pred.device)
        dynamic_loss = torch.tensor(0, device=dynamic_pred.device)

        # during training, we only need to compute the ego vehicle's gt loss
        static_gt = gt_dict['gt_static']
        dynamic_gt = gt_dict['gt_dynamic']
        static_gt = rearrange(static_gt, 'b l h w -> (b l) h w')
        dynamic_gt = rearrange(dynamic_gt, 'b l h w -> (b l) h w')

        if self.target == 'dynamic':
            dynamic_pred = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w')
            dynamic_loss = self.loss_func_dynamic(dynamic_pred, dynamic_gt)

        elif self.target == 'static':
            static_pred = rearrange(static_pred, 'b l c h w -> (b l) c h w')
            static_loss = self.loss_func_static(static_pred, static_gt)

        else:
            dynamic_pred = rearrange(dynamic_pred, 'b l c h w -> (b l) c h w')
            dynamic_loss = self.loss_func_dynamic(dynamic_pred, dynamic_gt)
            static_pred = rearrange(static_pred, 'b l c h w -> (b l) c h w')
            static_loss = self.loss_func_static(static_pred, static_gt)

        total_loss = self.s_coe * static_loss + self.d_coe * dynamic_loss
        self.loss_dict.update({'total_loss': total_loss,
                               'static_loss': static_loss,
                               'dynamic_loss': dynamic_loss})

        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss']
        static_loss = self.loss_dict['static_loss']
        dynamic_loss = self.loss_dict['dynamic_loss']

        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || static Loss: %.4f"
                " || Dynamic Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), static_loss.item(), dynamic_loss.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || static Loss: %.4f"
                  " || Dynamic Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item(), static_loss.item(), dynamic_loss.item()))


        writer.add_scalar('Static_loss', static_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('Dynamic_loss', dynamic_loss.item(),
                          epoch*batch_len + batch_id)





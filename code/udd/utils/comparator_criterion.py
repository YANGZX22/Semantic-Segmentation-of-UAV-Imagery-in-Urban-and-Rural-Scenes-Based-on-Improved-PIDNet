# ------------------------------------------------------------------------------
# Modified for UDD comparison models based on HRNet/PIDNet codebase
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F
from configs import config


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):
        return self.criterion(score, target)

    def forward(self, score, target):
        # 🌟 核心保护：单张量直接计算（单输出模型）
        if isinstance(score, torch.Tensor):
            return self._forward(score, target)

        if not isinstance(score, (list, tuple)):
            score = [score]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS

        if len(score) == 1:
            w = balance_weights[0] if len(balance_weights) == 1 else 1.0
            return w * self._forward(score[0], target)
        elif len(balance_weights) == len(score):
            return sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, score)])
        else:
            return sb_weights * self._forward(score[0], target)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7, min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        return self.criterion(score, target)

    def _ohem_forward(self, score, target, **kwargs):
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        pixel_losses = pixel_losses[mask]
        if pixel_losses.numel() == 0:
            return score.sum() * 0.0

        pred = F.softmax(score, dim=1)
        num_classes = score.size(1)

        safe_target = torch.clamp(target, 0, num_classes - 1)
        pred = pred.gather(1, safe_target.unsqueeze(1))
        pred = pred.contiguous().view(-1)[mask]

        if pred.numel() == 0:
            return score.sum() * 0.0

        pred, ind = pred.contiguous().sort()
        min_idx = max(0, min(self.min_kept, pred.numel() - 1))
        min_value = pred[min_idx]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[ind]
        pixel_losses = pixel_losses[pred < threshold]

        if pixel_losses.numel() == 0:
            return score.sum() * 0.0

        return pixel_losses.mean()

    def forward(self, score, target):
        if isinstance(score, torch.Tensor):
            return self._ohem_forward(score, target)

        if not isinstance(score, (list, tuple)):
            score = [score]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS

        if len(score) == 1:
            w = balance_weights[0] if len(balance_weights) == 1 else 1.0
            return w * self._ohem_forward(score[0], target)
        elif len(balance_weights) == len(score):
            functions = [self._ce_forward] * (len(balance_weights) - 1) + [self._ohem_forward]
            return sum([
                w * func(x, target)
                for (w, x, func) in zip(balance_weights, score, functions)
            ])
        else:
            return sb_weights * self._ohem_forward(score[0], target)


class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce=20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce

    def forward(self, bd_pre, bd_gt):
        return torch.tensor(0.0, device=bd_pre.device if isinstance(bd_pre, torch.Tensor) else 'cpu')
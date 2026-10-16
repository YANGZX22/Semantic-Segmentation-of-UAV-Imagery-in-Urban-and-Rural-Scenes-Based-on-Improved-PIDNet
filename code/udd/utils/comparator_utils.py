# ------------------------------------------------------------------------------
# Modified for UDD comparison models based on HRNet/PIDNet codebase
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config

class FullModel(nn.Module):
    def __init__(self, model, sem_loss, bd_loss=None):
        super(FullModel, self).__init__()
        self.model = model
        self.sem_loss = sem_loss
        self.bd_loss = bd_loss

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def forward(self, inputs, *args, **kwargs):
        # 🌟 灵活解包：模型只接受图像输入
        outputs = self.model(inputs)

        labels = args[0] if len(args) > 0 else kwargs.get('labels', None)
        bd_gts = args[1] if len(args) > 1 else kwargs.get('bd_gts', None)

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        # 尺寸对齐
        h, w = labels.size(1), labels.size(2)
        for i in range(len(outputs)):
            if outputs[i] is not None and (outputs[i].size(-2) != h or outputs[i].size(-1) != w):
                outputs[i] = F.interpolate(
                    outputs[i], size=(h, w),
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

        dummy_bce = torch.tensor(0.0, device=inputs.device)

        # PIDNet's explicit contract is [auxiliary, main, boundary].
        # Retain legacy per-head OHEM and zero boundary supervision.
        if config.MODEL.NAME.lower().startswith('pidnet_'):
            if len(outputs) != 3 or outputs[2].shape[1] != 1:
                raise ValueError('PIDNet output contract mismatch')
            if list(config.LOSS.BALANCE_WEIGHTS) != [0.4, 1.0]:
                raise ValueError('Expected historical PIDNet semantic weights')
            loss_aux = self.sem_loss(outputs[0], labels)
            loss_main = self.sem_loss(outputs[1], labels)
            total_loss = 0.4 * loss_aux + loss_main
            acc = self.pixel_acc(outputs[1], labels)
            return total_loss.unsqueeze(0), outputs[:2], acc, [total_loss, dummy_bce]

        # 单分支（UNet, HRNet, DeepLabV3+）
        if len(outputs) == 1:
            main_out = outputs[0]
            acc = self.pixel_acc(main_out, labels)
            loss_s = self.sem_loss(main_out, labels)
            return torch.unsqueeze(loss_s, 0), outputs, acc, [loss_s, dummy_bce]
        else:
            # 双分支兼容（BiSeNet, PSPNet）
            main_out = outputs[0]
            aux_out = outputs[1]
            acc = self.pixel_acc(main_out, labels)
            loss_main = self.sem_loss(main_out, labels)
            loss_aux = self.sem_loss(aux_out, labels)
            total_loss = 1.0 * loss_main + 0.4 * loss_aux
            return torch.unsqueeze(total_loss, 0), outputs, acc, [loss_main, dummy_bce]


class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    if not root_output_dir.exists():
        root_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / (cfg_name + '_' + time_str)
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
    return confusion_matrix


def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9):
    """
    🌟 核心保证：为 AdamW 的所有参数组提供一致的 Poly 衰减率，禁止放大 Bias / LayerNorm
    """
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
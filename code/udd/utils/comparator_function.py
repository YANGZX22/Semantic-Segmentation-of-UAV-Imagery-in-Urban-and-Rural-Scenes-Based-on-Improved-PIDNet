# ------------------------------------------------------------------------------
# Modified for UDD comparison models based on HRNet/PIDNet codebase
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

from utils.comparator_utils import AverageMeter
from utils.comparator_utils import get_confusion_matrix
from utils.comparator_utils import adjust_learning_rate


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, bd_gts, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()

        losses, _, acc, loss_list = model(images, labels, bd_gts)
        loss = losses.mean()
        acc = acc.mean()

        model.zero_grad()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - tic)
        tic = time.time()

        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(loss_list[0].mean().item())
        avg_bce_loss.update(loss_list[1].mean().item() if len(loss_list) > 1 else 0.0)

        # 🌟 统一为所有参数组进行相同比率的 Poly 衰减
        lr = adjust_learning_rate(optimizer, base_lr, num_iters, i_iter + cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            sb_loss_val = max(0.0, ave_loss.average() - avg_sem_loss.average() - avg_bce_loss.average())
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, lr: {:.7f}, ' \
                  'Loss: {:.6f}, Acc: {:.6f}, Semantic loss: {:.6f}, BCE loss: {:.6f}, SB loss: {:.6f}'.format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), optimizer.param_groups[0]['lr'], ave_loss.average(),
                      ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average(), sb_loss_val)
            logging.info(msg)

    writer_dict['last_train'] = {'loss': ave_loss.average(), 'lr_after_last_step': optimizer.param_groups[0]['lr'], 'steps': epoch_iters}
    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1


def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, bd_gts, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            bd_gts = bd_gts.float().cuda()

            losses, pred, _, _ = model(image, label, bd_gts)

            if not isinstance(pred, (list, tuple)):
                pred = [pred]

            # 🌟 修复：遍历实际存在的预测分支，防止索引越界
            for i, x in enumerate(pred):
                if i >= nums:
                    break
                if x.size()[-2:] != size[-2:]:
                    x = F.interpolate(
                        input=x, size=size[-2:],
                        mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                    )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            loss = losses.mean()
            ave_loss.update(loss.item())

    main_mean_IoU = 0.0
    main_IoU_array = np.zeros(config.DATASET.NUM_CLASSES)
    main_pixel_acc = 0.0

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        
        # 🌟 计算真实的总体像素准确率 (OA)
        total_valid = pos.sum()
        pixel_acc = tp.sum() / max(1.0, total_valid)

        logging.info('Branch [{}] | mIoU: {:.4f} | OA: {:.4f} | Class IoUs: {}'.format(
            i, mean_IoU, pixel_acc, np.round(IoU_array, 4)))

        # Select the explicitly registered semantic main head.
        if i == config.TEST.OUTPUT_INDEX:
            main_mean_IoU = mean_IoU
            main_IoU_array = IoU_array
            main_pixel_acc = pixel_acc

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', main_mean_IoU, global_steps)
    writer.add_scalar('valid_OA', main_pixel_acc, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    
    writer_dict['last_confusion_matrices'] = confusion_matrix.astype('int64').transpose(2, 0, 1).tolist()
    # Return all metrics from the selected head and this same epoch.
    return ave_loss.average(), main_mean_IoU, main_IoU_array, main_pixel_acc
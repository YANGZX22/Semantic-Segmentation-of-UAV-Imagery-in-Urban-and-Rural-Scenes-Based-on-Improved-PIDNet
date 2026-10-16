import os
import cv2
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset
from data_integrity_bridge import open_split


class Caihua(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=8, 
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=512,
                 crop_size=(512, 512),
                 scale_factor=16,
                 mean=None,
                 std=None,
                 bd_dilate_size=4):

        manifest, self.split_role = open_split(root, list_path, 'caihua', augment=bool(multi_scale or flip))
        normalization = manifest['training_statistics']['normalization']
        if mean is not None and list(mean) != normalization['mean']:
            raise ValueError('Normalization must match the frozen training protocol')
        if std is not None and list(std) != normalization['std']:
            raise ValueError('Normalization must match the frozen training protocol')
        mean, std = normalization['mean'], normalization['std']

        super(Caihua, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        list_file_path = os.path.join(root, list_path) if not os.path.isabs(list_path) else list_path
        self.img_list = [line.strip().split() for line in open(list_file_path, encoding="utf-8")]

        self.files = self.read_files()
        
        # 菜花村数据集 8 类别映射 (0~7)
        # 0:背景, 1:耕地, 2:裸土, 3:道路, 4:林地, 5:建筑, 6:水体, 7:其他
        self.label_mapping = {-1: ignore_label, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 255: ignore_label}
        
        # 类别平衡权重 (保留在 CPU 上，避免 DataLoader 多进程 worker 触发 CUDA 初始化死锁)
        self.class_weights = torch.FloatTensor(manifest['training_statistics']['class_weights'])
        
        self.bd_dilate_size = bd_dilate_size

        # RGB 调色板 (类别 ID 0~7)
        self.color_list = [
            [0, 0, 0],         # 0: 背景 (黑色)
            [128, 0, 0],       # 1: 耕地 (深红/红褐)
            [0, 128, 0],       # 2: 裸土 (绿色)
            [128, 96, 0],      # 3: 道路 (土黄/褐色)
            [128, 0, 128],     # 4: 林地 (紫色)
            [24, 0, 112],      # 5: 建筑 (深蓝/靛青)
            [0, 140, 150],     # 6: 水体 (青绿/鸭羽青)
            [128, 128, 128]    # 7: 其他 (灰色)
        ]
    
    def read_files(self):
        files = []
        for item in self.img_list:
            if len(item) == 1:
                # 兼容完全没有标签的纯预测场景
                image_path = item[0]
                name = os.path.splitext(os.path.basename(image_path))[0]
                files.append({
                    "img": image_path,
                    "label": None,
                    "name": name,
                })
            else:
                # 标准包含真实标签的情况 (train.lst / val.lst / test.lst)
                image_path, label_path = item[0], item[1]
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })
        return files
        
    def convert_label(self, label, inverse=False):
        """
        稳健的标签重映射：先初始化全为 ignore_label 的空白矩阵，再依次填入对应类别，
        防止字典遍历时的原位覆盖冲突。
        """
        temp = label.copy()
        new_label = np.ones_like(temp, dtype=np.uint8) * self.ignore_label
        mapping = {v: k for k, v in self.label_mapping.items()} if inverse else self.label_mapping
        for src_k, target_v in mapping.items():
            new_label[temp == src_k] = target_v
        return new_label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        
        image = cv2.imread(os.path.join(self.root, item["img"]), cv2.IMREAD_COLOR)
        size = image.shape

        # 1. 如果没有真实标签（纯盲测推理）
        if item["label"] is None:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            return image.copy(), np.zeros((size[0], size[1])), np.zeros((size[0], size[1])), np.array(size), name

        # 2. 正常读取真实标签
        label = cv2.imread(os.path.join(self.root, item["label"]), cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        # 3. 针对测试/评估阶段（验证集图像尺寸已严格统一，无需多尺度和翻转）
        if self.split_role != 'train':
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            edge = np.zeros((label.shape[0], label.shape[1]), dtype=np.float32)
            return image.copy(), label.copy(), edge, np.array(size), name

        # 4. 训练阶段生成切片增强与边界图
        # 设置 edge_pad=False，防止 512x512 尺度下边缘地物被抹掉
        image, label, edge = self.gen_sample(image, label,
                                             self.multi_scale, self.flip, 
                                             edge_pad=False,
                                             edge_size=self.bd_dilate_size)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name
    
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def label2color(self, label):
        color_map = np.zeros(label.shape + (3,))
        for i, v in enumerate(self.color_list):
            color_map[label == i] = self.color_list[i]

        return color_map.astype(np.uint8)

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))
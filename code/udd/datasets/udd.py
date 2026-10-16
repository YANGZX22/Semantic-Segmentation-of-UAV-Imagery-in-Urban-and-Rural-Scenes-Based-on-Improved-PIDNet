# datasets/udd.py
# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os
import cv2
import numpy as np
from PIL import Image
import torch
from .base_dataset import BaseDataset


class UDD(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=5, 
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=512,
                 crop_size=(512, 512),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(UDD, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip
        
        list_file_path = os.path.join(root, list_path) if not list_path.startswith('/') else list_path
        self.img_list = [line.strip().split() for line in open(list_file_path)]

        self.files = self.read_files()
        
        # 映射关系：将切片灰度值映射为 0~4 连续标签
        # 108 -> 0: Building (建筑)
        # 119 -> 1: Vegetation (植被)
        # 90  -> 2: Road (道路)
        # 16  -> 3: Vehicle (车辆)
        # 0   -> 4: Background (其他/背景)
        self.label_mapping = {
            108: 0,
            119: 1,
            90: 2,
            16: 3,
            0: 4,
            255: ignore_label,
            -1: ignore_label
        }
        
        # 类别平衡权重 (默认全 1)
        self.class_weights = torch.FloatTensor([1.0] * self.num_classes).cuda()
        self.bd_dilate_size = bd_dilate_size

        # UDD5 官方标准 RGB 可视化调色板
        # 0: Building (102, 102, 156)
        # 1: Vegetation (107, 142, 35)
        # 2: Road (128, 64, 128)
        # 3: Vehicle (0, 0, 142)
        # 4: Background (0, 0, 0)
        self.color_list = [
            [102, 102, 156],   # 0: Building
            [107, 142, 35],    # 1: Vegetation
            [128, 64, 128],    # 2: Road
            [0, 0, 142],       # 3: Vehicle
            [0, 0, 0]          # 4: Background
        ]
    
    def read_files(self):
        files = []
        for item in self.img_list:
            if len(item) == 1:
                image_path = item[0]
                name = os.path.splitext(os.path.basename(image_path))[0]
                files.append({
                    "img": image_path,
                    "label": None,
                    "name": name,
                })
            else:
                image_path, label_path = item[0], item[1]
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })
        return files
        
    
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for k, v in self.label_mapping.items():
                if k not in [-1, 255]:
                    label[temp == v] = k
            return label.astype(np.uint8)
        else:
            # 关键修改：将 dtype 改为 np.uint8，以满足 cv2.Canny 的 CV_8U 要求
            new_label = np.full(label.shape, self.ignore_label, dtype=np.uint8)
            for raw_val, target_class in self.label_mapping.items():
                if raw_val not in [-1, 255]:
                    new_label[temp == raw_val] = target_class
            return new_label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        
        image = cv2.imread(os.path.join(self.root, item["img"]), cv2.IMREAD_COLOR)
        size = image.shape

        # 1. 纯预测模式 (无真实标签)
        if item["label"] is None:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            return image.copy(), np.zeros((size[0], size[1])), np.zeros((size[0], size[1])), np.array(size), name

        # 2. 读取真实标签并转换
        label = cv2.imread(os.path.join(self.root, item["label"]), cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        # 3. 验证与评估阶段（不开启尺度缩放与随机翻转）
        if not self.multi_scale and not self.flip:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            edge = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
            return image.copy(), label.copy(), edge, np.array(size), name

        # 4. 训练阶段（多尺度缩放 + 翻转 + 边界提取）
        image, label, edge = self.gen_sample(image, label,
                                             self.multi_scale, self.flip, 
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

""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import logging
from .unet_parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=8, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        # 🌟 修复：无论训练还是评估，直接返回单张量 logits，坚决不返回伪分支列表 [logits, logits, None]
        return logits


def get_seg_model(cfg, imgnet_pretrained=False):
    model = UNet(
        n_channels=3,
        n_classes=cfg.DATASET.NUM_CLASSES,
        bilinear=False
    )
    
    if hasattr(cfg.MODEL, 'PRETRAINED') and cfg.MODEL.PRETRAINED:
        try:
            checkpoint = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            model_dict = model.state_dict()
            clean_dict = {
                k.replace('model.', '').replace('module.', ''): v 
                for k, v in state_dict.items() 
                if k.replace('model.', '').replace('module.', '') in model_dict 
                and v.shape == model_dict[k.replace('model.', '').replace('module.', '')].shape
            }
            model_dict.update(clean_dict)
            model.load_state_dict(model_dict, strict=False)
            logging.info(f"=> UNet successfully loaded {len(clean_dict)} pretrained layers")
        except Exception as e:
            logging.warning(f"=> UNet pretrained weight load skipped: {e}")
            
    return model
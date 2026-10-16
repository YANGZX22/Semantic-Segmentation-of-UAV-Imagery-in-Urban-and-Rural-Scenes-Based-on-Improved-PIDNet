import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo
from functools import partial

mobilenet_v2_url = 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2Backbone(nn.Module):
    def __init__(self, downsample_factor=16, width_mult=1.0, pretrained=True):
        super(MobileNetV2Backbone, self).__init__()
        block = InvertedResidual
        input_channel = 32
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(input_channel * width_mult, 8)
        self.features = [ConvBNReLU(3, input_channel, stride=2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        # 膨胀卷积保持分辨率
        total_idx = len(self.features)
        down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(down_idx[-2], down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(down_idx[-1], total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(down_idx[-1], total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

        # 🌟 载入 ImageNet 预训练权重
        if pretrained:
            self.init_weights()

    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2) or m.stride == 2:
                m.stride = (1, 1)
                if m.kernel_size == (3, 3) or m.kernel_size == 3:
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3) or m.kernel_size == 3:
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    # def init_weights(self):
    #     try:
    #         state_dict = modelzoo.load_url(mobilenet_v2_url)
    #         self_dict = self.state_dict()
    #         matched = {k: v for k, v in state_dict.items() if k in self_dict and v.shape == self_dict[k].shape}
    #         self_dict.update(matched)
    #         self.load_state_dict(self_dict, strict=False)
    #         logging.info(f"=> PSPNet loaded {len(matched)} layers from MobileNetV2 pretrained weights")
    #     except Exception as e:
    #         logging.warning(f"=> Failed to load MobileNetV2 pretrained weights: {e}")
    def init_weights(self, pretrained_path=''):
        try:
            if pretrained_path.startswith('http://') or pretrained_path.startswith('https://'):
            # 自动下载到指定缓存目录，后续无需重复下载
                state_dict = modelzoo.load_url(pretrained_path)
            elif pretrained_path and os.path.exists(pretrained_path):
                state_dict = torch.load(pretrained_path, map_location='cpu')
            else:
                state_dict = modelzoo.load_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth')

            self_dict = self.state_dict()
            matched = {k: v for k, v in state_dict.items() if k in self_dict and v.shape == self_dict[k].shape}
            self_dict.update(matched)
            self.load_state_dict(self_dict, strict=False)
            logging.info(f"=> PSPNet loaded {len(matched)} layers from MobileNetV2 pretrained weights successfully")
        except Exception as e:
            logging.warning(f"=> Failed to load MobileNetV2 pretrained weights: {e}")

    def forward(self, x):
        x_aux = self.features[:14](x)
        x = self.features[14:](x_aux)
        return x_aux, x


class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        self.stages = nn.ModuleList([
            self._make_stages(in_channels, out_channels, pool_size, norm_layer) 
            for pool_size in pool_sizes
        ])
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([
            F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) 
            for stage in self.stages
        ])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class PSPNet(nn.Module):
    def __init__(self, num_classes=8, downsample_factor=16, augment=True, aux_branch=True, pretrained=True):
        super(PSPNet, self).__init__()
        self.augment = augment
        self.aux_branch = aux_branch
        norm_layer = nn.BatchNorm2d

        # 🌟 主干加载预训练
        self.backbone = MobileNetV2Backbone(downsample_factor=downsample_factor, pretrained=pretrained)
        aux_channel = 96
        out_channel = 320

        self.master_branch = nn.Sequential(
            _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(out_channel // 4, num_classes, kernel_size=1)
        )

        if self.aux_branch:
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(aux_channel, out_channel // 8, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channel // 8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channel // 8, num_classes, kernel_size=1)
            )

        self.initialize_weights(self.master_branch)
        if self.aux_branch:
            self.initialize_weights(self.auxiliary_branch)

    def forward(self, x):
        input_size = (x.size(2), x.size(3))
        x_aux, x_feat = self.backbone(x)
        output = self.master_branch(x_feat)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)

        if self.training and self.augment:
            if self.aux_branch:
                output_aux = self.auxiliary_branch(x_aux)
                output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
                # 🌟 规范顺序：[主预测头, 辅助预测头]
                return [output, output_aux]
            return [output]
        
        return output

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)


def get_seg_model(cfg, imgnet_pretrained=True):
    model = PSPNet(
        num_classes=cfg.DATASET.NUM_CLASSES,
        downsample_factor=16,
        augment=True,
        aux_branch=True,
        pretrained=imgnet_pretrained
    )
    return model
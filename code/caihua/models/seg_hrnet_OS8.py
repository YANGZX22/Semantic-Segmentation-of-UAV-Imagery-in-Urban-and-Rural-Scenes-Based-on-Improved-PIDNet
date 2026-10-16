"""HRNet-W18-Small-v2 with the decoder fused and classified at 1/8 scale (64x64 for a 512x512 input).

Every parameter and buffer is identical to `seg_hrnet.HighResolutionNet` (same state-dict keys, same
initialisation, same pretrained loading). Only the resolution at which the four branches are fused
changes: the original upsamples the 1/8, 1/16 and 1/32 branches to the 1/4 branch and classifies on
the 1/4 grid (128x128); this variant average-pools the 1/4 branch onto the 1/8 grid, upsamples the
1/16 and 1/32 branches to it, and classifies on the 1/8 grid (64x64), matching PIDNet at output
stride 8. The runtime bilinearly resizes the logits to label size exactly as it does for PIDNet OS8.
"""
import torch
from torch.nn import functional as F
from . import seg_hrnet
from .seg_hrnet import HighResolutionNet as _Parent


def fuse_at_os8(branches, align_corners):
    x0, x1, x2, x3 = branches
    size = x1.shape[-2:]
    x0 = F.adaptive_avg_pool2d(x0, size)
    x2 = F.interpolate(x2, size=size, mode='bilinear', align_corners=align_corners)
    x3 = F.interpolate(x3, size=size, mode='bilinear', align_corners=align_corners)
    return torch.cat([x0, x1, x2, x3], 1)


class HighResolutionNet(_Parent):
    output_stride = 8

    def branch_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        return self.stage4(x_list)

    def forward(self, x):
        branches = self.branch_features(x)
        return self.last_layer(fuse_at_os8(branches, seg_hrnet.ALIGN_CORNERS))

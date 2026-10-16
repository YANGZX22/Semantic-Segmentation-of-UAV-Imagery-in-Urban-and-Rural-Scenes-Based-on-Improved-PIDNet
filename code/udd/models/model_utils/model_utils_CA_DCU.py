"""Prediction and pooling blocks for CA + DCU."""
import torch
import torch.nn as nn
import torch.nn.functional as F
BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out if self.no_relu else self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out if self.no_relu else self.relu(out)

class DepthwiseConvUpSample(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=8):
        super(DepthwiseConvUpSample, self).__init__()
        self.depthwise = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2 * scale_factor, stride=scale_factor, padding=scale_factor // 2, groups=in_channels, bias=False)
        self.bn = BatchNorm2d(in_channels, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pointwise(x)
        return x

class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=8):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor
        if self.scale_factor is not None:
            self.upconv = DepthwiseConvUpSample(outplanes, outplanes, scale_factor=self.scale_factor)

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:
            out = self.upconv(out)
        return out

class DAPPM(nn.Module):

    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(DAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2), BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4), BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8), BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale0 = nn.Sequential(BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.process1 = nn.Sequential(BatchNorm(branch_planes, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False))
        self.process2 = nn.Sequential(BatchNorm(branch_planes, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False))
        self.process3 = nn.Sequential(BatchNorm(branch_planes, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False))
        self.process4 = nn.Sequential(BatchNorm(branch_planes, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False))
        self.compression = nn.Sequential(BatchNorm(branch_planes * 5, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False))
        self.shortcut = nn.Sequential(BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False))

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []
        x_list.append(self.scale0(x))
        x_list.append(self.process1(F.interpolate(self.scale1(x), size=[height, width], mode='bilinear', align_corners=algc) + x_list[0]))
        x_list.append(self.process2(F.interpolate(self.scale2(x), size=[height, width], mode='bilinear', align_corners=algc) + x_list[1]))
        x_list.append(self.process3(F.interpolate(self.scale3(x), size=[height, width], mode='bilinear', align_corners=algc) + x_list[2]))
        x_list.append(self.process4(F.interpolate(self.scale4(x), size=[height, width], mode='bilinear', align_corners=algc) + x_list[3]))
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out

class CoordAttention(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity * a_h * a_w

class PAPPM(nn.Module):

    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2), BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4), BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8), BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale0 = nn.Sequential(BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False))
        self.scale_process = nn.Sequential(BatchNorm(branch_planes * 4, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(branch_planes * 4, branch_planes * 4, kernel_size=3, padding=1, groups=4, bias=False))
        self.compression = nn.Sequential(BatchNorm(branch_planes * 5, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False))
        self.shortcut = nn.Sequential(BatchNorm(inplanes, momentum=bn_mom), nn.ReLU(inplace=True), nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False))
        self.co_orattention = CoordAttention(outplanes, outplanes)

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale_list = []
        x_ = self.scale0(x)
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width], mode='bilinear', align_corners=algc) + x_)
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width], mode='bilinear', align_corners=algc) + x_)
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width], mode='bilinear', align_corners=algc) + x_)
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width], mode='bilinear', align_corners=algc) + x_)
        scale_out = self.scale_process(torch.cat(scale_list, 1))
        out = self.compression(torch.cat([x_, scale_out], 1)) + self.shortcut(x)
        return self.co_orattention(out)

class PagFM(nn.Module):

    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False), BatchNorm(mid_channels))
        self.f_y = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False), BatchNorm(mid_channels))
        if with_channel:
            self.up = nn.Sequential(nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False), BatchNorm(in_channels))
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]], mode='bilinear', align_corners=False)
        x_k = self.f_x(x)
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        y = F.interpolate(y, size=[input_size[2], input_size[3]], mode='bilinear', align_corners=False)
        return (1 - sim_map) * x + sim_map * y

class Light_Bag(nn.Module):

    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Light_Bag, self).__init__()
        self.conv_p = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), BatchNorm(out_channels))
        self.conv_i = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), BatchNorm(out_channels))

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv_p((1 - edge_att) * i + p) + self.conv_i(i + edge_att * p)

class DDFMv2(nn.Module):

    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(DDFMv2, self).__init__()
        self.conv_p = nn.Sequential(BatchNorm(in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), BatchNorm(out_channels))
        self.conv_i = nn.Sequential(BatchNorm(in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), BatchNorm(out_channels))

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv_p((1 - edge_att) * i + p) + self.conv_i(i + edge_att * p)

class Bag(nn.Module):

    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Bag, self).__init__()
        self.conv = nn.Sequential(BatchNorm(in_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att * p + (1 - edge_att) * i)

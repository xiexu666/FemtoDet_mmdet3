import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmcv.cnn.bricks import build_activation_layer
from torch.nn import Parameter
import torch, torch.nn as nn, torch.nn.functional as F


class IBEConvModule(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, act_cfg=None, **kwargs):

        super(IBEConvModule, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # self.att = DifferenceAttention(out_channels, out_channels, out_channels)

        self.bn = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = build_activation_layer(act_cfg) #nn.ReLU()
        self.theta = Parameter(torch.zeros([1]))

    def merge_bn(self, conv, bn):
        conv_w = conv
        conv_b = torch.zeros_like(bn.running_mean)

        factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        weight = nn.Parameter(conv_w *
                                factor.reshape([conv_w.shape[0], 1, 1, 1]))
        bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
        return weight, bias

    def forward(self, x):
        # if self.training:
        out_normal = self.conv(x)
        [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff, stride=self.conv.stride, padding=0, groups=self.conv.groups)

        theta = F.sigmoid(self.theta)#[None, :, None, None]
        # outs = self.att(out_normal, out_normal - theta * out_diff)
        outs = self.bn(out_normal) + self.bn2(out_normal - theta * out_diff)
        # outs = self.bn(out_normal) - theta * self.bn2(out_diff)
        # else:
        #     weight_conv = self.conv.weight

        #     theta = F.sigmoid(self.theta)#[:, None, None, None]
        #     kernel_diff = theta * self.conv.weight.sum(2).sum(2)[:, :, None, None]
        #     weight_diff = self.conv.weight - nn.ZeroPad2d(1)(kernel_diff)

        #     weight_conv, bias_conv = self.merge_bn(weight_conv, self.bn)
        #     weight_diff, bias_diff = self.merge_bn(weight_diff, self.bn2)
        #     weight_final = weight_conv + weight_diff
        #     bias_final = bias_conv + bias_diff

        #     outs = F.conv2d(input=x, weight=weight_final, bias=bias_final, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)
        #     # outs = self.act(outs)

        return self.act(outs)
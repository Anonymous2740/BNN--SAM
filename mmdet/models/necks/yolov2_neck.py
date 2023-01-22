# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import NECKS


class reorg_layer(nn.Module):
    def __init__(self, stride):
        super(reorg_layer, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _height, _width = height // self.stride, width // self.stride

        x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, self.stride * self.stride, _height, _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)

        return x

class DetectionBlockV2(nn.Module):
    """Detection block in YOLO neck.

    Let out_channels = n, the DetectionBlock contains:
    Six ConvLayers, 1 Conv2D Layer and 1 YoloLayer.
    The first 6 ConvLayers are formed the following way:
        1x1xn, 3x3x2n, 1x1xn, 3x3x2n, 1x1xn, 3x3x2n.
    The Conv2D layer is 1x1x255.
    Some block will have branch after the fifth ConvLayer.
    The input channel is arbitrary (in_channels)

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(DetectionBlockV2, self).__init__()

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(in_channels[0], out_channels[0], 3, padding=1, **cfg)
        self.conv2 = ConvModule(
            in_channels[0], out_channels[0], 3, padding=1, **cfg)
        self.route_layer = ConvModule(
            in_channels[1], out_channels[1], 1, **cfg)
        self.reorg = reorg_layer(stride=2)
        self.convsets_2 = ConvModule(
            in_channels[2], out_channels[2], 3, padding=1, **cfg)

    def forward(self, x1, x2):
        tmp2 = self.conv1(x2)
        tmp2 = self.conv2(tmp2)
        tmp1 = self.reorg(self.route_layer(x1))
        # route concatenate
        tmp = torch.cat([tmp1, tmp2], dim=1)
        tmp = self.convsets_2(tmp)
        return tmp

@NECKS.register_module()
class YOLOV2Neck(nn.Module):
    """The neck of YOLOV2.

    It can be treated as a simplified version of FPN. It
    will take the result from Darknet backbone and do some upsampling and
    concatenation. It will finally output the detection result.

    Note:
        The input feats should be from top to bottom.
            i.e., from high-lvl to low-lvl
        But YOLOV2Neck will process them in reversed order.
            i.e., from bottom (high-lvl) to top (low-lvl)

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(YOLOV2Neck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # To support arbitrary scales, the code looks awful, but it works.
        # Better solution is welcomed.
        self.DetectionBlock = DetectionBlockV2(in_channels, out_channels, **cfg)

    def forward(self, feat1, feat2):
        out = self.DetectionBlock(feat1, feat2)
        return out

    def init_weights(self):
        """Initialize the weights of module."""
        # init is done in ConvModule
        pass

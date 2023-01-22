# Copyright (c) Open-MMLab. All rights reserved.
import logging
from math import isnan
import torch
import torch.nn as nn


from mmcv.runner import load_checkpoint
from mmcv.cnn.utils import constant_init, kaiming_init, normal_init
from .irnet.binaryfunction import BinarizeConv2d

CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                 downsample=None):
        super(BasicBlock, self).__init__()
        self.conv = BinarizeConv2d(in_channels, out_channels,
                                   kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.downsample = downsample


    def forward(self, x):
        residual = x
        
        x = self.bn(self.conv(x))

        # if self.downsample is not None:
        #     residual = self.downsample(residual)

        # x += residual

        return x
def make_vgg_layer(cfg, i=3, bias=False):
    layers = []
    in_channels = i
    for j in range(len(cfg)):
        v = cfg[j]
        if v == 'M':#if Maxpool layer，replace it to Conv2d(stride=2)
            layers += [BasicBlock(in_channels=cfg[j - 1], out_channels=cfg[j + 1],
                                     kernel_size=3, stride=2, padding=1, bias=bias,
                                    #  downsample=nn.Sequential(
                                    #      nn.AvgPool2d(kernel_size=2, stride=2),
                                    #      nn.Conv2d(in_channels=cfg[j - 1], out_channels=cfg[j + 1],
                                    #                kernel_size=1, stride=1, padding=0, bias=bias),
                                    #      nn.BatchNorm2d(cfg[j + 1])
                                    #  )
                                     )]
            # layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            in_channels = cfg[j + 1]
        elif v == 'C':
            layers += [BasicBlock(in_channels=cfg[j - 1], out_channels=cfg[j + 1],
                                     kernel_size=3, stride=2, padding=1, bias=bias,
                                     downsample=nn.Sequential(
                                         nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),
                                         nn.Conv2d(in_channels=cfg[j - 1], out_channels=cfg[j + 1],
                                                   kernel_size=1, stride=1, padding=0, bias=bias),
                                         nn.BatchNorm2d(cfg[j + 1])
                                     ))]
            in_channels = cfg[j + 1]
        else:
            if in_channels == i:
                conv = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1, bias=bias)# shouldn't binarize first layer
                layers += [conv, nn.BatchNorm2d(v)]
            else:
                layers += [BasicBlock(in_channels, v, kernel_size=3, padding=1, bias=bias)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = BinarizeConv2d(512, 1024, kernel_size=3, padding=6, dilation=6, bias=bias)
    # conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6, bias=bias)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)  # shouldn't binarize fc layer
    layers += [pool5]
    layers += [conv6, nn.BatchNorm2d(1024), nn.ReLU(inplace=True)]#conv6进行了二值化，但是用到的激活函数是Relu
    layers += [conv7, nn.BatchNorm2d(1024), nn.ReLU(inplace=True)]

    return layers

    


class VGG(nn.Module):
    """VGG backbone.

    Args:
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        with_bn (bool): Use BatchNorm or not.
        num_classes (int): number of classes for classification.
        num_stages (int): VGG stages, normally 5.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers as eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
    """

    arch_settings = {
        11: (1, 1, 2, 2, 2),
        13: (2, 2, 2, 2, 2),
        16: (1, 2, 3, 3, 3),
        19: (2, 2, 4, 4, 4)
    }
    CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]

    def __init__(self,
                 depth,
                 with_bn=True,
                 num_classes=-1,
                 num_stages=5,
                 dilations=(1, 1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3, 4),
                 frozen_stages=-1,
                 bn_eval=False,
                 bn_frozen=False,
                 ceil_mode=False,
                 with_last_pool=True):
        super(VGG, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for vgg')
        assert num_stages >= 1 and num_stages <= 5
        stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        assert len(dilations) == num_stages
        assert max(out_indices) <= num_stages

        self.num_classes = num_classes
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen

        self.inplanes = 3

        vgg_layers = []

        vgg_layer = make_vgg_layer(CFG, i=3,bias=False)
        vgg_layers.extend(vgg_layer)

        self.module_name = 'features'
        self.add_module(self.module_name, nn.Sequential(*vgg_layers))

        if self.num_classes > 0:
            self.classifier = nn.Sequential(#一些全连接层
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )


    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = logging.getLogger()
    #         load_checkpoint(self, pretrained, strict=False, logger=logger)
    #     elif pretrained is None:
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 kaiming_init(m)
    #             elif isinstance(m, nn.BatchNorm2d):
    #                 constant_init(m, 1)
    #             elif isinstance(m, nn.Linear):
    #                 normal_init(m, std=0.01)
    #     else:
    #         raise TypeError('pretrained must be a str or None')

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self,x):
        outs = []
        x = self.vgg_layers(x)
        outs.append(x)
        return outs


    def train(self, mode=True):
        super(VGG, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        vgg_layers = getattr(self, self.module_name)
        if mode and self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                for j in range(*self.range_sub_modules[i]):
                    mod = vgg_layers[j]
                    mod.eval()
                    for param in mod.parameters():
                        param.requires_grad = False
   
# Copyright (c) Open-MMLab. All rights reserved.
import logging
import torch
import torch.nn as nn

import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from mmcv.cnn.utils import constant_init, kaiming_init, normal_init
from .irnet.ReActConv2d import ReActConv2d,prelu, Scale_Hardtanh

CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

#用二阶拟合 sign 的 ApproxSign 的导数来作为 sign的导数，从而缩小导数值的不匹配问题
class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        # out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

#用实数值参数的绝对值平均值计算的标量乘以实数值参数的符号作为网络计算导数的二值化参数
#对权重进行二值化
class HardBinaryConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1,bias=False):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_channels * out_channels * kernel_size * kernel_size
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.kernel_size = [kernel_size,kernel_size]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.groups = groups
        self.bias = bias
        
    def forward(self, x):
        real_weights = self.weights.view(self.shape)#前向传播中的real_weights是随机生成的
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)#STE
        #我可以统计一下cliped_weights的比例
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y

# def sign_new(self,x,t):
#     if x > t:
#         x = 1.0
#     elif x< -t:
#         x = -1.0
#     elif count(x,-1)>count(x,1):
#         x = 1
#     else:
#         x = 1 
        

    

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                 downsample=None):
        super(BasicBlock, self).__init__()
        """real value"""
        # self.conv = nn.Conv2d(in_channels, out_channels,
        #                            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        # self.bn = nn.BatchNorm2d(out_channels)
        # self.downsample = downsample
        """binarize"""
        self.move0 = LearnableBias(in_channels)
        self.binary_activation = BinaryActivation()
        # self.binary_conv = conv3x3(in_channels,out_channels,stride=stride)
        self.binary_conv = HardBinaryConv(in_channels,out_channels,stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

        """prelu"""
        self.move1 = LearnableBias(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.move2 = LearnableBias(out_channels)

        # self.Scale_Hardtanh = Scale_Hardtanh(out_channels)

    def forward(self, x):
        """real value"""
        # residual = x

        # out = self.move0(x)
        # x = self.bn(self.conv(x))

        # if self.downsample is not None:
        #     residual = self.downsample(residual)

        # x += residual

        # # x = self.relu(x)#new add

        # return x


        """binarize"""
        # residual = x

        out = self.move0(x)
        out = self.binary_activation(out)
        # out = out.half()
        out = self.binary_conv(out)
        out = self.bn1(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        # out += residual

        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)
        
        # out = self.Scale_Hardtanh(out)

        return out


def make_vgg_layer(cfg, i=3, bias=False):
    layers = []
    in_channels = i
    for j in range(len(cfg)):
        v = cfg[j]
        if v == 'M':#并不是把maxpool层去掉了，而是进行了替换，换成了Conv2d(stride=2)
            layers += [BasicBlock(in_channels=cfg[j - 1], out_channels=cfg[j + 1],
                                     kernel_size=3, stride=2, padding=1, bias=bias,
                                     downsample=nn.Sequential(#downsample的卷积核也没有进行二值化
                                         nn.AvgPool2d(kernel_size=2, stride=2),
                                         nn.Conv2d(in_channels=cfg[j - 1], out_channels=cfg[j + 1],
                                                   kernel_size=1, stride=1, padding=0, bias=bias),
                                         nn.BatchNorm2d(cfg[j + 1])
                                     ))]
            in_channels = cfg[j + 1]
        elif v == 'C':
            layers += [BasicBlock(in_channels=cfg[j - 1], out_channels=cfg[j + 1],
                                     kernel_size=3, stride=2, padding=1, bias=bias,
                                     downsample=nn.Sequential(
                                         nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True),
                                         nn.Conv2d(in_channels=cfg[j - 1], out_channels=cfg[j + 1],
                                                   kernel_size=1, stride=1, padding=0, bias=bias),
                                         nn.BatchNorm2d(cfg[j + 1])#shortcut的conv2d没有进行二值化
                                     ))]
            in_channels = cfg[j + 1]
        else:
            if in_channels == i:#第一层不进行二值化
                conv = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1, bias=bias)#第一层不进行二值化
                layers += [conv, nn.BatchNorm2d(v)]
            else:
                layers += [BasicBlock(in_channels, v, kernel_size=3, padding=1, bias=bias)]
            in_channels = v
#new add
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6, bias=bias)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)  # shouldn't binarize fc layer
    layers += [pool5]
    # layers += [ReActConv2d(conv6), nn.BatchNorm2d(1024), prelu(1024)]#conv6进行了二值化，但是用到的激活函数是Relu
    layers += [ReActConv2d(conv6), nn.BatchNorm2d(1024), prelu(1024)]
    # layers += [conv6, nn.BatchNorm2d(1024), nn.ReLU(inplace=True)]
    layers += [conv7, nn.BatchNorm2d(1024), nn.ReLU(inplace=True)]
# ##
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
                 with_bn=True,#False,
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


    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')



    def forward(self,x):
        outs = []
        x = self.vgg_layers(x)
        outs.append(x)#outs的长度为1了
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

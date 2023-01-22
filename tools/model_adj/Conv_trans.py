
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
import torch
import math

#把1*1卷积替换为3*3卷积
def Conv_trans(model):

    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = Conv_trans(model=module)
 
        if type(module) == nn.Conv2d:
            if module.kernel_size == (1,1): #把1*1的卷积换成3*3的卷积
                irnet_conv2d = Conv2d_kernel_chag(module)                           
            else:
                irnet_conv2d = module
            model._modules[name] = irnet_conv2d

    return model
                



class Conv2d_kernel_chag(Module):

    def __init__(self, conv):
        super(Conv2d_kernel_chag, self).__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = 3
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = Parameter(conv.weight.data.clone())

        try:
            self.bias = Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, input):

        w = self.weight
        a = input
        
        output = F.conv2d(a, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output
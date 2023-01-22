from torch.autograd import Function 
import torch
import torch.nn.functional as F
import torch.nn as nn
from mmcv.cnn.bricks.registry import CONV_LAYERS
# from mmcv.utils import registry


# from .Registry import CONV_LAYERS



class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None

# def binarize(x, k, t):
#     # print(x.abs().mean())
#     # t = x.abs().max()
#     # t = 10*x.std().detach()
#     # t = 10*x.std().detach()
#     # k = 1.0/t
#     # print(k)
#     # print(t)
#     # print(x)
#     k = k.cuda(x.cuda().device.index)
#     t = t.cuda(x.cuda().device.index)
#     clipped = k*torch.tanh(x*t)
#     rounded = torch.sign(clipped)
#     return clipped + (rounded - clipped).detach()

# def binarize_a(x, k, t):
#     clipped = torch.tanh(x)
#     rounded = torch.sign(clipped)
#     return clipped + (rounded - clipped).detach()

# def binarize(x, k, t):
#     clipped = torch.clamp(x,min=-1.0,max=1.0)
#     rounded = torch.sign(clipped)
#     return clipped + (rounded - clipped).detach()

# def binarize(x, k, t):
#     clipped = k*torch.tanh(t*x)
#     rounded = torch.sign(x)
#     return clipped + (rounded - clipped).detach()

# def binarize_clamp(x):
#     clipped = torch.clamp(x,min=-1.0,max=1.0)
#     rounded = torch.sign(x)
#     return clipped + (rounded - clipped).detach()

def binarize_clamp(x, alpha, beta):
    clipped = torch.clamp(alpha*x+beta,min=-1.0,max=1.0)#为什么要设置成alpha*x+beta?
    # rounded = torch.sign(clipped)
    rounded = torch.sign(x)
    return clipped + (rounded - clipped).detach()

# def binarize_a(x,l,r):
#     m1 = x < l
#     m2 = x > r
#     clipped = (l*m1.float() + r*m2.float() + x*(1-m1.float() )*(1-m2.float() ))
#     rounded = torch.sign(x)
#     return clipped + (rounded - clipped).detach()
#


def binarize_a(x):
    clipped = torch.clamp(x,min=-1.5,max=1.5)
    rounded = torch.sign(x)
    return clipped + (rounded - clipped).detach()

def binarize_w(x):
    clipped = torch.clamp(x,min=-1,max=1)
    rounded = torch.sign(x)
    return clipped + (rounded - clipped).detach()

# def binarize_a(x,l,r):
#     x = r*(x+l)
#     clipped = F.hardtanh(x)
#     rounded = torch.sign(clipped)
#     return clipped + (rounded - clipped).detach()


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
        
class SignSTEWeight(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod #y=x
    def backward(ctx, grad_output):
        grad_input = grad_output.new_empty(grad_output.size())
        grad_input.copy_(grad_output)
        return grad_input

# class SignSTEWeight(Function):
#     # def __init__(ctx, out_chn):
#     #     super(SignSTEWeight, self).__init__()
#     #     beta = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

#     @staticmethod
#     def forward(ctx, input):
#         # ctx.save_for_backward(input)
#         output = input.sign()
#         ctx.save_for_backward(input-output)
#         return output

#     @staticmethod #y=x
#     def backward( ctx, grad_output):
#         diff = ctx.saved_tensors[0]
#         grad_input = grad_output.new_empty(grad_output.size())
#         grad_input.copy_(grad_output)
#         scale = 1 +  0.2*torch.sign(grad_input)*diff
#         return grad_input*scale

class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.weights = self.weight
        # self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        # self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        # self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y


# class BinaryActivation(nn.Module):
#     def __init__(self):
#         super(BinaryActivation, self).__init__()

#     def forward(self, x):
def BinaryActivation(x):
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

class SignTwoOrders(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        grad_input_wrt_output2 = torch.zeros_like(grad_output)
        ge0_lt1_mask = input.ge(0) & input.lt(1)#[0,1]
        grad_input_wrt_output2 = torch.where(ge0_lt1_mask, (2 - 2 * input), grad_input_wrt_output2)
        gen1_lt0_mask = input.ge(-1) & input.lt(0)#[-1,0]
        grad_input_wrt_output2 = torch.where(gen1_lt0_mask, (2 + 2 * input), grad_input_wrt_output2)
        grad_input = grad_input_wrt_output2 * grad_output

        return grad_input

class SignSTE(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        mask = input.ge(-1) & input.le(1)
        grad_input = torch.where(mask, grad_output, torch.zeros_like(grad_output))
        return grad_input


class BinarizeConv2d(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, weight_magnitude_aware=True, activation_value_aware=True,
                 **kwargs):
        super(BinarizeConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.weight_magnitude_aware = weight_magnitude_aware
        self.activation_value_aware = activation_value_aware


    def forward(self, input):
        if self.activation_value_aware:
            input = SignTwoOrders.apply(input)
        else:
            input = SignSTE.apply(input)

        subed_weight = self.weight
        if self.weight_magnitude_aware:
            self.weight_bin_tensor = subed_weight.abs(). \
                                         mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True) \
                                     * SignSTEWeight.apply(subed_weight)#binarize_w(subed_weight) per_channel
#             self.weight_bin_tensor = subed_weight.abs().mean() \
#                                      * SignSTEWeight.apply(subed_weight)#binarize_w(subed_weight) # per_layer
        else:
            self.weight_bin_tensor = SignSTEWeight.apply(subed_weight)
        self.weight_bin_tensor.requires_grad_()

        input = F.pad(input, (self.padding[0], self.padding[0], self.padding[1], self.padding[1]),
                      mode='constant', value=-1)
        out = F.conv2d(input, self.weight_bin_tensor, self.bias, self.stride, 0, self.dilation, self.groups)
        return out
CONV_LAYERS.register_module('BinarizeConv2d',module=BinarizeConv2d)


"""the binary conv for the Bop(Binarized optimizer)"""
class BinarizeConv2dBop(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, weight_magnitude_aware=True, activation_value_aware=True,
                 **kwargs):
        super(BinarizeConv2dBop, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.weight_magnitude_aware = weight_magnitude_aware
        self.activation_value_aware = activation_value_aware

        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        # self.number_of_weights = in_channels * out_channels * kernel_size * kernel_size
        """weights全都初始化为+1"""
        # self.weight = nn.Parameter((torch.ones((self.number_of_weights,1))).view(self.shape), requires_grad=True)
        # self.weight = nn.Parameter(torch.ones((self.number_of_weights,1)), requires_grad=True)

    def forward(self, input):
        if self.activation_value_aware:
            input = SignTwoOrders.apply(input)
        else:
            input = SignSTE.apply(input)

        # self.bopweight = self.weight.view(self.shape)
        bopweight = self.weight
        bopweight.requires_grad_()

        input = F.pad(input, (self.padding[0], self.padding[0], self.padding[1], self.padding[1]),
                      mode='constant', value=-1)
        out = F.conv2d(input, bopweight, self.bias, self.stride, 0, self.dilation, self.groups)
        return out
CONV_LAYERS.register_module('BinarizeConv2dBop',module=BinarizeConv2dBop)
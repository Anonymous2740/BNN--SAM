from torch.autograd import Function 
import torch
import torch.nn.functional as F
import torch.nn as nn

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
        #self.bias的值等于0，增加这个bias的作用在哪里，起到位移的作用了吗？而且是一个通道一个偏移值  
        # 这应该只是一个初始值，后面训练的时候会随着梯度逐步迭代
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

    @staticmethod #类似于 y=x
    def backward(ctx, grad_output):
        grad_input = grad_output.new_empty(grad_output.size())
        grad_input.copy_(grad_output)
        return grad_input

# class HardBinaryConv(nn.Module):
#     def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
#         super(HardBinaryConv, self).__init__()
#         self.stride = stride
#         self.padding = padding
#         self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
#         self.shape = (out_chn, in_chn, kernel_size, kernel_size)
#         self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)

#     def forward(self, x):
#         real_weights = self.weights.view(self.shape)
#         scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
#         #print(scaling_factor, flush=True)
#         scaling_factor = scaling_factor.detach()
#         binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
#         cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
#         binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
#         #print(binary_weights, flush=True)
#         y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

#         return y


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
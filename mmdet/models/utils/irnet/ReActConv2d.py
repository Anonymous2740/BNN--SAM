import torch.nn as nn
import torch.nn.functional as F
from .binaryfunction import *
from torch.nn import Module, Parameter
import torch
import math


class ReActConv2d(Module):

    def __init__(self, conv):
        super(ReActConv2d, self).__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.k = torch.tensor([10]).float().to(device)
        # self.t = torch.tensor([0.1]).float().to(device)
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = Parameter(conv.weight.data.clone())
        
        # #new added
        self.move0 = LearnableBias(self.in_channels)
        # self.move1 = LearnableBias(self.in_channels)
        # self.newSTE = SignSTEWeight(self.out_channels)
        # self.binary_conv = HardBinaryConv(self.in_channels, self.out_channels, self.kernel_size,self.stride,self.padding)

        ####
        try:
            self.bias = Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, input):
        w = self.weight
        a = input


        if self.in_channels == 3:
            output = F.conv2d(a, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            """IRNet"""
            # # # 通过减均值，除方差的过程，可以使原始参数分布具有均值为0方差为1的属性。在此分布的基础上进行二值量化可以很好的平衡+1和-1的个数对称，使得层的二值激活值信息熵达到最大化，从而使二值的信息表征能力增强
            # bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)            #减均值
            # bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)         #除方差

            #w = bw

            # bw = BinaryQuantize().apply(w, self.k, self.t)
            # ba = BinaryQuantize().apply(a, self.k, self.t)

            """Clamp"""
            # # # 通过减均值，除方差的过程，可以使原始参数分布具有均值为0方差为1的属性。在此分布的基础上进行二值量化可以很好的平衡+1和-1的个数对称，使得层的二值激活值信息熵达到最大化，从而使二值的信息表征能力增强
            # bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)            #减均值
            # bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)         #除方差
            # # 引入了整数移位标量s，扩展了二值权重的表示能力
            # bw = w
            # scaling_factor = torch.mean(torch.mean(torch.mean(abs(w),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
            # # # sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()
            # bw = binarize_w(bw)
            # bw = bw*scaling_factor.detach()
            # ba = binarize_a(a)#STE

            # """Bidet"""
            # #ba = SignTwoOrders.apply(ba)
            # ba = BinaryActivation(a)
            # # bw = BinaryActivation(w)
            # bw = SignSTEWeight().apply(w)
            # scaling_factor = torch.mean(torch.mean(torch.mean(abs(w),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
            # scaling_factor = scaling_factor.detach()
            # bw = bw * scaling_factor
            
            # output = F.conv2d(ba, bw, self.bias, self.stride, self.padding, self.dilation, self.groups)


            # """original bidet"""
            # input = SignTwoOrders.apply(input)
            # subed_weight = self.weight
            # self.weight_bin_tensor = subed_weight.abs(). \
            #                              mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True) \
            #                          * binarize_w(subed_weight)#SignSTEWeight.apply(subed_weight)
            # self.weight_bin_tensor.requires_grad_()
            # input = F.pad(input, (self.padding[0], self.padding[0], self.padding[1], self.padding[1]),
            #             mode='constant', value=-1)
            # out = F.conv2d(input, self.weight_bin_tensor, self.bias, self.stride, 0, self.dilation, self.groups)
            # self.weight=self.weight_bin_tensor
            # """Bi-real"""
            # input = BinaryActivation.apply(input)
            # subed_weight = self.weight

            # out = self.binary_conv(subed_weight)

            """Re_Act"""
            ma = self.move0(a)
            ba = BinaryActivation(ma)
      
            bw = SignSTEWeight().apply(w)
            
   
            scaling_factor = torch.mean(torch.mean(torch.mean(abs(w),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
            scaling_factor = scaling_factor.detach()
            bw = bw * scaling_factor
            
            out = F.conv2d(ba, bw, self.bias, self.stride, self.padding, self.dilation, self.groups)



        return out


# IR_Net是对权重的分布做了改变,React_Net是对激活的分布做了改变

#     #React_Net
# def forward(self, input):
#     w = self.weight
#     a = input
#     if self.in_channels == 3:
#         output = F.conv2d(a, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
#     else:
#         #对input做一个bias的横向偏移
#         out = self.move0(a)
#         #对偏移后的input进行二值化
#         ba = self.binary_activation(out)
#         #对input的权重进行二值化，并进行卷积操作---这样是对随机化权重做的操作，而不是self.weight
#         # output = self.binary_conv(w)
#         #通过减均值，除方差的过程，可以使原始参数分布具有均值为0方差为1的属性。在此分布的基础上进行二值量化可以很好的平衡+1和-1的个数对称，使得层的二值激活值信息熵达到最大化，从而使二值的信息表征能力增强
#         bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)            #减均值
#         bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)         #除方差
#         #bw = BinaryQuantize().apply(bw, self.k, self.t)

#         # #通过减均值，除方差的过程，可以使原始参数分布具有均值为0方差为1的属性。在此分布的基础上进行二值量化可以很好的平衡+1和-1的个数对称，使得层的二值激活值信息熵达到最大化，从而使二值的信息表征能力增强
#         # bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)            #减均值
#         # bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)         #除方差
#         # # 引入了整数移位标量s，扩展了二值权重的表示能力
#         sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()
#         # ba = binarize(out, self.beta,self.alpha)#self.beta和self.beta并没有用到
#         bw = binarize(bw, self.beta1,self.alpha1) #对权重进行二值化
#         bw = bw*sw
#         # ba = binarize(a, self.k, self.t)
#         # bw = binarize(bw, self.k, self.t)
#         # bw = BinaryQuantize().apply(bw, self.k, self.t)
#         # ba = BinaryQuantize().apply(a, self.k, self.t)
#         # scaling_factor = torch.mean(torch.mean(torch.mean(abs(w),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
#         # scaling_factor = scaling_factor.detach()#用scaling的效果不太好吗？
#         # bw = bw * scaling_factor
        
#         output = F.conv2d(ba, bw, self.bias, self.stride, self.padding, self.dilation, self.groups)
#     return output       


# class Scale_Hardtanh(Module):
#     def __init__(self):
#         super(Scale_Hardtanh, self).__init__()
#         self.alpha = Parameter(torch.tensor(1.0,dtype=torch.float32))
#         self.beta = Parameter(torch.tensor(-1.0,dtype=torch.float32))
#     def forward(self, input):
#         m1 = input > self.alpha
#         m2 = input < self.beta
#         output = m1.float()*self.alpha + m2.float()*self.beta + input*(1.0-m1.float())*(1.0-m2.float())

#         return output

#如何在激活这里加learn_bias呢，因为learn_bias函数里面有个参数是out_channels,而代码中的Leaky Relu默认=1

# class Scale_Hardtanh(Module):
#     def __init__(self):
#         super(Scale_Hardtanh, self).__init__()
#         self.alpha = Parameter(torch.tensor(1.0,dtype=torch.float32))
#         self.beta = Parameter(torch.tensor(0.0,dtype=torch.float32))
#         # self.beta1 = Parameter(torch.tensor(0.0,dtype=torch.float32))
#     def forward(self, input):
#         input = self.alpha*(input+self.beta)
#         output = F.hardtanh(input)
#         # # new added one line
#         # output += self.beta1
#         ##
#         return output



class Scale_Hardtanh(Module):
    def __init__(self,out_chn):
        super(Scale_Hardtanh, self).__init__()
        # self.k = nn.Parameter(torch.ones(1,out_chn,1,1),required_grad=True)
        # self.b1 = nn.Parameter(torch.zeros(1,out_chn,1,1),required_grad=True)
        # self.b2 = nn.Parameter(torch.zeros(1,out_chn,1,1),required_grad=True)
        # self.k = Parameter(torch.tensor(1.0,dtype=torch.float32))

        self.b1 = Parameter(torch.tensor(0.0,dtype=torch.float32))
        self.b2 = Parameter(torch.tensor(0.0,dtype=torch.float32))
        self.k1 = Parameter(torch.tensor(1.0,dtype=torch.float32))
        self.k2 = Parameter(torch.tensor(0.1,dtype=torch.float32))
               

        # self.prelu = nn.PReLU(out_chn)
    def forward(self,input):
        # input = self.k*(input+self.b1)

        input = input + self.b1
        mask = input > 0
        output = input*mask.float()*self.k1 + input*(1.0-mask.float())*self.k2 + self.b2

        # output = self.prelu(input) + self.b2
        return output

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
        
class prelu(Module):
    def  __init__(self,out_chn):
        super(prelu,self).__init__()

        self.move1 = LearnableBias(out_chn)
        self.prelu = nn.PReLU(out_chn)
        self.move2 = LearnableBias(out_chn)

    def forward(self,x):
        x = self.move1(x)
        x = self.prelu(x)
        x = self.move2(x)

        return x

class Cprelu(Module):
    def  __init__(self,out_chn):
        super(Cprelu,self).__init__()
        # self.k1 = nn.Parameter(torch.tensor(0.4,dtype=torch.float32),requires_grad=True)
        # self.k2 = nn.Parameter(torch.tensor(1.0,dtype=torch.float32),requires_grad=True)
        self.k1 = nn.Parameter(torch.tensor(0.4,dtype=torch.float32).expand(1,out_chn,1,1),requires_grad=True)
        self.k2 = nn.Parameter(torch.ones(1,out_chn,1,1),requires_grad=True)
        self.move1 = LearnableBias(out_chn)
        self.prelu = nn.PReLU(out_chn)
        self.move2 = LearnableBias(out_chn)

    def forward(self,x):
        x = self.move1(x)
        x = self.hand_prelu(x,self.k1,self.k2)
        x = self.move2(x)

        return x

def hand_prelu(x,k1,k2):
    mask1= x < 0
    mask2 = x > 0
    out1 = (k1*x)*(mask1.type(torch.float32))
    out2 = (k2*x)*(mask2.type(torch.floate32)) + out1
    return out2


class Hardtanh(Module):
    def __init__(self):
        super(Hardtanh, self).__init__()
    def forward(self, input):
        output = F.hardtanh(input)
        return output


# class Scale_Hardtanh(Module):
#     def __init__(self):
#         super(Scale_Hardtanh, self).__init__()
#         self.alpha = Parameter(torch.tensor(1.0,dtype=torch.float32))
#         self.beta1 = Parameter(torch.tensor(0.0,dtype=torch.float32))
#         self.beta2 = Parameter(torch.tensor(0.0,dtype=torch.float32))
#     def forward(self, input):
#         input = (input+self.beta1) * self.alpha + self.beta2
#         output = F.hardtanh(input)
#         return output

# class Scale_Hardtanh(Module):
#     def __init__(self):
#         super(Scale_Hardtanh, self).__init__()
#         self.alpha = Parameter(torch.tensor(1.0,dtype=torch.float32))
#         self.beta1 = Parameter(torch.tensor(-1.0,dtype=torch.float32))
#         self.beta2 = Parameter(torch.tensor(1.0,dtype=torch.float32))
#         # self.beta = Parameter(torch.tensor(0.0,dtype=torch.float32))
#     def forward(self, input):
#         # input = self.alpha*input
#         m1 = input < self.beta1
#         m2 = input > self.beta2
#         output = m1.float()*self.beta1 + m2.float()*self.beta2 + (1.0-m1.float())*(1.0-m2.float())*input
#         return output
class Scale_RELU(Module):
    def __init__(self):
        super(Scale_RELU, self).__init__()
        self.k = Parameter(torch.tensor(1.0,dtype=torch.float32))#k,b参数就是这样固定的吗？不会逐步反向传播学习吗？
        self.b = Parameter(torch.tensor(0.0,dtype=torch.float32))
    def forward(self, input):
        input = self.k*(input+self.b)
        output = F.relu(input)
        return output

class PACT(Module):
    def __init__(self):
        super(PACT, self).__init__()
        self.alpha = Parameter(torch.tensor(10.0,dtype=torch.float32))
    def forward(self, input):
        output =  torch.abs(input) - torch.abs(input-self.alpha) + self.alpha 
        return output

class Scale_PACT(Module):
    def __init__(self):
        super(Scale_PACT, self).__init__()
        self.k = Parameter(torch.tensor(1.0,dtype=torch.float32))
        self.b = Parameter(torch.tensor(0.0,dtype=torch.float32))
        self.alpha = Parameter(torch.tensor(10.0,dtype=torch.float32))
    def forward(self, input):
        input = self.k*(input+self.b)
        output =  0.5*(torch.abs(input) - torch.abs(input-self.alpha) + self.alpha)
        return output

class PACT_PLUS(Module):
    def __init__(self):
        super(PACT_PLUS, self).__init__()
        self.k = Parameter(torch.tensor(1.0,dtype=torch.float32))
        self.alpha = Parameter(torch.tensor(10.0,dtype=torch.float32))
        self.beta = Parameter(torch.tensor(-10.0,dtype=torch.float32))
        # self.beta = 0.0
    def forward(self, input):
        output =  0.5*(torch.abs(self.k * input-self.beta) - torch.abs(self.k * input-self.alpha) + self.alpha + self.beta)
        # output =  0.5*(torch.abs(input-self.beta) - torch.abs(input-self.alpha) + self.alpha + self.beta)
        return output
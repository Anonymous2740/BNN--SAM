import torch
import numpy as np
import torch.nn as nn
import copy as copy
from .IRConv2d import *
import torch.nn.functional as F

# from model_adj.Conv_trans import *

cnt = 0
def IR_model(model):
    global cnt
    for name, module in (model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = IR_model(model=module)
            # darknet是18层卷积 + detection part的5层卷积
        if type(module) == nn.Conv2d:
        # if type(module) == nn.Conv2d or type(module) == Conv2d_kernel_chag:#把1*1卷积换为3*3卷积之后，也要进行二值化
            chn = module.out_channels 
            kernelsize = module.kernel_size
            # if module.in_channels == 3  or cnt <= 2: 
            # if module.in_channels == 3  or cnt <= 22 or cnt == 73:
            # if module.in_channels == 3  or cnt > 51 or cnt == 1 or cnt == 0 or module.kernel_size == (1,1):#Yolo_v3
            # if module.in_channels == 3  or cnt <= 5 or module.kernel_size == (1,1):

            # if module.in_channels == 3 or cnt > 13 or cnt == 0 or module.kernel_size == (1,1):#SSD binarize bn
            if module.in_channels == 3  or cnt ==22 or cnt>30  or module.kernel_size == (1,1):#SSD binarize bn+extra#18，22 #or cnt==21 
                if cnt == 23 or cnt == 25 or cnt == 27 or cnt == 29:
                    irnet_conv2d = IRConv2d(module)                   
                else:
                    irnet_conv2d = module

            else:
                irnet_conv2d = IRConv2d(module)


            # # if cnt == 5:
            # if cnt == 50:
            #     irnet_conv2d = IRConv2d(module)
            # else:
            #     irnet_conv2d = module

            model._modules[name] = irnet_conv2d
            cnt += 1
            print(cnt)

        # elif type(module) == nn.Linear:
        #     dsq_Linear = DorefaLinear(in_features=module.in_features, out_features=module.out_features, bias=module.bias is not None, 
        #                                 w_bit = 1, a_bit=32,  QInput = True, bSetQ = True)
        #     model._modules[name] = dsq_Linear

        # elif type(module) == nn.ReLU or type(module) == nn.ReLU6 or type(module) == nn.LeakyReLU :
            
        # #     # if cnt <= 22 or cnt == 74 or cnt == 73:#非二值化的层不进行替换
        # #     # if cnt > 52 or cnt == 2 or cnt == 1 or kernelsize == (1,1): #Yolo_v3

            
        # # #     if cnt > 14 or cnt ==1 :#SSD
        #     if cnt == 22 or cnt == 23 or cnt ==1 :#SSD
        #         activation = module
        # # #         # activation = nn.Sequential()

        #     else:

        #         activation =  prelu(chn)
        # #         # activation =  Scale_Hardtanh(chn)
        # # #         # activation = Scale_PACT()
        # # #         # activation =  Hardtanh()
        # # #         # activation = nn.Sequential()
        #     model._modules[name] = activation

            # if cnt == 50:
            #     activation =  Scale_Hardtanh()
            # #     activation = module
            # #     # activation = nn.Sequential()
            # else:
            #     activation = module
            # #     activation =  Scale_Hardtanh()
            # #     # activation = Scale_PACT()
            # #     # activation =  Hardtanh()
            # model._modules[name] = activation

            # cnt += 1
            # print(cnt)
    return model





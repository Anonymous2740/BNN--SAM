# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import logging

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES


@BACKBONES.register_module()
class DarkNet19(nn.Module):
    def __init__(self, num_classes=1000):
        print("Initializing the darknet19 network ......")

        super(DarkNet19, self).__init__()
        # backbone network : DarkNet-19
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            nn.MaxPool2d((2, 2), 2),
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 64, 1),
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 8, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 128, 1),
            Conv_BN_LeakyReLU(128, 256, 3, 1),
        )

        # output : stride = 16, c = 512
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
        )

        # output : stride = 32, c = 1024
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1)
        )


    def convert_keys(self, model, baseline):
        '''
        rename the baseline's key to model's name
        e.g.
            baseline_ckpt = torch.load(args.baseline, map_location=device)
            model.load_state_dict(convert_keys(model, baseline_ckpt))
        '''
        from collections import OrderedDict

        baseline_state_dict = OrderedDict()
        model_key = list(model.state_dict().keys())
        # print("model_key:")
        # print(model_key)
        baseline_key = list(baseline['state_dict'].keys())
        # print("baseline_key:")
        # print(baseline_key)
        # import pdb
        # pdb.set_trace()
        if(len(model_key)!=len(baseline_key)):
            print("ERROR: The model and the baseline DO NOT MATCH")
            pdb.set_trace()
            exit()
        else:
            for i in range(len(model_key)):
                baseline_state_dict[model_key[i]] = baseline['state_dict'][baseline_key[i]]
        return baseline_state_dict

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            # pretrained = torch.load(pretrained,map_location="cuda:0")
            logger = logging.getLogger()
            # from collections import OrderedDict
            # baseline_state_dict = OrderedDict()
            # model_key = list(self.state_dict().keys())
            # # baseline_key = list(pretrained['state_dict'].keys())
            # baseline_key = list(pretrained.keys())
            # # print("model的keys为：")
            # # print(model_dict.keys())
            # j = int(0)
            # for i in range(len(model_key)):
            #     if model_key[i][-4:]==baseline_key[j][-4:]:
            #         # baseline_state_dict[model_key[i]] = pretrained['state_dict'][baseline_key[j]].reshape_as(self.state_dict()[model_key[i]])
            #         baseline_state_dict[model_key[i]] = pretrained[baseline_key[j]]#.reshape_as(self.state_dict()[model_key[i]])
            #         j = j+1
            #     else:
            #         baseline_state_dict[model_key[i]]=torch.zeros(self.state_dict()[model_key[i]].size())
            # print("baseline_state_dict的keys为")
            # print(baseline_state_dict.keys())
            # torch.save(baseline_state_dict,'/home/ic611/workspace/hanhan/mmdetection/checkpoints/darknet19_convert.pth')
            # # pretrained = torch.load('/home/ic611/workspace/hanhan/mmdetection/checkpoints/darknet19.pt')
            # # # pretrained_dict = pretraiaz/
            # # print("pretrained的keys为：")
            # # #print(pretrained_dict.keys())?
            # # print(pretrained['state_dict'].keys())
            # self.load_state_dict(baseline_state_dict,strict=True)
            load_checkpoint(self, pretrained, strict=False, logger=logger)#,map_location="cuda:0")
            print("Load the pretrained model!")
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        C_4 = self.conv_4(x)
        C_5 = self.conv_5(self.maxpool_4(C_4))
        C_6 = self.conv_6(self.maxpool_5(C_5))


        return C_4, C_5, C_6

    def train(self, mode=True):
        super(DarkNet19, self).train(mode)
    

class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)

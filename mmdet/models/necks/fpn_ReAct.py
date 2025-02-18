import warnings

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

from ..builder import NECKS

from ..utils.irnet.binaryfunction import BinarizeConv2d
from ..utils.irnet.ReActConv2d_new import ReActConv2d,prelu

@NECKS.register_module()
class FPN_ReAct(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=dict(type='ReActConv2d'),
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN_ReAct, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        # 

        for i in range(self.start_level, self.backbone_end_level):
            # 水平卷积
            # l_conv = ConvModule(
            #     in_channels[i],
            #     out_channels,
            #     1,
            #     conv_cfg=conv_cfg,
            #     norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
            #     act_cfg=act_cfg,
            #     inplace=False)
            """
            原来是1*1卷积，我把它改成了3*3的1 bit conv
            """
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
           
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            
       
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

            # self.encoder_att_extra = nn.ModuleList([nn.ModuleList([self.att_layer_extra([out_channels, out_channels, out_channels])])])#in_channels[i]
            # for j in range(2):
            #     if j < 1:
            #         self.encoder_att_extra.append(nn.ModuleList([self.att_layer_extra([out_channels, out_channels, out_channels])]))#in_channels[i]
            
            # self.encoder_att_block_extra = nn.ModuleList([self.conv_layer([out_channels,out_channels])])

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # def att_layer_extra(self, channel):
    #     att_block = nn.Sequential(
    #         nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
    #         # BinarizeConv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
    #         nn.BatchNorm2d(channel[1]),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, stride=1, padding=0),
    #         # BinarizeConv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, stride=2, padding=0),
    #         nn.BatchNorm2d(channel[2]),
    #         nn.Sigmoid(),
    #     )
    #     return att_block    

    # def conv_layer(self, channel, pred=False):
    #     if not pred:
    #         conv_block = nn.Sequential(
    #             ReActConv2d(nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1)),
    #             # BinarizeConv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
    #             # BinarizeConv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=2, dilation=2),
    #             # BinarizeConv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
    #             nn.BatchNorm2d(num_features=channel[1]),
    #             prelu,# nn.ReLU(inplace=True),
    #         )
    #     else:
    #         conv_block = nn.Sequential(
    #             nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
    #             nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
    #         )
    #     return conv_block

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # # lateral的输出作为1*1 conv,1*1 conv, sigmoid的输入，生成mask,再和3*3的conv输出做elt_wise乘积
        # # 由原来的一个outs试图变为两个outs,每个task都会输出一个outs
        # outs_attention_cls = []
        # outs_attention_loc = []
        # for i in range(used_backbone_levels):
        #     x_lateral = laterals[i]
        #     x_encoder_cls = self.encoder_att_extra[0][0](x_lateral) # cls mask
        #     x_encoder_loc = self.encoder_att_extra[1][0](x_lateral) # loc mask
        #     x_fpn_cls = self.fpn_convs[i](x_lateral) * x_encoder_cls # the need feature
        #     x_fpn_loc = self.fpn_convs[i](x_lateral) * x_encoder_loc # the need feature
        #     outs_cls = self.encoder_att_block_extra[0](x_fpn_cls)
        #     outs_loc = self.encoder_att_block_extra[0](x_fpn_loc)
        #     outs_attention_cls.append(outs_cls)
        #     outs_attention_loc.append(outs_loc)

        # # Faster_RCNN的head部分也要改，outs_attention_cls输送给分类Head，outs_attention_loc输送给定位Head    
        # # 分别有RPN Head和RoI Head, RPN Head里本身就有rpn_cls和rpn_reg,从这里就要specific_task吗
        # # RoI Head里面有RoIAlign/RoIPool,还有两层FC

        # return tuple(outs_attention_cls), tuple(outs_attention_loc), tuple(outs)

        ## outs = tuple(outs_attention_cls,outs_attention_loc)

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

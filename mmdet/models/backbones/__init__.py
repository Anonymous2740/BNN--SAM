from .darknet import Darknet
from .darknet_19 import DarkNet19
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .hourglass import HourglassNet
from .hrnet import HRNet
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .ssd_vgg_ReAct import SSDVGG_ReAct
from .ssd_vgg_BiReal import SSDVGG_BiReal
from .trident_resnet import TridentResNet

from .ssd_vgg_BiReal_Attention import SSDVGG_BiReal_Attention

from .ssd_vgg_ReAct_Attention import SSDVGG_ReAct_Attention

from .resnet_BiReal import ResNet_BiReal

from .resnet_ReAct import ResNet_ReAct
from .ssd_vgg_ReAct_Attention_backbone import SSDVGG_ReAct_Attention_Backbone
from .ssd_vgg_ReAct_Attention_backbone_s8 import SSDVGG_ReAct_Attention_Backbone_s8
from .ssd_vgg_ReAct_Attention_backbone_s16 import SSDVGG_ReAct_Attention_Backbone_s16 
from .ssd_vgg_BiReal_latent_w import SSDVGG_BiReal_latent_w
from .repvgg import RepVGG

from .ssd_vgg_BiReal_Bop import SSDVGG_BiReal_Bop
from .resnet_BiReal_Bop import ResNet_BiReal_Bop

from .ssd_vgg_ReAct_Attention_Relu import SSDVGG_ReAct_Attention_Relu
from .ssd_vgg_ReAct_Attention_Tanh import SSDVGG_ReAct_Attention_Tanh
from .ssd_vgg_BiReal_channel_attention import SSDVGG_BiReal_channel_attention

from .ssd_vgg_ReAct_channel_attention import SSDVGG_ReAct_channel_attention
from .ssd_vgg_ReAct_Attention_One_TFD_branch import SSDVGG_ReAct_Attention_One_TFD_branch
from .ssd_vgg_ReAct_Attention_no_mask import SSDVGG_ReAct_Attention_no_mask
from .ssd_vgg_ReAct_Attention_conv3_specific import SSDVGG_ReAct_Attention_conv3_specific
from .ssd_vgg_ReAct_Attention_One_TFD_branch_no_mask import SSDVGG_ReAct_Attention_One_TFD_branch_no_mask

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net',
    'HourglassNet', 'DetectoRS_ResNet', 'DetectoRS_ResNeXt', 'Darknet', 'DarkNet19',
    'ResNeSt', 'TridentResNet','SSDVGG', 'SSDVGG_ReAct','SSDVGG_BiReal',#'ResNet_BiReal'
    'SSDVGG_BiReal_Attention','ResNet_BiReal','ResNet_ReAct','SSDVGG_ReAct_Attention_Backbone',
    'SSDVGG_ReAct_Attention_Backbone_s8','SSDVGG_ReAct_Attention_Backbone_s16','RepVGG','SSDVGG_BiReal_latent_w',
    'SSDVGG_BiReal_Bop','ResNet_BiReal_Bop','SSDVGG_ReAct_Attention_Relu','SSDVGG_ReAct_Attention_Tanh',
    'SSDVGG_BiReal_channel_attention','SSDVGG_ReAct_channel_attention','SSDVGG_ReAct_Attention_One_TFD_branch',
    'SSDVGG_ReAct_Attention_no_mask','SSDVGG_ReAct_Attention_conv3_specific',
    'SSDVGG_ReAct_Attention_One_TFD_branch_no_mask'
]

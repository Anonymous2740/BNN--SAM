import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .utils import weighted_loss
import torch
#https://zhuanlan.zhihu.com/p/265704145
@LOSSES.register_module()
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1,reduction='none'):#reduction='mean'#eps=0.1
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)
    # def forward(self, output, target):
    #     c = output.size()[-1]

    #     log_preds = F.log_softmax(output, dim=-1)
    #     max_list = log_preds.argmax(axis=-1)
    #     log_preds_new = torch.zeros_like(log_preds)
    #     for index in range(len(max_list)) : 
    #         if (max_list!=target)[index] == True:
    #             log_preds_new[index][target]=log_preds[index][target]*1.2
    #     if self.reduction=='sum':
    #         loss = -log_preds_new.sum()
    #     else:
    #         loss = -log_preds_new.sum(dim=-1)
    #         if self.reduction=='mean':
    #             loss = loss.mean()
    #     return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds_new, target, reduction=self.reduction)
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES

@LOSSES.register_module()
class loss_kl(nn.Module):
    def __init__(self, T = 1.0):
        super(loss_kl, self).__init__()
        self.T = T
        
    def forward(outputs, teacher_outputs, T):
        kl_loss = (T * T) * nn.KLDivLoss(size_average=False)(F.log_softmax(outputs / T),
                                                            F.softmax(teacher_outputs / T)) / outputs.shape[0]
        return kl_loss
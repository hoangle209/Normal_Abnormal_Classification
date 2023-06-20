import torch.nn as nn
import torch.nn.functional as F
# import torch
# from utils.general import check_version

# def smartCrossEntropyLoss(label_smoothing=0.0):
#     # Returns nn.CrossEntropyLoss with label smoothing enabled for torch>=1.10.0
#     if check_version(torch.__version__, '1.10.0'):
#         return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
#     if label_smoothing > 0:
#         print(f'WARNING ⚠️ label smoothing {label_smoothing} requires torch>=1.10.0')
#     return nn.CrossEntropyLoss()

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = F.binary_cross_entropy()
    
    def forward(self, out, batch):
        b = out.size(0)
        loss = self.bce(out, batch)

        return loss.sum() / b


class CELoss(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.ce = F.cross_entropy(label_smoothing=label_smoothing) 
    
    def forward(self, out, batch):
        b = out.szie(0)
        loss = self.ce(out, batch)

        return loss.sum() / b




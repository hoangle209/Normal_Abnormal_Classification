import torch.nn as nn
import torch.nn.functional as F
import torch
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
        self.bce = F.binary_cross_entropy(reduction='none')
    
    def forward(self, out, batch):
        b = out.size(0)
        loss = self.bce(out, batch)

        return loss.sum() / b


class CELoss(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none') 
    
    def forward(self, out, batch):
        b = out.size(0)
        loss = self.ce(out, batch)

        return loss.sum() / b



def focal_loss(out, batch, alpha=1., gamma=2.):
    b = out.size(0)

    max_value, _ = batch.max(axis=-1, keepdim=True)
    pos_ind = out.eq(max_value).float()
    neg_ind = out.lt(max_value).float()

    neg_weight = torch.pow(1-batch, 4)

    loss = 0
    pos_loss = torch.pow(1-out, gamma) * torch.log(out) * pos_ind
    neg_loss = torch.pow(1-out, gamma) * torch.log(out) * neg_ind * neg_weight
    loss = loss - (pos_loss + neg_loss)

    return loss.sum() / b


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_loss = focal_loss
    
    def forward(self, out, batch):
        loss = self.focal_loss(out, batch)
        return loss
import torch
import torch.nn as nn


class Yolov5Backbone(nn.Module):
    def __init__(self,
                 model=None, # TODO Loading model from source
                 requires_grad=True,
                 path = None,
                 cutoff=10):
        super().__init__()
        
        # DetectMultibackend -> DetectionModel -> Sequential 
        model = model.model.model
        model.model = model.model[:cutoff] # backbone 
        
        # track grad of model
        if requires_grad:
            for p in model.model.parameters():
                p.requires_grad = True

        m = model.model[-1]
        self.ch = m.conv.out_channels if hasattr(m, 'conv') else m.cv2.conv.out_channels  # ch out module
        self.stride = model.stride

        model = [l for l in model.model.children()]
        self.model = torch.nn.Sequential(*model)
        
        # self.__dict__.update(locals())
    
    def forward(self, x):
        '''
            param x: mxb x C x H x W
        '''
        return self.model(x)
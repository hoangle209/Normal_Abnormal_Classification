import torch
import torch.nn as nn

class ClassificationModel(nn.Module):
    def __init__(self, arch=None, att_layer=None, ch=None, nums_in=5):
        '''
            param model: model architecture
            param att_layer: attention layer
            param ch: output channel of _base model
            param nums_in: total number of input images
        '''
        super().__init__()
        self.rgb_base = arch[0]
        self.flow_base = arch[1]
        self.att_layer = att_layer

        self.nums_rgb = nums_in//2 + 1
        self.nums_flow = nums_in - self.nums_rgb 

        self.ch = self.rgb_base.ch

        self.conv_rgb = nn.Conv2d(self.ch*self.nums_rgb, self.ch, kernel_size=1, bias=False)
        self.conv_flow = nn.Conv2d(self.ch*self.nums_flow, self.ch, kernel_size=1, bias=False)
    
    def forward(self, x):
        '''
            param x: b x c x nums x h x w
        '''
        b, c, _, h, w = x.size()

        flow = (x[:, :, :self.nums_flow]).reshape(b*self.nums_flow, c, h, w).contiguous()
        rgb = (x[:, :, self.nums_flow: ]).reshape(b*self.nums_rgb, c, h, w).contiguous()

        rgb_features = self.rgb_base(rgb)
        rgb_features = rgb_features.view(b, self.ch*self.nums_rgb, rgb_features.size(2), rgb_features.size(3))
        rgb_features = self.conv_rgb(rgb_features)

        flow_features = self.flow_base(flow)
        flow_features = flow_features.view(b, self.ch*self.nums_flow, flow_features.size(2), flow_features.size(3))
        flow_features = self.conv_flow(flow_features)

        feat = [rgb_features, flow_features]
        feat = self.att_layer(feat) # b x c1 x h x w

        return feat


class Classify(nn.Module):
    def __init__(self, arch=None, att_layer=None, c1=1024, c2=None, nums_in=5, nc=1, **kwargs):
        super().__init__()
        self.classify_model = ClassificationModel(arch=arch, att_layer=att_layer, nums_in=nums_in)
        self.ch = self.classify_model.ch
        self.pool_1 = nn.AdaptiveAvgPool2d(1)
        self.linear_1 = nn.Linear(2*self.ch, nc)

    def forward(self, x):
        x = self.classify_model(x)
        x = self.pool_1(x)
        x = x.squeeze()
        x = self.linear_1(x)

        return x
    


        









import torch 
import torch.nn as nn

class CustomAttLayer(nn.Module):
    in_ch = 512

    def __init__(self, c1, c2=None):
        super.__init__()
        
        # self.conv1 = nn.Conv2d(c1, 512, 1, bias=False)
        # self.conv2 = nn.Conv2d(c2, 512, 1, bias=False)
        self.conv = nn.Conv2d(c1, c1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # rgb, flow = x
        # rgb = self.conv1(rgb)
        # flow = self.conv2(flow)

        if isinstance(x, list):
            x = torch.cat(x, dim=1)

        feat = self.conv(x)
        feat = self.sigmoid(feat)
        feat = x*feat

        return feat



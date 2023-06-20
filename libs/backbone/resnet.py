import torch
import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, depth=50):
        super().__init__()
        assert depth in [18, 34, 50, 101, 152]
        self.depth = depth

        if self.depth==18:
            resnet = models.resnet18(weights='DEFAULT')
        elif self.depth==34:
            resnet = models.resnet34(weights='DEFAULT')
        elif self.depth==50:
            resnet = models.resnet50(weights='DEFAULT')
        elif self.depth==101:
            resnet = models.resnet101(weights='DEFAULT')
        elif self.depth==152:
            resnet = models.resnet152(weights='DEFAULT')

        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        last_conv = [x for x in self.resnet[-1][-1].modules() if isinstance(x, nn.Conv2d)]
        self.ch = last_conv[-1].out_channels # output channel of model
    
    def forward(self, x):
        return self.resnet(x)

if __name__ == '__main__':
    r = ResNet(50)
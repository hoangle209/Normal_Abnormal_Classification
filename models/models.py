import torch
import torch.nn as nn


class Yolov5Backbone(nn.Module):
    def __init__(self,
                 model=None, # TODO Loading model from source
                 requires_grad=True,
                 path = None,
                 cutoff=10):
        super().__init__()
        if path is not None:
            model = torch.hub.load("ultralytics/yolov5", "custom", path=path)  # or yolov5n - yolov5x6, custom
        else:
            model = torch.hub.load("ultralytics/yolov5", "yolov5x")

        # DetectMultibackend -> DetectionModel -> Sequential 
        model = model.model.model
        model.model = model.model[:cutoff] # backbone 
        
        # track grad of model
        if requires_grad:
            for p in model.model.parameters():
                p.requires_grad = True

        m = model.model[-1]
        self.ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        self.stride = model.stride

        model = [l for l in model.model.children()]
        self.model = torch.nn.Sequential(*model)
        
        # self.__dict__.update(locals())
    
    def forward(self, x):
        '''
            param x: mxb x C x H x W
        '''
        return self.model(x)


# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        #2*h*w
        x = self.conv(x)
        #1*h*w
        return self.sigmoid(x)
    
class CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2=None, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)

        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return out


class ClassificationModel(nn.Module):
    def __init__(self, nums_in=5):
        '''
            param nums_in: total number of input images
        '''
        super().__init__()
        self.rgb_base = Yolov5Backbone()
        self.optiF_base = Yolov5Backbone()
        self.c1 = self.rgb_base.ch * nums_in
        self.cbam = CBAM(c1=self.c1)

        self.ch = int(self.rgb_base.ch)
        self.stride = self.rgb_base.stride
    
    def forward(self, x):
        '''
            param x: b x nums x c x h x w
        '''
        b, nums, c, h, w = x.size()
        nums_rgb = nums//2 + 1
        nums_optiF = nums - nums_rgb 

        rgb = (x[:, :nums_rgb]).view(b*nums_rgb, c, h, w)
        optiF = (x[:, nums_rgb: ]).view(b*nums_optiF, c, h, w)

        rgb_features = self.rgb_base(rgb)
        rgb_features = rgb_features.view(b, self.ch*nums_rgb, rgb_features.size(2), rgb_features.size(3))
        optiF_features = self.optiF_base(optiF)
        optiF_features = optiF_features.view(b, self.ch*nums_optiF, optiF_features.size(2), optiF_features.size(3))

        feat = torch.cat([rgb_features, optiF_features], dim=1)

        # conv1 = nn.Conv2d(feat.size(1), 512, kernel_size=1, bias=False)
        # feat = conv1(feat)
        # conv2 = nn.Conv2d(512, self.c1, kernel_size=1, bias=False)
        # feat = conv2(feat)

        feat = self.cbam(feat) # b x c1 x h x w

        return feat


class Classify(nn.Module):
    def __init__(self, c1=None, c2=None, nums_in=5, nc=1):
        super().__init__()
        self.classify_model = ClassificationModel(nums_in=nums_in)
        c1 = self.classify_model.c1
        self.pool_1 = nn.AdaptiveAvgPool2d(1)
        self.linear_1 = nn.Linear(c1, nc)

    def forward(self, x):
        x = self.classify_model(x)
        x = self.pool_1(x)
        x = x.squeeze()
        x = self.linear_1(x)

        return x
    

# if __name__ == '__main__':
#     m = Classify(nums_in=7)
#     x = torch.zeros(1, 7, 3, 224, 224)
#     a = m(x)
#     print(m)
#     def count_parameters(model):
#         return sum(p.numel() for p in model.parameters() if p.requires_grad)
#     a = count_parameters(m)
#     print(a)


        









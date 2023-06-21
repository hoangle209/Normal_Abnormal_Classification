import torch
import torch.nn as nn

from libs.trainer import Trainer
from libs.dataset.dataset import GenericDataset
from libs.models import Classify
from libs.backbone.yolov5 import Yolov5Backbone
from libs.backbone.resnet import ResNet
from libs.attention_layer import cbam, custom
from libs.backbone.utils import load_model, save_model
from opts import opts

import os

def get_arch(opt, **kwargs):
    if opt.arch == 'Yolov5':
        path = kwargs.get('yolo_v5_weight',None)
        if path is None:
            arch_rgb = torch.hub.load("ultralytics/yolov5", "yolov5x")
            arch_flow = torch.hub.load("ultralytics/yolov5", "yolov5x")
        else:
            arch_rgb = torch.hub.load("ultralytics/yolov5", "custom", path=path)
            arch_flow = torch.hub.load("ultralytics/yolov5", "custom", path=path)
        
        arch_rgb = Yolov5Backbone(arch_rgb)
        arch_flow = Yolov5Backbone(arch_flow)
    elif opt.arch == 'Resnet':
        arch_rgb = ResNet(opt.depth)
        arch_flow = ResNet(opt.depth)
    else:
        print(opt.arch)
        raise NotImplementedError
    
    arch = [arch_rgb, arch_flow] 
    return arch


def get_att_layer(opt, **kwargs):
    if opt.att == 'cbam':
        att_layer = cbam.CBAM(c1=kwargs['ch'])
    elif opt.att == 'custom':
        att_layer = custom.CustomAttLayer(kwargs['ch'], kwargs['ch'])
    else:
        raise NotImplementedError

    return att_layer


def get_model(opt, **kwargs):
    arch = get_arch(opt, **kwargs)
    ch = arch[0].ch
    att_layer = get_att_layer(opt, ch=2*ch, **kwargs)

    nc = 2 if opt.loss == 'CE' else 1
    model = Classify(arch=arch, att_layer=att_layer, nums_in=kwargs['nums_in'], nc=nc)
    
    return model


def get_optimizer(opt, model):
    if opt.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    elif opt.optim == 'SGD':
        optimizer = torch.optim.SGD(
        model.parameters(), opt.lr, momentum=0.9, weight_decay=5e-4)
    else:
        assert 0, opt.optim

    return optimizer


def main(opt):
    torch.manual_seed(opt.seed)
    # torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = GenericDataset(opt.path, opt)
    train_loader = torch.utils.data.DataLoader(
        Dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True
    )

    if opt.val > 0:
        Dataset_val = GenericDataset(opt.val_path, opt)
        val_loader = torch.utils.data.DataLoader(
            Dataset_val, batch_size=1, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True, drop_last=True
        )

    # module_list = nn.ModuleList([])

    model = get_model(opt, nums_in=15) # TODO
    optimizer = get_optimizer(opt, model)
    start_epoch = 0
    if opt.load_model != '' and opt.resume:
        model, optimizer, start_epoch = load_model(model, opt.load_model, opt, optimizer)
    elif opt.load_model != '':
        model = load_model(model, opt.load_model, opt)
    
    trainer = Trainer(model, opt, optimizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.set_device(device)

    print('Start training ...')
    print(opt)
    for epoch in range(start_epoch+1, opt.num_epochs+1):
        ret, _ = trainer.train(epoch, train_loader)
        
        if opt.val > 0 and epoch % opt.val:
            with torch.no_grad():
                trainer.val(epoch, val_loader)
        
        save_model(os.path.join(opt.save_path, 'model_last.pth'), epoch, model, optimizer) # save model last


if __name__ == '__main__':
    opt = opts().parse()
    main(opt) 
   




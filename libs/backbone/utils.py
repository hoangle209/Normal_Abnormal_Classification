'''
    partial of this code is referenced from TraDes: https://github.com/JialianW/TraDeS
'''
import torch
from .yolov5 import Yolov5Backbone
from .resnet import ResNet
from ..attention_layer import cbam, custom
from ..models import Classify


def load_model(model, model_path, opt, optimizer=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
    
    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and opt.resume:
        if 'optimizer' in checkpoint:
            # optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = opt.lr
            for step in opt.lr_step:
                    if start_epoch >= step:
                        start_lr *= 0.1
            for param_group in optimizer.param_groups:
                        param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
        
    data = {'epoch': epoch,
            'state_dict': state_dict}
    
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


def get_arch(opt, **kwargs):
    if opt.arch == 'yolov5':
        path = kwargs.get('yolo_v5_weight', None)
        if path is None:
            arch_rgb = torch.hub.load("ultralytics/yolov5", f"{opt.arch}{opt.depth}")
            arch_flow = torch.hub.load("ultralytics/yolov5", f"{opt.arch}{opt.depth}")
        else:
            arch_rgb = torch.hub.load("ultralytics/yolov5", "custom", path=path)
            arch_flow = torch.hub.load("ultralytics/yolov5", "custom", path=path)
        
        arch_rgb = Yolov5Backbone(arch_rgb)
        arch_flow = Yolov5Backbone(arch_flow)
    elif opt.arch == 'resnet':
        arch_rgb = ResNet(int(opt.depth))
        arch_flow = ResNet(int(opt.depth))
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


def create_model(opt, **kwargs):
    arch = get_arch(opt, **kwargs)
    ch = arch[0].ch
    att_layer = get_att_layer(opt, ch=2*ch, **kwargs)

    nc = 2 if opt.loss == 'CE' else 1
    model = Classify(arch=arch, att_layer=att_layer, nums_in=kwargs['nums_in'], nc=nc)

    return model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # model
        self.parser.add_argument('--arch', type=str, default='yolov5', 
                                 help='model architect, yolov5 | resnet | resnet3d')
        self.parser.add_argument('--depth', type=str,
                                 help='model depth')
        self.parser.add_argument('--att', type=str, default='cbam',
                                 help='attention layer module, cbam | custom')
        
        # trainer
        self.parser.add_argument('--batch-size', type=int, default=8)
        self.parser.add_argument('--num-workers', type=int, default=2)
        self.parser.add_argument('--loss', type=str, default='CE', 
                                 help='loss function, should be CE | BCE')
        self.parser.add_argument('--label-smoothing', action='store_true')
        self.parser.add_argument('--optim', type=str, default='SGD')
        self.parser.add_argument('--lr', type=float, default=1.25e-2)
        self.parser.add_argument('--lr-step',type=str, default='')
        self.parser.add_argument('--resume', action='store_true')
        self.parser.add_argument('--save-path', type=str, default='')
        self.parser.add_argument('--load-model', type=str, default='')
        self.parser.add_argument('--num-epochs', type=int, default=250)
        self.parser.add_argument('--input-w', type=int, default=416)
        self.parser.add_argument('--input-h', type=int, default=416)
        self.parser.add_argument('--val', type=int, default=-1,
                                 help='validation period')
        self.parser.add_argument('--save-point', type=str, default='',
                                 help='save checkpoints')
        
        # dataset
        self.parser.add_argument('--dataset', type=str, default='Online',
                                 help='dataset used for training')
        self.parser.add_argument('--path', type=str, default='')
        self.parser.add_argument('--val-path', type=str, default='')
        self.parser.add_argument('--flip', type=float, default=0.5,
                                 help='probability to vertical flip')
        self.parser.add_argument('--rotate', type=float, default=0.3,
                                 help='probability to rotate')
        self.parser.add_argument('--color-aug', type=float, default=0.3,
                                 help='color augmentation probability')
        self.parser.add_argument('--seed', type=int, default=5,
                                 help='seed')
    

    def parse(self):
        opt = self.parser.parse_args()

        # model_step
        if opt.save_path == '':
            _path = os.path.dirname(__file__)
            opt.save_path = os.path.join(_path, 'exp')

            if not os.path.isdir(opt.save_path):
                os.mkdir(opt.save_path)
        
        # model_name
        if opt.arch != 'Yolov5':
            opt.model_name = f'{opt.arch}-{opt.depth}_{opt.att}'
        else:
            opt.model_name = f'{opt.arch}{opt.depth}_{opt.att}'

        # lr_step
        if opt.lr_step != '':
            opt.lr_step = [int(s) for s in opt.lr_step.split(',')]
        else:
            opt.lr_step = []

        #save checkpoint
        if opt.save_point != '':
            opt.save_point = [int(s) for s in opt.save_point.split(',')]
        else:
            opt.save_point = []

        return opt


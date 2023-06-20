from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # model
        self.parser.add_argument('--arch', type=str, default='Yolov5', 
                                 help='model architect, Yolov5 | Resnet | Resnet3d')
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
        
        # dataset
        self.parser.add_argument('--path', type=str, default='')
        self.parser.add_argument('--val_path', type=str, default='')
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

        if opt.save_path == '':
            _path = os.path.dirname(__file__)
            opt.save_path = os.path.join(_path, 'exp')

            if not os.path.isdir(opt.save_path):
                os.mkdir(opt.save_path)
        
        if opt.arch != 'Yolov5':
            opt.arch = opt.arch + opt.depth

        return opt


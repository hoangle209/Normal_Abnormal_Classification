import os
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader
import cv2 as cv
import glob

from ..utils.image import color_aug


class GenericDataset(torch.utils.data.Dataset):
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
    
    std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    
    _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                        dtype=np.float32)
    
    _eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

    label_encoder_bce = {
        'Abnormal': np.float32(0),
        'Normal'  : np.float32(1)
    }

    label_encoder_ce = {
        'Abnormal': [1., 0.],
        'Normal'  : [0., 1.]
    }

    def __init__(self, path=None, opt=None, num_RGB=None, num_OptiF=None, num_Heat=None, split='train'):
        '''
            ../(abnormal | normal)/name_vid/sub_name_vid/image.jpg
        '''
        super().__init__()
        self.split= split
        # path to folder containing images (rgb, flow, ? heatmap)
        self.sub_folder_paths = glob.glob(os.path.join(path, '*', '*')) 
        if path is not None:
            print(f'==> initializing {split} data from {path}')
        
        self.label_len = len(self.label_encoder)
        self.opt = opt

        self.num_rgb = 3 if num_RGB is None else num_RGB
        self.num_optiF = 2 if num_OptiF is None else num_OptiF
        self.num_heat = 1 if num_Heat is None else num_Heat
        self._data_rng = np.random.RandomState(123)


    def __len__(self):
        '''
            return total number of sub_name_vid
        '''
        return len(self.sub_folder_paths)
    

    def __getitem__(self, index):
        '''
            return images, label
        '''
        opt = self.opt

        ret = {}
        # ret['meta'] = {
        #     'mean': self.mean,
        #     'std': self.std
        # }
        sub_folder_path = self.sub_folder_paths[index]
        concat_img, label, img_infor = self.load_input(sub_folder_path, opt)

        ret['image'] = concat_img
        ret['label'] = label
        ret['info'] = img_infor

        return ret
        
    
    def load_input(self, sub_folder_path, opt):
        img_list, label, img_infor = self.load_imgs_and_label(sub_folder_path)
        concat_img = self._concat_input(img_list)

        return concat_img, label, img_infor


    def _concat_input(self, img_list):
        concat_img = self._augment(img_list)

        concat_img = concat_img / 255.
        concat_img = (concat_img - self.mean[None, ...]) / self.std[None, ...]
        concat_img = concat_img.transpose(3, 0, 1, 2) # c x nums x h x w
        
       # print('----concat shape', concat_img.shape)
        return concat_img


    def load_imgs_and_label(self, sub_folder_path):
        img_list = glob.glob(os.path.join(sub_folder_path, '*.jpg'))
        img_list = list(sorted(img_list))
        
        label = sub_folder_path.split(os.sep)[-2]
        assert label in ['Abnormal', 'Normal']
        
        if self.opt.loss == 'CE':
            label = self.label_encoder_ce[label]
        else:
            label = self.label_encoder_bce[label]

        infor = {
            'video': sub_folder_path.split(os.sep)[-1],
            #'sub_video': sub_folder_path.split(os.sep)[-1],
            #'image': sub_folder_path.split(os.sep)[-1]
        }

        return img_list, label, infor


    def _get_aug_param(self):
        scale = np.random.choice(np.arange(0.8, 1.2, 0.1))
        rot = np.random.randint(-20, 20)

        return scale, rot


    def _flipV(self, img):
        return img[:, ::-1, :]
    

    def _new_rotated_wh(self, rot, w, h):
        assert -180 <= rot < 180
        if 0 <= rot < 90:
            rot_radian = rot / 180 * np.pi
            new_w = w*np.cos(rot_radian) + h*np.sin(rot_radian)
            new_h = w*np.sin(rot_radian) + h*np.cos(rot_radian)
        elif 90 <= rot < 180:
            rot_radian = (rot-90) / 180 * np.pi
            new_w = h*np.cos(rot_radian) + w*np.sin(rot_radian)
            new_h = h*np.sin(rot_radian) + w*np.cos(rot_radian)
        elif -90 <= rot < 0:
            rot_radian = (rot+90) / 180 * np.pi
            new_w = h*np.cos(rot_radian) + w*np.sin(rot_radian)
            new_h = h*np.sin(rot_radian) + w*np.cos(rot_radian)
        elif -180 <= rot < -90 :
            rot_radian = (rot+180) / 180 * np.pi
            new_w = w*np.cos(rot_radian) + h*np.sin(rot_radian)
            new_h = w*np.sin(rot_radian) + h*np.cos(rot_radian)
        return new_w, new_h


    def _augment(self, img_list, opt=None):
        num_imgs = len(img_list)
        input_h, input_w = opt.input_h, opt.input_w
        concat_img = np.zeros((num_imgs, input_h, input_w, 3), dtype=np.float32)

        scale, rot = self._get_aug_param()

        if self.split == 'train':
            flip = np.random.rand() < opt.flip
            rotate = np.random.rand() < opt.rotate
            color_aug = np.random.rand() < opt.color_aug
        elif self.split == 'val':
            flip, rotate, color_aug = False, False, False

        for idx, p in enumerate(img_list):
            img = cv.imread(p)
            img_type = os.path.split(img)[-1]
            if flip:
                img = self._flipV(img)
            
            if rotate:
                h, w, _ = img.shape
                new_w, new_h = self._new_rotated_wh(rot, w, h)
                rot_mat = cv.getRotationMatrix2D((new_w/2, new_h/2), rot, scale)
                img = cv.warpAffine(img, rot_mat, (int(new_w), int(new_h))) 

            if color_aug and img_type.startswith('rgb'):
                color_aug(self._data_rng, img, self._eig_val, self._eig_vec)

            img = cv.resize(img, (input_w, input_h), interpolation=cv.INTER_AREA)
            concat_img[idx] = img

        return concat_img

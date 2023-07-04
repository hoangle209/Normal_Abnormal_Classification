import torch

from libs.dataset.dataset_factory import get_dataset
from libs.backbone.utils import create_model, load_model
from opts import opts

import os
import glob
import numpy as np
import cv2 as cv
import time

class Runner():
    def __init__(self, opt):
        train_dataset = get_dataset(opt.dataset)
        self.class_name = train_dataset.class_name
        self.mean = np.array(train_dataset.mean, dtype=np.float32).reshape(1,1,3)
        self.std = np.array(train_dataset.std, dtype=np.float32).reshape(1,1,3)
        self.default_resolution = train_dataset.default_resolution
        nums_in = train_dataset.num_RGB + train_dataset.num_Flow + train_dataset.num_Heat

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Creating model...")
        self.model = create_model(opt, nums_in=nums_in)
        self.model = load_model(self.model, opt.load_model, opt)
        self.model = self.model.to(device)
        self.model.eval()    

        self.opt = opt
        
    def pre_process(self, img_list):
        num_imgs = len(img_list)
        if self.opt.input_h < 0 or self.opt.input_w < 0:
            input_h, input_w = self.default_resolution
        else:
            input_h, input_w = self.opt.input_h, self.opt.input_w 

        concat_img = np.zeros((num_imgs, input_h, input_w, 3), dtype=np.float32)

        for idx, p in enumerate(img_list):
            img = cv.imread(p)
            img = cv.resize(img, (input_w, input_h), interpolation=cv.INTER_AREA)
            concat_img[idx] = img

        concat_img = concat_img / 255.
        concat_img = (concat_img - self.mean[None, ...]) / self.std[None, ...]
        concat_img = concat_img.transpose(3, 0, 1, 2) # c x nums x h x w
        concat_img = torch.from_numpy(concat_img)

        return concat_img


    def run(self, imgs):
        '''
        :Parameters
        -----------
        > imgs: [*.jpg] | torch.tensor, c x nums x h x w
            can be image list or a concatenated image
        '''
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        pre_begin = time.time()
        if isinstance(imgs, list):
            img = self.pre_process(imgs)
        else:
            img = imgs
        img = img.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        pre_time += (time.time() - pre_begin)

        net_begin = time.time()
        with torch.no_grad():
            output = self.model(img)
        net_time += (time.time() - net_begin)

        post_begin = time.time()
        output = output.cpu().view(-1).float().numpy()
        label_id = np.argmax(output)
        label = self.class_name[label_id]
        post_time += (time.time() - post_begin)

        return {
            'label_id': label_id,
            'label': label,
            'pre_time': pre_time,
            'net_time': net_time,
            'post_time': post_time
                }

        

import torch
import cv2
import numpy as np
from progress.bar import Bar
import glob
import os

from libs.dataset.dataset_factory import get_dataset
from libs.runner import Runner
from opts import opts

class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset):
        self.nums_in = dataset.num_RGB + dataset.num_Flow + dataset.num_Heat
        self.opt = opt
        self.sub_folder_paths = dataset.sub_folder_paths

    def __len__(self):
        return len(self.sub_folder_paths)

    def __getitem__(self, index):
        sub_folder_path = self.sub_folder_paths[index]
        img_list = glob.glob(os.path.join(sub_folder_path, '*.jpg'))
        img_list = list(sorted(img_list))
        return img_list


def _eval(opt):
    train_dataset = get_dataset[opt.dataset](opt.test_path, opt, split='test')
    dataset = PrefetchDataset(opt, train_dataset)
    




    

        
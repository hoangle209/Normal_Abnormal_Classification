import torch

from libs.dataset.dataset_factory import get_dataset
from libs.backbone.utils import create_model, load_model
from opts import opts

import os
import glob
import numpy as np

class Runner():
    def __init__(self, opt, RAFT=None):
        Dataset = get_dataset(opt.dataset)
        nums_in = Dataset.num_RGB + Dataset.num_Flow + Dataset.num_Heat
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Creating model...")
        self.model = create_model(opt, nums_in=nums_in)
        self.model = load_model(self.model, opt.load_model, opt)
        self.model = self.model.to(device)
        self.model.eval()
        
        if RAFT is not None:
            self.raft = RAFT
            self.raft.eval()

        self.opt = opt
        self.mean = np.array(Dataset.mean, dtype=np.float32).reshape(1,1,3)
        self.std = np.array(Dataset.std, dtype=np.float32).reshape(1,1,3)
    

    def run(self, link):
        '''
        :Parameters
        ------------
        > link: *.mp4, [*.mp4] or [*.jpg] or [[*.jpg]]
            can be video or folder containing videos or link to folder containing images
            or folder containing folders containing images
        '''
        pass

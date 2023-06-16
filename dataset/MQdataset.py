from .dataset import GenericDataset

class MQDataset(GenericDataset):
    default_resolution = [512, 512]

    def __init__(self, opt, split):
        path = ''
        num_RGB = 8
        num_OptiF = 7
        num_Heat = -1

        super().__init__(path, opt, num_RGB, num_OptiF, num_Heat, split)

    def run_eval(self):
        pass
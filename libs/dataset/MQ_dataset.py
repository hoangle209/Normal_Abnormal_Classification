from .dataset import GenericDataset
import numpy as np

class MQDataset(GenericDataset):
    default_resolution = [416, 416]
    num_categories = 2
    class_name = ['Abnormal', 'Normal']
    _valid_ids = [1, 2]
    cat_ids = {_cls: _id-1 for _cls, _id in zip(class_name, _valid_ids)}
    labels = np.eye(num_categories, dtype=np.float32)

    def __init__(self, path, opt, split):
        # path = ''
        num_RGB = 8
        num_Flow = 7
        num_Heat = 0

        super().__init__(path, opt, num_RGB, num_Flow, num_Heat, split)

    def run_eval(self, gts, outs):
        TP = (gts == outs).sum()
        P = TP / gts.shape[0]
        return P
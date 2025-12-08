# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class MydataDataset(CustomDataset):


    CLASSES = ('background', 'object')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(MydataDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            ignore_index=10,
            classes= ('background', 'object'),
            palette=[[0, 0, 0], [255, 255, 255]],
            **kwargs)
        assert osp.exists(self.img_dir)


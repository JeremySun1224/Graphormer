# -*- coding: utf-8 -*-
# -*- author: jeremysun1224 -*-

import os
import pickle

raw_dir = "/root/sx/sx_pytorch/Graphormer/datasets/zinc/raw"
split = "val"

with open(os.path.join(raw_dir, f'{split}.pickle'), 'rb') as f:  # 加载原始训练数据*pickle文件
    mols = pickle.load(f)

print(mols)

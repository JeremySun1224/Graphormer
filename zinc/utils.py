# -*- coding: utf-8 -*-
# -*- author: jeremysun1224 -*-

import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def set_seed(args: dataclass):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def cal_parameters_num(model: nn.Module) -> Dict:
    total_num = sum(p.numel() for p in model.parameters())
    total_trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_num": {total_num}, "total_trainable_num": {total_trainable_num}}

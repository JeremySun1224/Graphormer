# -*- coding: utf-8 -*-
# -*- author: jeremysun1224 -*-

import torch
import numpy as np


@torch.jit.script  # 使得函数可以在torch.jit编译模式下运行, 提升性能
def convert_to_single_emb(x, offset: int = 512):  # 为每个特征加上基于原始类别的索引偏移量, 将嵌入空间由[k, 512]降维至[k*512, ], 简化嵌入向量表示
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x =




if __name__ == '__main__':
    _x = torch.arange(10, dtype=torch.float32).reshape(2, 1, 5)
    print(_x)
    print(_x.size())
    _convert_x = convert_to_single_emb(x=_x)
    print(_convert_x)

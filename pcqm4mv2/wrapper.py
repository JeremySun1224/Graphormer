# -*- coding: utf-8 -*-
# -*- author: jeremysun1224 -*-

import numpy as np
import pyximport
import torch

pyximport.install(setup_args={"include_dirs": np.get_include()})

import algos


@torch.jit.script  # 使得函数可以在torch.jit编译模式下运行, 提升性能
def convert_to_single_emb(x, offset: int = 512):  # 为每个特征加上基于原始类别的索引偏移量, 将嵌入空间由[k, 512]降维至[k*512, ], 简化嵌入向量表示
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x  # edge_attr: [num_edges, num_features], edge_index: [2, num_edges], x: [num_nodes, num_features]
    N = x.size(0)  # 节点数
    x = convert_to_single_emb(x)  # 一维特征嵌入

    adj = torch.zeros([N, N], dtype=torch.bool)  # node adj matrix
    adj[edge_index[0, :], edge_index[1, :]] = True

    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]  # 新增一维
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
            convert_to_single_emb(edge_attr) + 1  # TODO: with graph token ???
    )  # 节点间边嵌入特征

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())  # shortest_path_result, 最短距离, [n_node, n_node]; path: 最短路径上第一个中间节点, [n_node, n_node]  #
    max_dist = np.max(shortest_path_result)  # 最短路径中允许的最大步数
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())  # 从节点i到节点j的最短路径上的边的特征
    spatial_pos = torch.from_numpy(shortest_path_result).long()  # 从节点i到节点j的最短距离
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=0).view(-1)  # 入度即对行求和, dim=0, 即B->A, C->A, D->A, E->A, F-A ...
    item.out_degree = item.in_degree
    item.edge_input = torch.from_numpy(edge_input).long()

    return item


if __name__ == '__main__':
    # _x = torch.arange(10, dtype=torch.float32).reshape(2, 1, 5)
    # print(_x)
    # print(_x.size())
    # _convert_x = convert_to_single_emb(x=_x)
    # print(_convert_x)

    from train import GraphormerDataset

    _dataset_spec = "pcqm4mv2"
    _dataset_source = "ogb"
    _dataset = GraphormerDataset(
        dataset_spec=_dataset_spec,
        dataset_source=_dataset_source
    )
    _dataset_train = _dataset.dataset_train.dataset
    _dataset_val = _dataset.dataset_val.dataset
    _dataset_test = _dataset.dataset_test.dataset

    _item = _dataset_train[0]
    __item = preprocess_item(item=_item)

    print(_dataset)

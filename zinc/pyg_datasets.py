# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
from functools import lru_cache

import numpy as np
import pyximport
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset

pyximport.install(setup_args={"include_dirs": np.get_include()})
import algos


@torch.jit.script  # pytorch会尝试将该函数编译为torch脚本, 可以脱离python环境, 便于生产部署
def convert_to_single_emb(x, offset: int = 512):  # 将嵌入空间转为嵌入向量
    feature_num = x.size(1) if len(x.size()) > 1 else 1  # edge特征数
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)  # 不同特征赋予不同偏移量以确保合并后的嵌入向量不会重叠, 保持特征区分度, TODO: 为什么加1
    x = x + feature_offset
    return x


def preprocess_item(item):
    """
    x: node features; 节点特征, 也即每个节点类型; [n_node, 1]
    edge_index: pairs of nodes constituting edges; 节点索引对之间的连边; [2, n_edge]
    edge_attr: edge features, for the aforementioned edges, contains their feature; 每条连边的特征, 也可以说是类型; [n_edge, ]
    """
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)  # n_node
    x = convert_to_single_emb(x)  # 一维特征嵌入

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True  # 将有边相连的节点对标记出来, 即邻接矩阵

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]  # 增加一维
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1  # 加1是为了pad
    )  # 边特征嵌入

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())  # 从节点i到节点j的最短距离矩阵; 从节点i经过节点k到达节点j的最短路径中间节点
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())  # 经过回溯的, 返回的数字表示的是边的类型, 不同的类型就组成立从节点i到节点j的路径
    spatial_pos = torch.from_numpy((shortest_path_result)).long()  # 从节点i到节点j的最短距离矩阵
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # print(f"======= algos start =======")
    # if N == 11:
    #     print(f"N: {N}")
    #     print(f"adj: {adj}")
    #     print(f"adj shape: {adj.size()}")
    #     shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    #     print(f"shortest_path_result: {shortest_path_result}")
    #     print(f"path: {path}")
    #     max_dist = np.amax(shortest_path_result)
    #     print(f"max_dist: {max_dist}")
    #     print(f"attn_edge_type: {attn_edge_type}")
    #     edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    #     # print(f"edge_input: {edge_input}")
    #     print(f"edge_input shape: {type(edge_input)}")
    #     spatial_pos = torch.from_numpy((shortest_path_result)).long()
    #     # print(f"spatial_pos: {spatial_pos}")
    #     attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token
    #     # print(f"attn_bias: {attn_bias}")
    #     exit()
    # print(f"======= algos end =======")

    # combine
    item.x = x  # [n_node, 1], 图中每个节点特征
    item.attn_bias = attn_bias  # [n_node+1, n_node+1], 自定义节点间最短距离是否超过了spatial_pos_max, 超过则设置为-inf
    item.attn_edge_type = attn_edge_type  # [n_node, n_node, 1], 每对相连节点对之间连边类型的嵌入
    item.spatial_pos = spatial_pos  # [n_node, n_node], 每对相连节点对之间的最短距离
    item.in_degree = adj.long().sum(dim=1).view(-1)  # [n_node, ], 入度
    item.out_degree = item.in_degree  # [n_node, ], 出度, for undirected graph, 对于无向图, 入度==出度
    item.edge_input = torch.from_numpy(edge_input).long()  # [n_node, n_node, max_dist, 1], 每对相连节点的最短距离的回溯路径, 由经过的边的类型组成, spatial_pos[0][1] = len(edge_input[0][1])

    return item


class GraphormerPYGDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        seed: int = 0,
        train_idx=None,
        valid_idx=None,
        test_idx=None,
        train_set=None,
        valid_set=None,
        test_set=None,
    ):
        self.dataset = dataset
        if self.dataset is not None:
            self.num_data = len(self.dataset)
        self.seed = seed
        if train_idx is None and train_set is None:
            train_valid_idx, test_idx = train_test_split(
                np.arange(self.num_data),
                test_size=self.num_data // 10,
                random_state=seed,
            )
            train_idx, valid_idx = train_test_split(
                train_valid_idx, test_size=self.num_data // 5, random_state=seed
            )
            self.train_idx = torch.from_numpy(train_idx)
            self.valid_idx = torch.from_numpy(valid_idx)
            self.test_idx = torch.from_numpy(test_idx)
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
        elif train_set is not None:
            self.num_data = len(train_set) + len(valid_set) + len(test_set)
            self.train_data = self.create_subset(train_set)
            self.valid_data = self.create_subset(valid_set)
            self.test_data = self.create_subset(test_set)
            self.train_idx = None
            self.valid_idx = None
            self.test_idx = None
        else:
            self.num_data = len(train_idx) + len(valid_idx) + len(test_idx)
            self.train_idx = train_idx
            self.valid_idx = valid_idx
            self.test_idx = test_idx
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
        self.__indices__ = None

    def index_select(self, idx):
        dataset = copy.copy(self)
        dataset.dataset = self.dataset.index_select(idx)
        if isinstance(idx, torch.Tensor):
            dataset.num_data = idx.size(0)
        else:
            dataset.num_data = idx.shape[0]
        dataset.__indices__ = idx
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset

    def create_subset(self, subset):
        dataset = copy.copy(self)
        dataset.dataset = subset
        dataset.num_data = len(subset)
        dataset.__indices__ = None
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.dataset[idx]
            item.idx = idx
            item.y = item.y.reshape(-1)
            return preprocess_item(item)
        else:
            raise TypeError("index to a GraphormerPYGDataset can only be an integer.")

    def __len__(self):
        return self.num_data

    def get(self, idx: int):
        pass

    def len(self) -> int:
        pass

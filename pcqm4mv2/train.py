# -*- coding: utf-8 -*-
# -*- author: jeremysun1224 -*-

import copy
import os
from abc import ABC
from functools import lru_cache
from typing import Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset
from torch_geometric.data import Data as PYGDataset
from dgl.data import DGLDataset
from fairseq.data import FairseqDataset

from config import TrainArgs
from wrapper import preprocess_item

args = TrainArgs


def load_dataset(dm, split):
    assert split in ["train", "valid", "test"]
    if split == "train":
        batched_data = dm.dataset_train
    elif split == "valid":
        batched_data = dm.dataset_val
    elif split == "test":
        batched_data = dm.dataset_test

    batched_data = BatchedDataDataset(

    )


class BatchedDataDataset(FairseqDataset, ABC):
    def __init__(
            self,
            dataset,
            max_node=128,
            multi_hop_max_dist=5,
            spatial_pos_max=1024
    ):
        super(BatchedDataDataset, self).__init__()
        self.dataset = dataset
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def collater(self, samples):
        return


class GraphormerDataset:
    def __init__(
            self,
            dataset: Optional[Union[PYGDataset, DGLDataset]] = None,
            dataset_spec: Optional[str] = None,
            dataset_source: Optional[str] = None,
            seed: int = 0,
            train_idx=None,
            valid_idx=None,
            test_idx=None
    ):
        super(GraphormerDataset, self).__init__()
        if dataset is not None:
            if dataset_source == "dgl":
                pass
            elif dataset_source == "pyg":
                self.dataset = GraphormerPYGDataset(
                    dataset=dataset,
                    seed=seed,
                    train_idx=train_idx,
                    valid_idx=valid_idx,
                    test_idx=test_idx
                )
            else:
                raise ValueError("customized data can only have source dgl or pyg.")
        elif dataset_source == "dgl":
            pass
        elif dataset_source == "pyg":
            pass
        elif dataset_source == "ogb":
            self.dataset = OGBDatasetLookupTable.get_ogb_dataset(dataset_name=dataset_spec, seed=seed)

        self.train_idx = None
        self.valid_idx = None
        self.test_idx = None
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

        self.setup()

    def setup(self):
        self.train_idx = self.dataset.train_idx
        self.valid_idx = self.dataset.valid_idx
        self.test_idx = self.dataset.test_idx

        self.dataset_train = self.dataset.train_data
        self.dataset_val = self.dataset.valid_data
        self.dataset_test = self.dataset.test_data


class GraphormerPYGDataset(Dataset, ABC):
    def __init__(
            self,
            dataset: Dataset,
            seed: int = 0,
            train_idx=None,
            valid_idx=None,
            test_idx=None,
            train_set=None,
            valid_set=None,
            test_set=None
    ):
        super(GraphormerPYGDataset).__init__()
        self.dataset = dataset
        if self.dataset is not None:
            self.num_data = len(self.dataset)
        self.seed = seed
        if train_idx is None and train_set is None:  # 没有指定索引或数据集时, 自动生成索引进行数据集切分
            train_valid_idx, test_idx = train_test_split(
                np.arange(self.num_data),
                test_size=self.num_data // 10,
                random_state=seed
            )
            train_idx, valid_idx = train_test_split(
                train_valid_idx,
                test_size=self.num_data // 5,
                random_state=seed
            )
            self.train_idx = torch.from_numpy(train_idx)
            self.valid_idx = torch.from_numpy(valid_idx)
            self.test_idx = torch.from_numpy(test_idx)
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
        elif train_set is not None:  # 指定数据集, 将数据集设计为类属性
            self.num_data = len(train_set) + len(valid_set) + len(test_set)
            self.train_data = self.create_subset(train_set)
            self.valid_data = self.create_subset(valid_set)
            self.test_data = self.create_subset(test_set)
            self.train_idx = None
            self.valid_idx = None
            self.test_idx = None
        else:  # 指定了索引时, 根据索引从数据集中获取相应数据
            self.num_data = len(train_idx) + len(valid_idx) + len(test_idx)
            self.train_idx = train_idx
            self.valid_idx = valid_idx
            self.test_idx = test_idx
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)

        self.__indices__ = None

    def index_select(self, idx):  # self即GraphormerPYGDataset(3599188)
        dataset = copy.copy(self)  # 创建当前对象的浅拷贝, 新的对象将有和原对象相同的属性值, 但不会共享复杂子对象的内存引用, 以便在新对象中保留对原始数据的引用
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

    def create_subset(self, subset):  # 根据传入的数据集创建一个新的子集
        dataset = copy.copy(self)  # 创建浅拷贝后, 重新设置数据集属性, 并将其他不相关的属性(如训练/验证/测试和索引)设置为None, 这样可以确保新对象独立于原对象, 并专注于处理子集数据
        dataset.dataset = subset
        dataset.num_data = len(subset)
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None

        return dataset

    def __len__(self):
        return self.num_data

    @lru_cache(maxsize=16)  # least recently used, 用于缓存函数调用的结果, 避免重复计算相同输入时的开销. maxsize, 最大缓存条目数. typed, 若为True, 不同类型的参数将分别缓存.
    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.dataset[idx]
            item.idx = idx
            item.y = item.y.reshape[-1]
            return preprocess_item(item)
        else:
            raise TypeError("Index to a GraphormerPYGDataset can only be an integer.")

    def get(self, idx: int):
        pass

    def len(self):
        pass


class MyPygPCQM4Mv2Dataset(PygPCQM4Mv2Dataset):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:  # 检查分布式环境是否被初始化或当前进程是否为主进程
            super(MyPygPCQM4Mv2Dataset, self).download()  # 调用父类download()方法
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:  # 检查分布式环境是否被初始化或当前进程是否为主进程
            super(MyPygPCQM4Mv2Dataset, self).process()  # 调用父类process()方法
        if dist.is_initialized():
            dist.barrier()


class OGBDatasetLookupTable:
    @staticmethod
    def get_ogb_dataset(dataset_name: str, seed: int) -> Optional[Dataset]:
        inner_dataset = None
        train_idx = None
        valid_idx = None
        test_idx = None
        if dataset_name == "ogbg-molhiv":
            pass
        elif dataset_name == "ogbg-molpcba":
            pass
        elif dataset_name == "pcqm4m":
            pass
        elif dataset_name == "pcqm4mv2":
            os.system("mkdir -p dataset/pcqm4m-v2")
            os.system("touch dataset/pcqm4m-v2/RELEASE_v1.txt")
            inner_dataset = MyPygPCQM4Mv2Dataset(root=os.path.join(args.root_path, "datasets"))
            idx_split = inner_dataset.get_idx_split()
            train_idx = idx_split["train"]
            valid_idx = idx_split["valid"]
            test_idx = idx_split["test-dev"]
        else:
            raise ValueError(f"Unknown dataset name {dataset_name} for ogb source.")

        return (
            None
            if inner_dataset is None
            else GraphormerPYGDataset(
                dataset=inner_dataset,
                seed=seed,
                train_idx=train_idx,
                valid_idx=valid_idx,
                test_idx=test_idx
            )
        )


if __name__ == '__main__':
    _inner_dataset = MyPygPCQM4Mv2Dataset(root=os.path.join(args.root_path, "datasets"))
    _idx_split = _inner_dataset.get_idx_split()
    print(_idx_split)

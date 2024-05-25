# -*- coding: utf-8 -*-
# -*- author: jeremysun1224 -*-
import copy
import os
from abc import ABC
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
from torch_geometric.data import Data, Dataset
from sklearn.model_selection import train_test_split

from config import TrainArgs

args = TrainArgs


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
        if train_idx is None and train_set is None:
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
        elif train_set is not None:
            self.num_data = len(train_set) + len(valid_set) + len(test_set)
            self.train_data = self.create_data(train_set)
            self.valid_data = self.create_data(valid_set)
            self.test_data = self.create_data(test_set)
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
        dataset.dataset = self.index_select(idx)
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
    def get_ogb_dataset(dataset_name: str, seed: int) -> Optional[Data]:
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
            None if inner_dataset is None else 1
        )


if __name__ == '__main__':
    _inner_dataset = MyPygPCQM4Mv2Dataset(root=os.path.join(args.root_path, "datasets"))
    _idx_split = _inner_dataset.get_idx_split()
    print(_idx_split)

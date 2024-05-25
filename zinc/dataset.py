import logging
from functools import lru_cache

import numpy as np
import torch
from fairseq.data import NestedDictionaryDataset, NumSamplesDataset
from fairseq.data import data_utils, FairseqDataset, BaseWrapperDataset

from collator import collator

logger = logging.getLogger(__name__)


class BatchedDataDataset(FairseqDataset):
    def __init__(
        self, dataset, max_node=128, multi_hop_max_dist=5, spatial_pos_max=1024
    ):
        super().__init__()
        self.dataset = dataset
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return collator(
            samples,
            max_node=self.max_node,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
        )  # 打包成张量


class TargetDataset(FairseqDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index].y

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return torch.stack(samples, dim=0)


class GraphormerDataset:
    def __init__(
            self,
            dataset
    ):
        super().__init__()
        self.dataset = dataset
        self.setup()

    def setup(self):
        self.dataset_train = self.dataset.train_data
        self.dataset_val = self.dataset.valid_data
        self.dataset_test = self.dataset.test_data


class EpochShuffleDataset(BaseWrapperDataset):
    def __init__(self, dataset, num_samples, seed):
        super().__init__(dataset)
        self.num_samples = num_samples
        self.seed = seed
        self.set_epoch(1)

    def set_epoch(self, epoch):
        with data_utils.numpy_seed(self.seed + epoch - 1):
            self.sort_order = np.random.permutation(self.num_samples)

    def ordered_indices(self):
        return self.sort_order

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False


def load_dataset(dm, split):
    """Load a given dataset split (e.g., train, valid, test)."""

    assert split in ["train", "valid", "test"]

    if split == "train":
        batched_data = dm.dataset_train
    elif split == "valid":
        batched_data = dm.dataset_val
    elif split == "test":
        batched_data = dm.dataset_test

    batched_data = BatchedDataDataset(
        batched_data
    )

    data_sizes = np.array([batched_data.max_node] * len(batched_data))

    target = TargetDataset(batched_data)

    dataset = NestedDictionaryDataset(
        {
            "nsamples": NumSamplesDataset(),
            "net_input": {"batched_data": batched_data},
            "target": target,
        },
        sizes=data_sizes,
    )

    if split == "train":
        dataset = EpochShuffleDataset(
            dataset, num_samples=len(dataset), seed=1
        )

    return dataset

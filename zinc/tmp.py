# -*- coding: utf-8 -*-
# -*- author: jeremysun1224 -*-

import math
import os

import torch
from torch.optim.adam import Adam
from torch.utils.data import RandomSampler, DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from config import TrainArgs
from dataset import GraphormerDataset, load_dataset
from model import GraphormerModelSlim
from pyg_datasets import GraphormerPYGDataset
from utils import set_seed, cal_parameters_num
from zinc_dataset import ZINC
import logging

args = TrainArgs
args.batch_size = 256
args.n_gpu = 1

logger = logging.getLogger(__file__)

print(f"loss with flag: {args.flag}")

device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.device = device

set_seed(args=args)

train_data = ZINC(root=os.path.join(args.root_path, "datasets/zinc"), subset=True, split="train")
valid_data = ZINC(root=os.path.join(args.root_path, "datasets/zinc"), subset=True, split="val")
test_data = ZINC(root=os.path.join(args.root_path, "datasets/zinc"), subset=True, split="test")

datasets = GraphormerPYGDataset(None, 1, None, None, None, train_data, valid_data, test_data)
datasets = GraphormerDataset(datasets)

train_datasets = load_dataset(dm=datasets, split="train")  # hasattr(dataset, "collater")
train_sampler = RandomSampler(train_datasets)
train_dataloader = DataLoader(
    dataset=train_datasets,
    # batch_size=args.batch_size,
    batch_size=2,
    sampler=train_sampler,
    collate_fn=train_datasets.collater
)
print(f"num of train_datasets: {len(train_datasets)}")
print(f"num of train_dataloader: {len(train_dataloader)}")


# node_list, edge_list, in_degree_list, out_degree_list, spatial_pos_list = [], [], [], [], []
# for batch in tqdm(train_dataloader):
#     batched_data = batch["net_input"]["batched_data"]
#     # idx = batched_data["idx"]
#     # attn_bias = batched_data["attn_bias"]
#     # attn_edge_type = batched_data["attn_edge_type"]
#     # spatial_pos = batched_data["spatial_pos"]
#     # in_degree = batched_data["in_degree"]
#     # out_degree = batched_data["out_degree"]
#     edge_input = batched_data["edge_input"]
#     # x = batched_data["x"]  # [n_graph, n_node, 1]
#     # y = batched_data["y"]
#
#     # test
#     # print(spatial_pos.size())
#     spatial_pos_list.extend(edge_input.view(-1).tolist())
#     # print(node_list)
# print(set(spatial_pos_list))
# exit()

model = GraphormerModelSlim(args)
model.to(args.device)

print(model)

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.numel()}")

parameters = cal_parameters_num(model=model)
print(parameters)

# exit()

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]

optimizer = Adam(
    params=optimizer_grouped_parameters,
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    eps=args.eps
)

scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=args.max_steps
)

epochs = args.max_steps // len(train_dataloader)

model.zero_grad()

global_step = 0

for epoch in range(epochs):
    epoch_loss = 0.
    for step, batch in enumerate(train_dataloader):
        nsamples = batch["nsamples"]
        net_input = batch["net_input"]
        target = batch["target"]

        batched_data = batch["net_input"]["batched_data"]
        idx = batched_data["idx"]
        attn_bias = batched_data["attn_bias"]
        attn_edge_type = batched_data["attn_edge_type"]
        spatial_pos = batched_data["spatial_pos"]
        in_degree = batched_data["in_degree"]
        out_degree = batched_data["out_degree"]
        edge_input = batched_data["edge_input"]
        x = batched_data["x"]  # [n_graph, n_node, 1]
        y = batched_data["y"]

        model.train()  # model.eval()是为了取消dropout等损失输入精度的操作
        batches = {k: v.to(args.device) for k, v in batched_data.items()}
        targets = target.to(args.device)

        if args.flag:  # loss with flag, 构建形同输入embed的扰动perturb
            n_graph, n_node = x.shape[:2]
            perturb_shape = n_graph, n_node, args.encoder_embed_dim

            if args.flag_mag > 0:
                perturb = (torch.FloatTensor(*perturb_shape).uniform_(-1, 1).to(args.device))
                perturb = perturb * args.flag_mag / math.sqrt(perturb_shape[-1])
            else:
                perturb = (torch.FloatTensor(*perturb_shape).uniform_(-args.flag_step_size, args.flag_step_size).to(args.device))

            perturb.requires_grad_()

            logits, loss = model(batched_data=batches, perturb=perturb, targets=targets)
            loss /= args.flag_m
            for _ in range(args.flag_m - 1):
                loss.backward()

                perturb_data = perturb.detach() + args.flag_step_size * torch.sign(perturb.grad.detach())
                if args.flag_mag > 0:
                    perturb_data_norm = torch.norm(perturb_data, dim=-1).detach()  # [n_graph, n_node]
                    exceed_mask = (perturb_data_norm > args.flag_mag).to(perturb_data)
                    reweights = (args.flag_mag / perturb_data_norm * exceed_mask + (1 - exceed_mask)).unsqueeze(-1)
                    perturb_data = (perturb_data * reweights).detach()  # [n_graph, n_node, n_hidden]
                perturb.data = perturb_data.data
                perturb.grad[:] = 0  # [n_graph, n_node, n_hidden]
                logits, loss = model(batched_data=batches, perturb=perturb, targets=targets)
                loss /= args.flag_m

            epoch_loss += loss

            loss.backward()
            optimizer.step()
            scheduler.step()
        else:
            logits, loss = model(batched_data=batches, targets=targets)  # 前向传播
            epoch_loss += loss

            loss.backward()  # 求导计算梯度
            optimizer.step()  # 更新权重参数
            scheduler.step()

            model.zero_grad()  # 因为当前step的的参数已经被更新, 无需继续保存梯度, 故zero_grad()

        global_step += 1
        if global_step > args.max_steps:
            break

        if step % args.logging_step == 0:
            logger.info(
                f"epoch: {epoch}/{epochs}, "
                f"global_step: {global_step}/{args.max_steps}, "
                f"avg_loss: {epoch_loss / (step + 1)}"
            )
        # print(f"logits: {logits}")
        # print(f"loss: {loss}")
        # print(f"logits: {logits.size()}")
        # print(f"loss: {loss.size()}")

        # print(f"========= value =========")
        # print(f"idx: {idx.view(-1).tolist()}")
        # print(f"attn_bias: {attn_bias.view(-1).tolist()}")
        # print(f"attn_edge_type: {attn_edge_type.view(-1).tolist()}")
        # print(f"spatial_pos: {spatial_pos.view(-1).tolist()}")
        # print(f"in_degree: {in_degree.view(-1).tolist()}")
        # print(f"out_degree: {out_degree.view(-1).tolist()}")
        # print(f"edge_input: {edge_input.view(-1).tolist()}")
        # print(f"x: {x.view(-1).tolist()}")
        # print(f"y: {y.view(-1).tolist()}")

        # print(f"idx: {idx.size()}")
        # print(f"attn_bias: {attn_bias.size()}")
        # print(f"attn_edge_type: {attn_edge_type.size()}")
        # print(f"spatial_pos: {spatial_pos.size()}")
        # print(f"in_degree: {in_degree.size()}")
        # print(f"out_degree: {out_degree.size()}")
        # print(f"edge_input: {edge_input.size()}")
        # print(f"x: {x.size()}")
        # print(f"y: {y.size()}")

    if epoch > args.max_epochs:
        break

    logger.info(
        f"epoch: {epoch}/{epochs}, "
        f"global_step: {global_step}/{args.max_steps}, "
        f"avg_epoch_loss: {epoch_loss / len(train_dataloader)}"
    )

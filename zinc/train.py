# -*- coding: utf-8 -*-
# -*- author: jeremysun1224 -*-

import os

import torch.distributed
from torch.cuda.amp import GradScaler, autocast
from torch.optim.adam import Adam
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader
from transformers import get_linear_schedule_with_warmup

from config import TrainArgs
from dataset import GraphormerDataset, load_dataset
from model import GraphormerModelSlim
from pyg_datasets import GraphormerPYGDataset
from utils import set_seed
from zinc_dataset import ZINC

local_rank = int(os.environ.get("LOCAL_RANK", -1))

if __name__ == '__main__':
    args = TrainArgs
    args.local_rank = local_rank

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    print("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".
          format(device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    set_seed(args)

    model = GraphormerModelSlim(args)
    model.to(args.device)
    print(model)

    train_data = ZINC(root=os.path.join(args.root_path, "datasets/zinc"), subset=True, split="train")
    valid_data = ZINC(root=os.path.join(args.root_path, "datasets/zinc"), subset=True, split="val")
    test_data = ZINC(root=os.path.join(args.root_path, "datasets/zinc"), subset=True, split="test")

    datasets = GraphormerPYGDataset(None, 1, None, None, None, train_data, valid_data, test_data)
    datasets = GraphormerDataset(datasets)
    train_datasets = load_dataset(dm=datasets, split="train")
    train_sampler = RandomSampler(train_datasets) if args.local_rank == -1 else DistributedSampler(train_datasets)
    train_dataloader = DataLoader(
        dataset=train_datasets,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=train_datasets.collater
    )

    # valid_datasets = load_dataset(dm=datasets, split="valid")
    # valid_sampler = DistributedSampler(valid_datasets)
    # valid_dataloader = DataLoader(
    #     dataset=valid_datasets,
    #     batch_size=args.batch_size,
    #     collate_fn=valid_datasets.collater
    # )

    if args.local_rank in [-1, 0]:
        print(f"num of train_datasets: {len(train_datasets)}")
        print(f"num of train_dataloader: {len(train_dataloader)}")

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    # if local_rank != -1:
    #     args.lr *= torch.distributed.get_world_size()
    optimizer = Adam(optimizer_grouped_parameters, betas=(args.beta1, args.beta2), eps=args.eps, lr=args.lr)

    if args.warmup_steps > 0:
        assert args.max_steps > 0
        args.max_steps //= torch.distributed.get_world_size()
        args.warmup_steps //= torch.distributed.get_world_size()

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    if not os.path.exists(os.path.join(args.save_path, args.pth_name)):
        start_epoch = 0
        global_step = 0
        print('fine tuning model from scratch.')
    else:
        checkpoint = torch.load(os.path.join(args.save_path, args.pth_name), map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        if args.local_rank in [-1, 0]:
            print(f'load checkpoint, epoch: {start_epoch}, global_step: {global_step}')

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    scaler = GradScaler()
    model.zero_grad()

    for _ in range(start_epoch, args.max_epochs):
        epoch_loss = 0
        train_sampler.set_epoch(_)
        for step, instance in enumerate(train_dataloader):
            model.train()
            batch = {k: v.to(args.device) for k, v in instance['net_input']['batched_data'].items()}
            targets = instance['target'].to(args.device)
            if args.fp16:
                with autocast():
                    logits, loss = model(batched_data=batch, targets=targets)
            else:
                logits, loss = model(batched_data=batch, targets=targets)

            epoch_loss += loss.item()

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if args.fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()

            scheduler.step()
            model.zero_grad()

            global_step += 1

            if global_step > args.max_steps:
                break

        if _ > args.max_epochs:
            break

        if args.local_rank in [-1, 0]:
            print(f'epoch: {_}/{args.max_epochs},   '
                  f'global step: {global_step}/{args.max_steps},   avg loss: {epoch_loss / len(train_dataloader)},   '
                  f'learning_rate: {scheduler.get_last_lr()[0]}')

        torch.save({
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': scheduler.state_dict(),
            'epoch': _,
            'global_step': global_step},
            os.path.join(args.save_path, args.pth_name))

import argparse
from contextlib import contextmanager
import logging
import os

import torch
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    default_data_collator,
)

from checkpoint import (
    ensure_experiment_dir,
    resolve_experiment_dir,
    resume_if_available,
    save_checkpoint,
)
from data import load_and_preprocess_data

LOGGER = logging.getLogger(__name__)


@record
def main():
    parser = _get_parser()
    args = parser.parse_args()

    # torchrun 注入：RANK 全局编号，LOCAL_RANK 本机编号（用来绑 GPU），WORLD_SIZE 进程总数
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(rank=rank, world_size=world_size, device_id=device)

    logging.basicConfig(
        format=f"[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )

    LOGGER.debug(os.environ)
    LOGGER.debug(args)
    LOGGER.debug(f"local_rank={local_rank} rank={rank} world_size={world_size}")

    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    # 注意：假设 $HF_HOME 是共享存储；rank0_first 避免多进程同时下载/写缓存
    with rank0_first(), device:
        config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    LOGGER.info(
        f"Training {sum(p.numel() for p in model.parameters())} model parameters"
    )

    model = torch.compile(model)
    LOGGER.info(f"Initialized model uses {get_mem_stats(device)['curr_alloc_gb']}gb")

    model = DistributedDataParallel(
        model, device_ids=[local_rank], bucket_cap_mb=500, gradient_as_bucket_view=True
    )
    LOGGER.info(f"After DDP: model uses {get_mem_stats(device)['curr_alloc_gb']}gb")

    with rank0_first():
        train_data = load_and_preprocess_data(args, config)
    LOGGER.debug(f"{len(train_data)} training samples")

    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=default_data_collator,
        # DistributedSampler：按进程切分样本；不要再设 shuffle=True
        sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
    )
    LOGGER.info(f"{len(dataloader)} batches per epoch")

    # ZeRO：切分优化器状态以省显存；保存其 state_dict 很慢，故存盘时跳过优化器
    optimizer = ZeroRedundancyOptimizer(
        model.parameters(), optimizer_class=torch.optim.AdamW, lr=args.lr, fused=True
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=args.lr * 1e-2
    )

    is_experiment, exp_dir = resolve_experiment_dir(args)

    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
    }
    running_loss = torch.zeros((), device=device, dtype=torch.float32)

    if is_experiment:
        state, resumed = resume_if_available(
            exp_dir,
            device,
            model,
            optimizer,
            lr_scheduler,
            state,
            load_optimizer=False,
        )
        if rank == 0:
            LOGGER.info(f"Resumed={resumed} | {state}")
    dist.barrier()

    if is_experiment and rank == 0:
        ensure_experiment_dir(exp_dir)
    dist.barrier()

    for state["epoch"] in range(state["epoch"], args.num_epochs):
        LOGGER.info(f"Begin epoch {state['epoch']} at step {state['epoch_step']}")

        # 每个 epoch 都要 set_epoch，打乱顺序才会对
        dataloader.sampler.set_epoch(state["epoch"])

        progress_bar = tqdm.tqdm(range(len(dataloader)), disable=rank > 0)
        if state["epoch_step"] > 0:
            progress_bar.update(state["epoch_step"])

        for i_step, batch in enumerate(dataloader):
            if i_step < state["epoch_step"]:
                continue

            batch = {
                k: v.to(device=device, non_blocking=True) for k, v in batch.items()
            }

            outputs = model(**batch)
            del batch

            outputs.loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            state["global_step"] += 1
            state["epoch_step"] += 1
            running_loss += outputs.loss.detach().float()
            progress_bar.update(1)

            if state["global_step"] % args.log_freq == 0:
                info = {
                    "global_step": state["global_step"],
                    "lr": lr_scheduler.get_last_lr()[0],
                    "running_loss": (running_loss / args.log_freq).item(),
                    "epoch": state["epoch"],
                    "epoch_progress": state["epoch_step"] / len(dataloader),
                    "num_batches_remaining": len(dataloader) - i_step,
                    "world_size": world_size,
                    **get_mem_stats(device),
                }
                LOGGER.info(info)
                torch.cuda.reset_peak_memory_stats(device)
                running_loss.zero_()

            if is_experiment and state["global_step"] % args.ckpt_freq == 0:
                if rank == 0:
                    # ZeRO 优化器 state 很大且保存慢，本章不写 optimizer.pt
                    save_checkpoint(
                        exp_dir,
                        model,
                        optimizer,
                        lr_scheduler,
                        state,
                        include_optimizer=False,
                    )
                dist.barrier()

        state["epoch_step"] = 0


def get_mem_stats(device=None):
    """读取当前 GPU 显存占用（已分配 / 峰值 / 预留），单位 GB。"""
    mem = torch.cuda.memory_stats(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "total_gb": 1e-9 * props.total_memory,
        "curr_alloc_gb": 1e-9 * mem["allocated_bytes.all.current"],
        "peak_alloc_gb": 1e-9 * mem["allocated_bytes.all.peak"],
        "curr_resv_gb": 1e-9 * mem["reserved_bytes.all.current"],
        "peak_resv_gb": 1e-9 * mem["reserved_bytes.all.peak"],
    }


@contextmanager
def rank0_first():
    """让 rank 0 先跑完一段逻辑（如下载），再让其余 rank 继续。"""
    rank = dist.get_rank()
    if rank == 0:
        yield
    dist.barrier()
    if rank > 0:
        yield
    dist.barrier()


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment-name", default=None)
    parser.add_argument("-d", "--dataset-name", default=None, required=True)
    parser.add_argument("--dataset-subset", default=None)
    parser.add_argument("-m", "--model-name", default=None, required=True)
    parser.add_argument("--save-dir", default="../outputs")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument("--log-freq", default=10, type=int)
    parser.add_argument("--ckpt-freq", default=500, type=int)
    parser.add_argument("-s", "--seq-length", default=1024, type=int)
    return parser


if __name__ == "__main__":
    main()

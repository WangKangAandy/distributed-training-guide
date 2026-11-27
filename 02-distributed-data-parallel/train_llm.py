import argparse
import logging
import multiprocessing
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

# 本模块专用的 logger，方便按模块过滤日志
LOGGER = logging.getLogger(__name__)


# @record：worker 异常时把完整堆栈写入 TORCHELASTIC_ERROR_FILE（需配合 torchrun 环境变量）
@record
def main():
    # 解析命令行参数（模型名、数据集、学习率等）
    parser = _get_parser()
    args = parser.parse_args()

    # torchrun 注入的环境变量（每个进程各自一份）
    rank = int(os.environ["RANK"])  # 全局进程编号 0 … world_size-1
    local_rank = int(os.environ["LOCAL_RANK"])  # 本机内编号，用来绑 GPU
    world_size = int(os.environ["WORLD_SIZE"])  # 进程总数（≈ GPU 数，单节点时）

    # 本进程只用一张卡：cuda:local_rank（不要用 rank % 卡数，多节点会绑错）
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)  # 设当前默认 CUDA 设备，后续分配落在这张卡

    # 初始化进程组：阻塞直到 WORLD_SIZE 个进程都到齐
    # backend=nccl：GPU 常用通信后端；device_id：告知进程组本进程对应哪张 GPU
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    # 日志带上 rank，多进程写同一文件时也好区分
    logging.basicConfig(
        format=f"[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )

    # 多节点时 DEBUG 打印环境，便于确认各节点配置一致
    LOGGER.debug(os.environ)
    LOGGER.debug(args)
    LOGGER.debug(f"local_rank={local_rank} rank={rank} world_size={world_size}")

    # 本指南假设 CUDA 可用，并统一用 bf16 训练
    dtype = torch.bfloat16
    # 为 PyTorch 随机数生成器设种子（各进程相同种子 + DDP 广播，副本可对齐）
    torch.manual_seed(args.seed)

    # 训练阶段只读本地/缓存制品（local_files_only），不负责联网下载。
    # from_config：按配置随机初始化（从头训练），不加载预训练权重。
    # 本章不引入 torch.compile，避免把编译器优化混入 DDP 入门主线。
    # with device: 新建参数默认落在本进程 GPU，省去再 .to(device)
    with device:
        # 只读模型配置（层数、隐层等）；use_cache=False 关掉 KV Cache（训练不需要）
        config = AutoConfig.from_pretrained(
            args.model_name, local_files_only=True, use_cache=False
        )
        # 按配置新建 Causal LM；权重是随机的，不是 Hub 上的预训练权重
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    LOGGER.info(
        f"Training {sum(p.numel() for p in model.parameters())} model parameters"
    )
    LOGGER.info(f"Initialized model uses {get_mem_stats(device)['curr_alloc_gb']}gb")

    # DDP 包装：反向时在进程间同步（平均）梯度；不负责切分数据
    # device_ids：本进程可见的 GPU；bucket_cap_mb / gradient_as_bucket_view：通信分桶相关优化
    model = DistributedDataParallel(
        model, device_ids=[local_rank], bucket_cap_mb=500, gradient_as_bucket_view=True
    )
    LOGGER.info(f"After DDP: model uses {get_mem_stats(device)['curr_alloc_gb']}gb")

    # 每个 rank 都会跑 datasets.map；按 world_size 均分 CPU，避免 8 进程各开满核
    num_proc = max(1, multiprocessing.cpu_count() // world_size)
    # 加载/分词/切块；local_files_only ≠ 预处理已完成，缓存未命中时仍会 map
    train_data = load_and_preprocess_data(args, config, num_proc=num_proc)
    LOGGER.debug(f"{len(train_data)} training samples")

    # 标准 PyTorch DataLoader；分布式下用 sampler 分片，不要再设 shuffle=True
    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,  # 每进程 batch；全局 ≈ batch_size * world_size
        num_workers=2,  # 每进程额外 2 个取数 worker（可与 GPU 计算重叠）
        prefetch_factor=2,  # 每个 worker 预取 2 个 batch（仅 num_workers>0 时有效）
        pin_memory=True,  # 页锁定内存，便于后面 non_blocking 异步 H2D
        collate_fn=default_data_collator,  # HF collator；拼成模型可用的 batch 字典
        # DistributedSampler：按索引给各进程不同样本；drop_last 保证各进程步数一致
        sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
    )
    LOGGER.info(f"{len(dataloader)} batches per epoch")

    # ZeRO：把优化器状态切分到各 GPU 以省显存；本章不保存/不恢复优化器 state（非完整 resume）
    optimizer = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.AdamW,  # 底层仍是 AdamW
        lr=args.lr,
        fused=True,  # 融合 CUDA kernel，通常更快
    )
    # 余弦退火：学习率从 args.lr 平滑降到 eta_min（T_max / eta_min 为示例取值）
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=args.lr * 1e-2
    )

    # 传了 --experiment-name 才开启存盘 / 续训
    is_experiment, exp_dir = resolve_experiment_dir(args)

    # state：训练进度（与权重分开存，方便续训时知道从哪一步接着跑）
    state = {
        "epoch": 0,  # 当前第几轮
        "global_step": 0,  # 全局已更新多少步（跨 epoch 累计）
        "epoch_step": 0,  # 本 epoch 内已走到第几步
    }
    # 两次打日志之间的 loss 累加在 GPU 上，避免每步 .item() 触发同步
    running_loss = torch.zeros((), device=device, dtype=torch.float32)

    if is_experiment:
        # 恢复模型权重、scheduler 与进度；load_optimizer=False → Adam 动量等会重新初始化
        state, resumed = resume_if_available(
            exp_dir,
            device,
            model,
            optimizer,
            lr_scheduler,
            state,
            load_optimizer=False,
        )
        # 只让 rank 0 打续训日志，避免刷屏
        if rank == 0:
            LOGGER.info(f"Resumed={resumed} | {state}")
    # 等所有进程读完（或跳过）checkpoint，再一起往下走
    dist.barrier()

    # 只由 rank 0 创建实验目录；前后 barrier 保证「先检查完再创建、创建完再训练」
    if is_experiment and rank == 0:
        ensure_experiment_dir(exp_dir)
    dist.barrier()

    # 从上次停的 epoch 继续，训到 num_epochs
    for state["epoch"] in range(state["epoch"], args.num_epochs):
        LOGGER.info(f"Begin epoch {state['epoch']} at step {state['epoch_step']}")

        # 每个 epoch 必须 set_epoch，DistributedSampler 的打乱顺序才会随 epoch 变化
        dataloader.sampler.set_epoch(state["epoch"])

        # 进度条只在 rank 0 显示，避免多进程抢同一终端
        progress_bar = tqdm.tqdm(range(len(dataloader)), disable=rank > 0)
        if state["epoch_step"] > 0:
            # 断点续训时，进度条先跳到上次停的位置
            progress_bar.update(state["epoch_step"])

        for i_step, batch in enumerate(dataloader):
            # 续训：跳过本 epoch 里已经训过的 step（仍会取 batch，以保持 sampler 顺序一致）
            if i_step < state["epoch_step"]:
                continue

            # 异步拷到本进程 GPU（需配合 pin_memory=True）；用到该 batch 时流上仍会等拷完
            batch = {
                k: v.to(device=device, non_blocking=True) for k, v in batch.items()
            }

            # 前向：算 logits 和 loss（Causal LM 用 labels 算交叉熵）
            outputs = model(**batch)
            # 尽早释放输入占用的显存（可选；大批次 / 长序列时有时能挤一点空间）
            del batch

            # 反向：算梯度；DDP 在此处通过 hook 做跨进程梯度归约（平均）
            outputs.loss.backward()

            # 各进程用已同步的梯度更新本地参数，再推进学习率
            optimizer.step()
            lr_scheduler.step()
            # set_to_none=True：把 .grad 置 None 省显存（PyTorch 2.0+ 默认已是 True）
            optimizer.zero_grad(set_to_none=True)

            state["global_step"] += 1
            state["epoch_step"] += 1
            # detach：不建图；留在 GPU 上累加，避免每步 .item() 同步
            running_loss += outputs.loss.detach().float()
            progress_bar.update(1)

            # 每隔 log_freq 步打一次日志（此时才 .item() 同步一次）
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
                # 清峰值显存计数（不释放显存），让下一段 peak_* 只反映本窗口
                torch.cuda.reset_peak_memory_stats(device)
                # 清零累加器，下一段 log_freq 步重新统计平均 loss
                running_loss.zero_()

            # 仅实验模式：每隔 ckpt_freq 步由 rank 0 写断点
            if is_experiment and state["global_step"] % args.ckpt_freq == 0:
                if rank == 0:
                    # 不写 optimizer.pt：省时间；续训时优化器状态会重初始化（非完整 resume）
                    save_checkpoint(
                        exp_dir,
                        model,
                        optimizer,
                        lr_scheduler,
                        state,
                        include_optimizer=False,
                    )
                # 等 rank 0 写完，其他进程再继续，避免读到半截文件
                dist.barrier()

        # 本 epoch 结束，下一轮从 step 0 开始
        state["epoch_step"] = 0

    # 释放进程组资源（训练正常结束时显式销毁）
    dist.destroy_process_group()


def get_mem_stats(device=None):
    """读取当前 GPU 显存占用（已分配 / 峰值 / 预留），单位 GB。"""
    mem = torch.cuda.memory_stats(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "total_gb": 1e-9 * props.total_memory,  # 卡总显存
        "curr_alloc_gb": 1e-9 * mem["allocated_bytes.all.current"],  # 当前已分配
        "peak_alloc_gb": 1e-9 * mem["allocated_bytes.all.peak"],  # 分配峰值
        "curr_resv_gb": 1e-9 * mem["reserved_bytes.all.current"],  # 当前缓存预留
        "peak_resv_gb": 1e-9 * mem["reserved_bytes.all.peak"],  # 预留峰值
    }


def _get_parser() -> argparse.ArgumentParser:
    """定义命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment-name", default=None)  # 实验名；有则开启存盘/续训
    parser.add_argument("-d", "--dataset-name", default=None, required=True)  # HF 数据集名
    parser.add_argument("--dataset-subset", default=None)  # 数据集子集（可选）
    parser.add_argument("-m", "--model-name", default=None, required=True)  # 模型配置/分词器名
    parser.add_argument("--save-dir", default="../outputs")  # checkpoint 根目录
    parser.add_argument("--seed", default=0, type=int)  # 随机种子
    parser.add_argument("--num-epochs", default=100, type=int)  # 训练轮数
    parser.add_argument("--lr", default=3e-5, type=float)  # 学习率
    parser.add_argument(
        "-b", "--batch-size", default=1, type=int
    )  # 每进程 batch（非全局）
    parser.add_argument("--log-freq", default=10, type=int)  # 每隔多少 step 打日志
    parser.add_argument("--ckpt-freq", default=500, type=int)  # 每隔多少 step 存 checkpoint
    parser.add_argument("-s", "--seq-length", default=1024, type=int)  # 序列长度
    return parser


# 直接 python / torchrun … train_llm.py 时才进 main；被 import 时不自动训练
if __name__ == "__main__":
    main()

import argparse
from itertools import chain
import json
import multiprocessing
import os
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader
import tqdm
import datasets
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

# 本模块专用的 logger，方便按模块过滤日志
LOGGER = logging.getLogger(__name__)


def main():
    # 解析命令行参数（模型名、数据集、学习率等）
    parser = _get_parser()
    args = parser.parse_args()

    # 后续版本会修改这里，加入 rank 信息。
    logging.basicConfig(
        format=f"[%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )

    # 在多节点运行时记录这些信息很有帮助，可以确认所有节点环境相同。
    LOGGER.debug(os.environ)
    LOGGER.debug(args)

    # 本指南假设 CUDA 设备可用，并且全部使用 bf16 训练。
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # 为 PyTorch 的随机数生成器设置种子。见 https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.seed)

    # 注意：这里初始化的是一个**未训练**的模型。
    model: torch.nn.Module
    # with device: 让新建的张量/参数默认放到该 GPU 上（省去再 .to(device)）
    with device:
        # 只加载模型配置（层数、隐层大小等），不下载权重；use_cache=False 关掉 KV Cache（训练不需要）
        config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
        # 按配置随机初始化一个 Causal LM（因果语言模型），权重是新的，不是预训练权重
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    LOGGER.info(
        f"Training {sum(p.numel() for p in model.parameters())} model parameters"
    )

    # torch.compile：把模型编译成更高效的执行图，通常能加速训练（首次会多花一点编译时间）
    model = torch.compile(model)

    LOGGER.info(f"Initialized model uses {get_mem_stats(device)['curr_alloc_gb']}gb")

    # 下载/加载数据集，分词并切成固定长度的训练样本
    train_data = _load_and_preprocess_data(args, config)
    LOGGER.debug(f"{len(train_data)} training samples")

    # 标准的 PyTorch 数据集迭代器。
    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,  # 每个 step 取多少条样本
        shuffle=True,  # 每个 epoch 打乱顺序
        drop_last=True,  # 丢掉最后不足一个 batch 的尾巴，保证每批大小一致
        num_workers=2,  # 额外开 2 个进程在后台取数 / collate（可与 GPU 计算重叠）
        prefetch_factor=2,  # 每个 worker 预取 2 个 batch（仅 num_workers>0 时有效）
        pin_memory=True,  # 页锁定内存，便于后面 non_blocking 异步 H2D
        collate_fn=default_data_collator,  # HF collator；不写则默认是 torch 的 default_collate
    )
    LOGGER.info(f"{len(dataloader)} batches per epoch")

    # AdamW：常用优化器；fused=True 用融合 CUDA kernel，通常更快
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)

    # 注意：T_max 和 eta_min 是任意选择的。
    # 余弦退火：学习率从 args.lr 平滑降到 eta_min
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=args.lr * 1e-2
    )

    # 「实验」= 给这次 run 起名，用来存 checkpoint / 断点续训。
    # 不传 --experiment-name：只训练、不落盘；传了：目录为 save_dir/实验名/
    is_experiment = False
    exp_dir: Path = Path(args.save_dir)
    if args.experiment_name is not None:
        is_experiment = True
        exp_dir = exp_dir / args.experiment_name  # 同名实验共享同一目录

    # state：训练进度（与权重分开存，方便续训时知道从哪一步接着跑）
    state = {
        "epoch": 0,  # 当前第几轮
        "global_step": 0,  # 全局已更新多少步（跨 epoch 累计）
        "epoch_step": 0,  # 本 epoch 内已走到第几步
    }
    # 两次打日志之间的 loss 累加在 GPU 上，避免每步 .item() 触发同步
    running_loss = torch.zeros((), device=device, dtype=torch.float32)
    resumed = False  # 本次是否从旧 checkpoint 恢复

    # 同名实验目录里已有 state.json → 视为续训：权重 + 进度一并读回
    if is_experiment and (exp_dir / "state.json").exists():
        # map_location：把保存时的张量加载到当前 device
        # weights_only=True：只反序列化张量，降低恶意 pickle 风险
        def _load_to_device(p):
            return torch.load(p, map_location=device, weights_only=True)

        model.load_state_dict(_load_to_device(exp_dir / "model.pt"))  # 模型权重
        optimizer.load_state_dict(_load_to_device(exp_dir / "optimizer.pt"))  # 优化器状态（动量等）
        lr_scheduler.load_state_dict(_load_to_device(exp_dir / "lr_scheduler.pt"))  # 调度器进度（当前 lr）
        with open(exp_dir / "state.json") as fp:
            state = json.load(fp)  # 覆盖上面的默认进度
        state.pop("running_loss", None)  # 旧 checkpoint 可能带该字段；累加器不落盘
        resumed = True
    if is_experiment:
        LOGGER.info(f"Resumed={resumed} | {state}")

    if is_experiment:
        LOGGER.info(f"Creating experiment root directory")
        # 确保存盘目录存在（续训时目录已在，exist_ok 避免报错）
        exp_dir.mkdir(parents=True, exist_ok=True)

    # 从上次停的 epoch 继续，训到 num_epochs
    for state["epoch"] in range(state["epoch"], args.num_epochs):
        LOGGER.info(f"Begin epoch {state['epoch']} at step {state['epoch_step']}")

        progress_bar = tqdm.tqdm(range(len(dataloader)))
        if state["epoch_step"] > 0:
            # 断点续训时，进度条先跳到上次停的位置
            progress_bar.update(state["epoch_step"])

        # 标准写法：直接遍历 DataLoader
        for i_step, batch in enumerate(dataloader):
            # 续训：跳过本 epoch 里已经训过的 step（仍会取到 batch，以保持打乱顺序一致）
            if i_step < state["epoch_step"]:
                continue

            # 异步拷到 GPU（需配合 DataLoader pin_memory=True）；用到该 batch 时流上仍会等拷完
            batch = {
                k: v.to(device=device, non_blocking=True) for k, v in batch.items()
            }

            # 前向：算 logits 和 loss（Causal LM 会用 labels 算交叉熵）
            outputs = model(**batch)
            # 可选：尽早释放输入占用的显存；不写通常也能在离开作用域后回收
            # 大批次、长序列、大模型，OOM 或峰值贴近上限时，早点丢掉输入有时能挤出一点空间
            del batch

            # 反向：梯度写到各参数的 .grad
            outputs.loss.backward()

            # 用当前 lr 更新参数，再把调度器推到下一步的 lr
            optimizer.step()
            lr_scheduler.step()
            # set_to_none=True：置 None 省显存（1.x 默认填 0，2.0+ 默认已是 True）
            optimizer.zero_grad(set_to_none=True)

            state["global_step"] += 1
            state["epoch_step"] += 1
            # detach：不建图；留在 GPU 上累加，一般不强制同步
            # 不要用 running_loss += outputs.loss.item()，因为这会每个step都同步一次，影响性能
            running_loss += outputs.loss.detach().float()
            progress_bar.update(1)

            # 每隔 log_freq 步打一次训练日志（此时才 .item() 同步一次）
            if state["global_step"] % args.log_freq == 0:
                info = {
                    "global_step": state["global_step"],
                    "lr": lr_scheduler.get_last_lr()[0],
                    "running_loss": (running_loss / args.log_freq).item(),  # 这里才开始同步一次
                    "epoch": state["epoch"],
                    "epoch_progress": state["epoch_step"] / len(dataloader),   
                    "num_batches_remaining": len(dataloader) - i_step,
                    **get_mem_stats(device),
                }
                LOGGER.info(info)
                # 清峰值显存计数（不释放显存、一般不触发计算同步），让下一段日志的 peak_* 只反映本窗口
                torch.cuda.reset_peak_memory_stats(device)
                # 清零累加器，下一段 log_freq 步重新统计平均 loss
                running_loss.zero_()

            # 仅实验模式：每隔 ckpt_freq 步写断点，同名实验可续训
            if is_experiment and state["global_step"] % args.ckpt_freq == 0:
                LOGGER.info("Saving checkpoint.")
                torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
                torch.save(model.state_dict(), exp_dir / "model.pt")
                torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
                with open(exp_dir / "state.json", "w") as fp:
                    json.dump(state, fp)  # 有 state.json 启动时才会走恢复逻辑

        # 本 epoch 结束，下一轮从 step 0 开始
        state["epoch_step"] = 0


def _load_and_preprocess_data(args, config):
    """
    Function created using code found in
    https://github.com/huggingface/transformers/blob/v4.45.1/examples/pytorch/language-modeling/run_clm_no_trainer.py
    """
    # 分词器：把文本变成 token id，需与模型词表匹配
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 从 Hugging Face Hub（或本地）加载数据集；subset 是子集名（有的数据集需要）
    data = datasets.load_dataset(args.dataset_name, args.dataset_subset)

    # 找出文本所在的列名（常见叫 "text"，否则用第一列）
    column_names = data["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # 对一批样本文本做分词
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    # map：对整份数据集并行分词；去掉原始文本列，只保留 token 相关字段
    tokenized_datasets = data.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=multiprocessing.cpu_count(),  # 开多个进程并行跑分词，进程数设为 CPU 核数，提高数据预处理效率
        load_from_cache_file=True,  # 有缓存则直接读，避免重复分词
        desc="Running tokenizer on dataset",
    )

    # 训练序列长度：命令行指定，或退回 tokenizer 默认最大长度
    seq_length = args.seq_length or tokenizer.model_max_length
    # 不能超过模型位置编码上限；过大时压到 min(1024, max_position_embeddings)
    if seq_length > config.max_position_embeddings:
        seq_length = min(1024, config.max_position_embeddings)

    # 主要的数据处理函数：拼接数据集中的全部文本，并按 seq_length 切分文本块。
    def group_texts(examples):
        # 拼接全部文本。
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # 总长度超过 seq_length 时，截去不足整块的尾部；正长度不足一块时，当前实现仍保留短块。
        # 如果模型支持填充，也可以改成填充，而不是丢弃；这部分可以按需要自定义。
        if total_length > seq_length:
            total_length = (total_length // seq_length) * seq_length
        # 按 seq_length 切成块。
        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated_examples.items()
        }
        # Causal LM 的标签通常就是 input_ids 本身（预测下一个 token）
        result["labels"] = result["input_ids"].copy()
        return result

    # 再 map 一次：把 token 流切成固定长度块，供训练直接用
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {seq_length}",
    )

    # 本脚本只用训练集
    return lm_datasets["train"]


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
    parser.add_argument("-b", "--batch-size", default=1, type=int)  # batch 大小
    parser.add_argument("--log-freq", default=10, type=int)  # 每隔多少 step 打日志
    parser.add_argument("--ckpt-freq", default=500, type=int)  # 每隔多少 step 存 checkpoint
    parser.add_argument("-s", "--seq-length", default=1024, type=int)  # 序列长度
    return parser


# 直接 python train_llm.py ... 运行时才进 main；被 import 时不自动训练
if __name__ == "__main__":
    main()

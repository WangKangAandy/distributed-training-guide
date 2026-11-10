import argparse
from itertools import chain
import json
import multiprocessing
import os
import time
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

LOGGER = logging.getLogger(__name__)


def main():
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
    with device:
        config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    LOGGER.info(
        f"Training {sum(p.numel() for p in model.parameters())} model parameters"
    )

    model = torch.compile(model)

    LOGGER.info(f"Initialized model uses {get_mem_stats(device)['curr_alloc_gb']}gb")

    train_data = _load_and_preprocess_data(args, config)
    LOGGER.debug(f"{len(train_data)} training samples")

    # 标准的 PyTorch 数据集迭代器。
    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=1,
        prefetch_factor=2,
        collate_fn=default_data_collator,
    )
    LOGGER.info(f"{len(dataloader)} batches per epoch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)

    # 注意：T_max 和 eta_min 是任意选择的。
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=args.lr * 1e-2
    )

    is_experiment = False
    exp_dir: Path = Path(args.save_dir)
    if args.experiment_name is not None:
        is_experiment = True
        exp_dir = exp_dir / args.experiment_name

    # 尝试恢复训练。
    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
        "running_loss": 0,
    }
    resumed = False
    if is_experiment and (exp_dir / "state.json").exists():
        # 注意：weights_only 用于防止 pickle 解码时执行任意代码。
        def _load_to_device(p):
            return torch.load(p, map_location=device, weights_only=True)

        model.load_state_dict(_load_to_device(exp_dir / "model.pt"))
        optimizer.load_state_dict(_load_to_device(exp_dir / "optimizer.pt"))
        lr_scheduler.load_state_dict(_load_to_device(exp_dir / "lr_scheduler.pt"))
        with open(exp_dir / "state.json") as fp:
            state = json.load(fp)
        resumed = True
    if is_experiment:
        LOGGER.info(f"Resumed={resumed} | {state}")

    if is_experiment:
        LOGGER.info(f"Creating experiment root directory")
        exp_dir.mkdir(parents=True, exist_ok=True)

    # 用来了解速度分别花在哪些部分。
    timers = {k: LocalTimer(device) for k in ["data", "forward", "backward", "update"]}

    for state["epoch"] in range(state["epoch"], args.num_epochs):
        LOGGER.info(f"Begin epoch {state['epoch']} at step {state['epoch_step']}")

        progress_bar = tqdm.tqdm(range(len(dataloader)))
        if state["epoch_step"] > 0:
            progress_bar.update(state["epoch_step"])

        # 注意：这不是标准写法。通常可以直接遍历 dataloader。
        #       这里手动取出下一个 batch，是为了明确测量生成一个 batch 所需的时间。
        batches = iter(dataloader)

        for i_step in range(len(dataloader)):
            # 这里测量生成一个 batch 并把它移到 GPU 所需的时间。
            with timers["data"], torch.no_grad():
                batch = next(batches)
                batch = {k: v.to(device=device) for k, v in batch.items()}

            # 为了恢复训练，这必须放在取出下一个 batch 之后，才能正确遍历数据集。
            if i_step < state["epoch_step"]:
                # 注意：用于恢复训练。
                continue

            with timers["forward"]:
                outputs = model(**batch)
                del batch

            with timers["backward"]:
                outputs.loss.backward()

            with timers["update"]:
                optimizer.step()
                lr_scheduler.step()
                # 注意：set_to_none=True 会释放梯度，从而节省一些内存。
                optimizer.zero_grad(set_to_none=True)

            state["global_step"] += 1
            state["epoch_step"] += 1
            state["running_loss"] += outputs.loss.item()
            progress_bar.update(1)

            if state["global_step"] % args.log_freq == 0:
                tok_per_step = args.batch_size * args.seq_length
                ms_per_step = sum(t.avg_elapsed_ms() for t in timers.values())
                info = {
                    "global_step": state["global_step"],
                    "lr": lr_scheduler.get_last_lr()[0],
                    "running_loss": state["running_loss"] / args.log_freq,
                    "epoch": state["epoch"],
                    "epoch_progress": state["epoch_step"] / len(dataloader),
                    "num_batches_remaining": len(dataloader) - i_step,
                    **get_mem_stats(device),
                    "tokens_per_s": 1000 * tok_per_step / ms_per_step,
                    "time/total": ms_per_step,
                    **{
                        f"time/{k}": timer.avg_elapsed_ms()
                        for k, timer in timers.items()
                    },
                }

                LOGGER.info(info)

                torch.cuda.reset_peak_memory_stats(device)
                state["running_loss"] = 0
                for t in timers.values():
                    t.reset()

            if is_experiment and state["global_step"] % args.ckpt_freq == 0:
                LOGGER.info("Saving checkpoint.")
                torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
                torch.save(model.state_dict(), exp_dir / "model.pt")
                torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
                with open(exp_dir / "state.json", "w") as fp:
                    json.dump(state, fp)

        state["epoch_step"] = 0


def _load_and_preprocess_data(args, config):
    """
    Function created using code found in
    https://github.com/huggingface/transformers/blob/v4.45.1/examples/pytorch/language-modeling/run_clm_no_trainer.py
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    data = datasets.load_dataset(args.dataset_name, args.dataset_subset)

    column_names = data["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = data.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    seq_length = args.seq_length or tokenizer.model_max_length
    if seq_length > config.max_position_embeddings:
        seq_length = min(1024, config.max_position_embeddings)

    # 主要的数据处理函数：拼接数据集中的全部文本，并生成长度为 block_size 的块。
    def group_texts(examples):
        # 拼接全部文本。
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # 丢弃不足一个块的剩余部分。如果 total_length < block_size，就排除这个 batch 并返回空字典。
        # 如果模型支持填充，也可以改成填充，而不是丢弃；这部分可以按需要自定义。
        if total_length > seq_length:
            total_length = (total_length // seq_length) * seq_length
        # 按 max_len 切成块。
        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {seq_length}",
    )

    return lm_datasets["train"]


def get_mem_stats(device=None):
    mem = torch.cuda.memory_stats(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "total_gb": 1e-9 * props.total_memory,
        "curr_alloc_gb": 1e-9 * mem["allocated_bytes.all.current"],
        "peak_alloc_gb": 1e-9 * mem["allocated_bytes.all.peak"],
        "curr_resv_gb": 1e-9 * mem["reserved_bytes.all.current"],
        "peak_resv_gb": 1e-9 * mem["reserved_bytes.all.peak"],
    }


class LocalTimer:
    def __init__(self, device: torch.device):
        if device.type == "cpu":
            self.synchronize = lambda: torch.cpu.synchronize(device=device)
        elif device.type == "cuda":
            self.synchronize = lambda: torch.cuda.synchronize(device=device)
        self.measurements = []
        self.start_time = None

    def __enter__(self):
        self.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if traceback is None:
            self.synchronize()
            end_time = time.time()
            self.measurements.append(end_time - self.start_time)
        self.start_time = None

    def avg_elapsed_ms(self):
        return 1000 * (sum(self.measurements) / len(self.measurements))

    def reset(self):
        self.measurements = []
        self.start_time = None


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

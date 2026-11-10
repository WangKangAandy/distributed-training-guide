# 单卡 GPU

这是一份「标准」的单卡训练脚本。它不做任何分布式操作，目标是尽量简单。

本指南后面的内容都以这份代码为基础，所以这一章会说明代码的各个部分，以及为什么要这样写。

## 命令

```bash
cd distributed-training-guide/01-single-gpu
python train_llm.py -d tatsu-lab/alpaca -m openai-community/gpt2
```

## 代码说明

说明大致按代码顺序，从文件顶部开始。

### 参数解析

训练脚本是一个命令行程序，也就是从终端运行。我们希望使用者能在命令行里修改一批参数。下面是很常见的 Python 写法：

```python
def main():
    parser = _get_parser()
    args = parser.parse_args()


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default=None, required=True)
    parser.add_argument("--dataset-name", default=None, required=True)
    parser.add_argument("--model-name", default=None, required=True)
    parser.add_argument("--save-dir", default="../outputs")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--log-freq", default=10, type=int)
    parser.add_argument("--ckpt-freq", default=500, type=int)
    return parser


if __name__ == "__main__":
    main()
```

### 配置日志

本指南只用 Python 内置的 `logging`。日志会输出到标准输出和标准错误，之后再用命令行工具把输出重定向到文件。

```python
LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    format=f"[%(asctime)s] %(levelname)s:%(message)s",
    level=logging.INFO,
)

LOGGER.info(os.environ)
LOGGER.info(args)
```

能看到程序运行时的环境变量和命令行参数很有用，后面涉及多台机器时尤其如此。所以这两项最先记录。

### PyTorch 初始化

使用 PyTorch 时，在初始化其他对象之前有几件有用的事要做：

```python
device = torch.device("cuda")
dtype = torch.bfloat16
torch.manual_seed(args.seed)
```

这里指定脚本其余部分使用 GPU（具体是 CUDA 设备），并用 bfloat16（也叫 bf16）训练。它是 16 位浮点数；float 是 32 位，double 是 64 位。

### 初始化模型

我们用 `transformers` 训练一个 BF16 因果语言模型（可以把它理解成 GPT）：

```python
config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
model = AutoModelForCausalLM.from_config(config, dtype=dtype).to(device)
```

### 初始化数据集

我们用 `datasets` 加载并预处理数据。本指南的处理代码来自 https://github.com/huggingface/transformers/blob/v4.45.1/examples/pytorch/language-modeling/run_clm_no_trainer.py。

如果想了解更多，建议读者直接查看 `datasets`。

### 数据加载、学习率调度和优化器

下一段是比较标准的 PyTorch：用 DataLoader 遍历数据集，优化器是 AdamW，学习率调度是余弦退火。

```python
dataloader = DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=default_data_collator,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=1000, eta_min=args.lr * 1e-2
)
```

注意：优化器的 `fused=True` 会使用融合内核。几乎所有情况下都更快，所以本指南每一章都打开它。

### 输出与恢复训练

检查点保存在 `args.save_dir/args.experiment_name`。`--experiment-name` 是一次运行的**唯一**标识。

```python
exp_dir: Path = Path(args.save_dir) / args.experiment_name
```

如果 `args.save_dir/args.experiment_name/state.json` 已经存在，就尝试恢复。也就是说，这个实验名已经有检查点时，我们把它解释为一次续训。

```python
state = {
    "epoch": 0,
    "global_step": 0,
    "epoch_step": 0,
    "running_loss": 0,
}
resumed = False
if (exp_dir / "state.json").exists():
    model.load_state_dict(_load_to_device(exp_dir / "model.pt"))
    optimizer.load_state_dict(_load_to_device(exp_dir / "optimizer.pt"))
    lr_scheduler.load_state_dict(_load_to_device(exp_dir / "lr_scheduler.pt"))
    with open(exp_dir / "state.json") as fp:
        state = json.load(fp)
    resumed = True
```

### 遍历 batch

这里没有用最常见的写法，是为了给训练循环的各部分计时。通常无法计时 batch 的实际构造过程；手动用 `next()` 取出下一个 batch 后，就可以计时：

```python
batches = iter(dataloader)

for i_step in range(len(dataloader)):
    # 这里测量生成一个 batch 并把它移到 GPU 所需的时间
    with timers["data"], torch.no_grad():
        batch = next(batches)
        batch = {k: v.to(device=device) for k, v in batch.items()}
```

### 前向、反向和参数更新

这是标准 PyTorch 代码，只是加了计时，方便做基准测试：

```python
with timers["forward"]:
    outputs = model(**batch)

with timers["backward"]:
    # set_to_none=True 会释放梯度占用的内存。
    optimizer.zero_grad(set_to_none=True)
    outputs.loss.backward()

with timers["update"]:
    optimizer.step()
    lr_scheduler.step()
```

### 输出到标准输出

接下来的代码记录训练进展的各项信息。

记录频率由 `--log-freq` 决定。例如 `--log-freq 100` 表示每 100 步记录一次。

注意：我们既写入 LOGGER，也写入 wandb。

```python
if state["global_step"] % args.log_freq == 0:
    info = {
        "global_step": state["global_step"],
        "lr": lr_scheduler.get_last_lr()[0],
        "running_loss": state["running_loss"] / args.log_freq,
        "epoch": state["epoch"],
        "epoch_progress": state["epoch_step"] / len(dataloader),
        "num_batches_remaining": len(dataloader) - i_step,
        "time/total": sum(t.avg_elapsed_ms() for t in timers.values()),
        **{
            f"time/{k}": timer.avg_elapsed_ms()
            for k, timer in timers.items()
        },
    }

    LOGGER.info(info)

    state["running_loss"] = 0
    for t in timers.values():
        t.reset()
```

### 检查点

最后一段是保存检查点的逻辑，这里直接使用 `torch.save`。

注意：除了模型，还要保存优化器和学习率调度器。

```python
if state["global_step"] % args.ckpt_freq == 0:
    LOGGER.info("Saving checkpoint.")
    torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
    torch.save(model.state_dict(), exp_dir / "model.pt")
    torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
    with open(exp_dir / "state.json", "w") as fp:
        json.dump(state, fp)
```

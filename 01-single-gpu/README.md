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

> 译注：上面的参数解析是原文的简化示例。实际脚本中 `--experiment-name` 可省略，`--dataset-name` 和 `--model-name` 分别支持 `-d`、`-m` 别名；完整参数以 `python train_llm.py --help` 为准。

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

能看到程序运行时的环境变量和命令行参数很有用，后面涉及多台机器时尤其如此。所以示例把这两项放在最前面记录。

> 译注：实际脚本对这两项使用 `LOGGER.debug(...)`，默认 INFO 级别不会显示；需要查看时应将日志级别设为 DEBUG。

### PyTorch 初始化

使用 PyTorch 时，在初始化其他对象之前有几件有用的事要做：

```python
device = torch.device("cuda")
dtype = torch.bfloat16
torch.manual_seed(args.seed)
```

这里指定脚本其余部分使用 GPU（具体是 CUDA 设备），并用 bfloat16（也叫 bf16）训练。它是 16 位浮点数；float 是 32 位，double 是 64 位。

### 初始化模型

我们用 `transformers` 训练一个 BF16 因果语言模型（可以把它理解成 GPT）。脚本里用 `with device:` 在 GPU 上直接建模型，而不是先在 CPU 建好再 `.to(device)`：

```python
with device:
    # 只加载配置，不下载权重；use_cache=False 关掉 KV Cache（训练不需要）
    config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
    # 按配置随机初始化（本指南练的是未训练模型，不是加载预训练权重）
    model = AutoModelForCausalLM.from_config(config, dtype=dtype)
```

`with device:` 是 PyTorch 的设备上下文：块内**新创建**的张量/参数默认建在该 device 上，所以 `from_config` 时权重一开始就在 CUDA。

和 `.to(device)` 比，主要好处：

| | `with device:` 里直接建 | 先建再 `.to(device)` |
|---|---|---|
| 显存 | 只在 GPU 上占一份 | 常先在 CPU 建好，再拷到 GPU，峰值≈两份 |
| 速度 | 少一次整模拷贝 | 大模型搬迁更慢 |
| 写法 | 创建时就定好设备 | 创建 + 再搬一次 |

大模型时「先 CPU 再 `.to("cuda")`」很容易短暂爆内存；`with device:` 从源头就在 GPU 上分配，更省峰值。

注意：这只优化**初始化**（更快、峰值更低），**不加速后续训练**。模型已经在 CUDA 上之后，forward / backward 速度与用 `.to(device)` 搬过去的相同。它也只影响块内新建的对象；DataLoader 在 CPU 上拼出的 batch，训练循环里仍要 `.to(device, non_blocking=True)`（见下文）。

### torch.compile

模型建好之后，脚本还会做一步：

```python
model = torch.compile(model)
```

很多「传统」教程 / 老脚本**不会**写这一行。原因很简单：`torch.compile` 是 PyTorch 2.0 才正式推广的；在那之前，训练循环里直接 `outputs = model(**batch)` 就是标准写法，功能完全够用。本指南从单卡开始就打开它，是为了让后面各章在相近写法下尽量吃到吞吐收益。

它做什么：把模型的前向（以及参与编译的反向路径）编译成更高效的执行图（默认后端多为 TorchInductor），减少 Python 开销、融合部分算子。多数情况下能**加速训练**，属于「改一行就可能变快」的优化。

代价与注意：

- **首次**跑若干 step 会偏慢（要追踪、编译）；编译完成后才是稳态速度。看吞吐时别拿最初几步当基准。
- 需要较新的 PyTorch（建议 2.x）和可用的 GPU 编译环境；出奇怪报错时，可先注释掉这一行做对比排查。
- 它加速的是**已在 GPU 上的计算**；和上一节的 `with device:` 不同——后者只影响建模时的设备与峰值显存，不加速 step。

更激进的用法（连 `loss_function`、`optimizer.step` 一起 compile）会出现在后面章节，例如 [05-training-llama-405b](../05-training-llama-405b/README.md#torchcompile)。单卡这一章先掌握「整模 `torch.compile(model)`」即可。

### 初始化数据集

我们用 `datasets` 加载并预处理数据。本指南的处理代码来自 https://github.com/huggingface/transformers/blob/v4.45.1/examples/pytorch/language-modeling/run_clm_no_trainer.py。

如果想了解更多，建议读者直接查看 `datasets`。

### 数据加载（DataLoader）

```python
dataloader = DataLoader(
    train_data,
    batch_size=args.batch_size,  # 每个 step 的样本数
    shuffle=True,  # 每个 epoch 打乱顺序
    drop_last=True,  # 丢掉不足一整批的尾巴，保证 batch 大小一致
    num_workers=2,  # 后台进程取数，与 GPU 计算重叠
    prefetch_factor=2,  # 每个 worker 预取几个 batch（需 num_workers>0）
    pin_memory=True,  # 页锁定内存，配合 .to(..., non_blocking=True) 做异步 H2D
    collate_fn=default_data_collator,  # 多条样本如何拼成一个 batch
)
```

- `shuffle`：训练一般 `True`；验证 / 测试用 `False`。要对齐复现时，除了 `torch.manual_seed`，最好再给 DataLoader 传固定的 `generator`；`num_workers>0` 时还要设 `worker_init_fn`，否则 batch 顺序仍可能对不齐。详见 [determinism](../related-topics/determinism/README.md)。
- `num_workers`：后台取数进程数；`prefetch_factor`：每个进程提前准备几个 batch。默认相当于 `num_workers=0`（主进程同步取数），GPU 容易空等。适当增大后，取数可与计算重叠，通常能加快训练中的数据加载。多机场景见 [优化数据加载](../related-topics/optimizing-data-loading/README.md)。
- `pin_memory`：默认 `False`。设为 `True` 后，batch 落在页锁定 CPU 内存，主机→GPU 拷贝可以走更快的 DMA，也是 `non_blocking` 异步拷贝的前提之一。
- `collate_fn`：不写则用 PyTorch 的 `default_collate`；这里显式用 HF 的 `default_data_collator`，和 Transformers 字段更合拍。

### 优化器与学习率调度

训练一步的核心是：算出 loss → `backward` 得到梯度 → **优化器**按梯度改参数。优化器决定「怎么根据梯度更新权重」。

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
```

- `model.parameters()`：要更新的模型参数。
- `lr`：基础学习率（步长）；太大易不稳，太小收敛慢。
- `fused`：默认 `False`；`True` 走融合 CUDA kernel，更新通常更快。

学习率不必全程固定。**学习率调度器**（lr_scheduler）在训练过程中调整 `optimizer` 的 lr 。这里用余弦退火：从当前 lr 按余弦曲线降到 `eta_min`。

```python
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=1000, eta_min=args.lr * 1e-2
)
```

- `optimizer`：要调度哪一个优化器。
- `T_max`：余弦半周期的 step 数；此处为示例取值。
- `eta_min`：学习率下限，这里取 `args.lr * 1e-2`。

训练循环里常见顺序是：`optimizer.step()` 用当前学习率更新参数，再 `lr_scheduler.step()` 调度器步数 +1，按余弦规则算出下一步的学习率并写回优化器。不调用后者的话，学习率会一直停在初始值。

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
}
# loss 在 GPU 上累加，不放进 state（避免每步 .item() 同步，也无法直接写入 JSON）
running_loss = torch.zeros((), device=device, dtype=torch.float32)
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

标准写法是直接 `enumerate(dataloader)`，每步取出一个 batch 再搬到 GPU：

```python
for i_step, batch in enumerate(dataloader):
    if i_step < state["epoch_step"]:
        continue

    batch = {
        k: v.to(device=device, non_blocking=True) for k, v in batch.items()
    }
```

- `enumerate`：同时拿到本 epoch 内的步号 `i_step` 和数据 `batch`。
- `i_step < state["epoch_step"]` 时 `continue`：断点续训用。`epoch_step` 记录中断前已训到哪一步；前面的步只跳过训练，但 `enumerate` 仍会按同样顺序取到 batch，这样在 `shuffle=True` 时后面接着训的样本顺序才对得上。
- `.to(device, non_blocking=True)`：按 batch 从 CPU 拷到 GPU（不是整库常驻显存）。`non_blocking=True` **让拷贝异步提交，常与 DataLoader 的** `pin_memory=True` **一起用，便于和 GPU 计算重叠**；GPU 真正用到这些张量时，流上仍会等拷完。注意这里的 `.items()` 是字典遍历，不是会强制同步的 `tensor.item()`。

### 前向、反向和参数更新

标准的一步训练：

```python
outputs = model(**batch)
outputs.loss.backward()

optimizer.step()
lr_scheduler.step()
optimizer.zero_grad(set_to_none=True)
```

- `model(**batch)`：前向。把 `input_ids` / `labels` 等送进模型，算出 logits 和 `loss`（Causal LM 会按下一个 token 预测算交叉熵）。
- `loss.backward()`：反向。沿计算图反传，把梯度写到各参数的 `.grad`。
- `optimizer.step()`：用**当前**学习率和刚算出的梯度更新权重。
- `lr_scheduler.step()`：调度器步数 +1，按规则把学习率改成**下一步**要用的值。
- `optimizer.zero_grad(set_to_none=True)`：清掉 `.grad`；`set_to_none = True` 置 `None` 更省显存（1.x 默认填 0，2.0+ 默认已是 `True`）。

顺序记成：前向 → 反向 → 更新参数 → 推进学习率 → 清梯度，准备下一步。

### 输出到标准输出

记录频率由 `--log-freq` 决定。例如 `--log-freq 100` 表示每 100 步记录一次（step、lr、平均 loss、显存等）。

每步用 `running_loss += outputs.loss.detach().float()` 在 GPU 上累加，一般不强制同步；只在打日志时 `.item()` 一次，把同步次数从「每步」降到「每 log_freq 步」。

```python
running_loss += outputs.loss.detach().float()

if state["global_step"] % args.log_freq == 0:
    info = {
        "global_step": state["global_step"],
        "lr": lr_scheduler.get_last_lr()[0],
        "running_loss": (running_loss / args.log_freq).item(),
        "epoch": state["epoch"],
        "epoch_progress": state["epoch_step"] / len(dataloader),
        "num_batches_remaining": len(dataloader) - i_step,
        **get_mem_stats(device),
    }

    LOGGER.info(info)
    running_loss.zero_()
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

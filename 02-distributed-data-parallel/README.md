# 单节点多 GPU

本章目标：从单卡训练扩展到**单节点多 GPU 数据并行**，讲清三件事——`torchrun` 的进程启动、`DistributedDataParallel` 的梯度同步、`DistributedSampler` 的数据分片。

训练循环本身还是前向、反向、更新；变化是由多个进程一起跑这份代码。

脚本同样拆成三份，与第 1 章同构、各自独立（不跨章 import）：

```text
├── train_llm.py    # 分布式启动、DDP / ZeRO、DistributedSampler、训练循环
├── data.py         # 加载 / 分词 / 切块（与第 1 章同构）
└── checkpoint.py   # 实验目录、保存与恢复（与第 1 章同构；ZeRO 时由 train_llm 跳过优化器落盘）
```

主线假定 config / tokenizer / **原始数据集文件**已在训练启动前准备好；各进程用 `local_files_only=True` 独立读取（禁止启动时联网下载）。注意：这不等于 tokenization / grouping 已完成——`map` 缓存未命中时，首次启动仍会做预处理；命中后才会跳过。checkpoint 的 rank0 写盘与 barrier、ZeRO 跳过优化器仍留在 `train_llm.py`。

## 1. 从单卡训练到多 GPU 训练

上一章用一个 Python 进程和一张 GPU 完成训练：读 batch、前向、算梯度、更新参数。

模型变大、或希望缩短训练时间时，可以用多张 GPU 同时算。本章采用的 DDP（DistributedDataParallel）方式是：**每张 GPU 一个进程**，而不是一个进程同时去调多张卡。

以一台 4 卡机器为例：

| | 单卡训练 | 单节点多 GPU（DDP） |
|---|---|---|
| 进程 | 1 个 | 4 个 |
| GPU | 1 张 | 每个进程 1 张 |
| 数据 | 一个进程看完整数据流 | 各进程看不同样本 |
| 梯度 | 只在本进程里算 | 反向时在进程间同步 |

每个进程持有一份完整模型副本，独立做前向和反向。要让副本保持一致，需要同时满足：

1. 模型和优化器都能放进每张 GPU。
2. 各进程用同样的方式初始化模型。
3. 各进程用同样的优化器设置。
4. 更新前，大家用的是同一份（已同步的）梯度。

训练一步仍然是：

```python
outputs = model(**batch)  # 前向
outputs.loss.backward()   # 反向；DDP 在这里同步梯度
optimizer.step()          # 各进程用同步后的梯度更新本地参数
lr_scheduler.step()
optimizer.zero_grad(set_to_none=True)
```

`backward()` 时，DDP 通过 autograd hook 做梯度归约（先按进程数平均，再求和），并尽量让通信和反向计算重叠。`optimizer.step()` 并不负责发起这次同步，它只用已经同步好的梯度。依据见 [PyTorch DDP 设计说明](https://docs.pytorch.org/docs/stable/notes/ddp.html)。

<img width="800" alt="DDP 一步训练" src="https://github.com/user-attachments/assets/a988f513-a5b3-4e5f-bf0d-019e796ee4e0" />

图里的同步发生在参数更新之前。

于是要分开解决两件事：

- 谁来启动多个进程，并告诉每个进程自己的编号？→ `torchrun`
- 谁在训练时同步梯度、谁切分数据？→ DDP 与 `DistributedSampler`

三者不能互相替代。`torchrun` 不会自动把单卡脚本变成 DDP。

## 2. torchrun 是什么

`torchrun` 是 PyTorch 自带的分布式启动器（launcher），等价于：

```bash
python -m torch.distributed.run
```

它按配置拉起多个训练进程，并写入分布式环境变量。它不做前向、反向，也不做梯度同步。

GPU 训练里常见的对应关系是一个进程一张卡：

```text
torchrun
  ├── Worker 0 ── GPU 0
  ├── Worker 1 ── GPU 1
  ├── Worker 2 ── GPU 2
  └── Worker 3 ── GPU 3
```

这些 Worker 跑同一份 `train_llm.py`。进程编号不同，所以绑定的 GPU 不同、读到的数据也不同。

## 3. 使用 torchrun 启动多 GPU 训练

单卡是直接起一个进程：

```bash
python train_llm.py
```

4 张 GPU 时，用 `torchrun` 起 4 个进程：

```bash
torchrun \
    --standalone \
    --nproc-per-node=4 \
    train_llm.py
```

- `torchrun`：启动器。
- `--standalone`：单节点，自动配好本机 rendezvous（进程互相发现的会合点）。多节点的会合方式见[任务启动器](../03-job-launchers/)。
- `--nproc-per-node=4`：本机启动 4 个进程。单节点时通常等于 GPU 数。写成 `gpu` 表示用上当前机器全部 GPU。
- `train_llm.py`：每个进程执行的脚本。

只换成 `torchrun` 还不够。脚本里仍要初始化进程组、绑定 GPU、用 DDP 包模型，并用 `DistributedSampler` 分数据。

本章实际命令还会带上数据集、日志和调试相关设置：

```bash
cd distributed-training-guide/02-distributed-data-parallel
export TORCHELASTIC_ERROR_FILE=../error.json
export OMP_NUM_THREADS=1
torchrun \
    --standalone \
    --nproc-per-node=gpu \
    --redirects 3 \
    --log-dir ../logs \
    train_llm.py \
    -d tatsu-lab/alpaca \
    -m openai-community/gpt2
```

- `--redirects 3`：把标准输出和标准错误重定向到文件。
- `--log-dir ../logs`：日志目录。
- `TORCHELASTIC_ERROR_FILE`：某个 worker 出错时，把错误写到这个文件。脚本里的 `main` 需要 `@record`（来自 `torch.distributed.elastic.multiprocessing.errors`）才会记下完整堆栈。
- `OMP_NUM_THREADS`：多进程时每个进程默认都想占满 CPU。按「CPU 核数 / GPU 数」设一个上限，避免互相抢核。例如 32 核、8 卡，可以设为 4。上面示例用 `1` 是为了先排除 CPU 过度订阅。

弹性训练（进程挂了再拉起、节点动态加入）以及 rendezvous 后端的细项，本章不展开。多节点启动见第 3 章，弹性相关见 [elastic-training](../related-topics/elastic-training/README.md)。

## 4. torchrun 启动后发生了什么

执行 `torchrun --standalone --nproc-per-node=4 train_llm.py` 后，会有 4 个进程各自跑一遍脚本，并带上这些环境变量：

| 环境变量 | 含义 | 单节点 4 GPU |
|---|---|---|
| `LOCAL_RANK` | 本节点内的编号，用来选 GPU | 0、1、2、3 |
| `RANK` | 整个任务里的全局编号 | 0、1、2、3 |
| `WORLD_SIZE` | 进程总数 | 4 |

单节点上 `LOCAL_RANK` 和 `RANK` 通常相同。多节点上就不一样了：`LOCAL_RANK` 仍是本机 0 起编，`RANK` 在全局连续编号。两台机器、每台 4 进程时：

```text
Node 0                    Node 1
LOCAL_RANK=0  RANK=0      LOCAL_RANK=0  RANK=4
LOCAL_RANK=1  RANK=1      LOCAL_RANK=1  RANK=5
LOCAL_RANK=2  RANK=2      LOCAL_RANK=2  RANK=6
LOCAL_RANK=3  RANK=3      LOCAL_RANK=3  RANK=7
WORLD_SIZE = 8
```

脚本用 `LOCAL_RANK` 绑定 GPU，不要用 `RANK % 卡数` 去猜（多节点时会绑错）：

```python
import os
import torch

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
```

绑好设备之后，再 `init_process_group`，并用 DDP 包装模型。

## 5. 使用 DDP 改造训练代码

### <a id="calling-distinit_process_group-and-torchcudaset_device"></a>调用 `dist.init_process_group()` 和 `torch.cuda.set_device()`

调用其他 `dist` API 之前，这两个都必须先调用。

`dist.init_process_group()` 会阻塞，直到 `WORLD_SIZE` 个进程都调用了它。

```diff
+from torch import distributed as dist

 def main():
     parser = _get_parser()
     args = parser.parse_args()

+   rank = int(os.environ["RANK"])
+   local_rank = int(os.environ["LOCAL_RANK"])
+   world_size = int(os.environ["WORLD_SIZE"])

-   device = torch.device(f"cuda") 
+   device = torch.device(f"cuda:{local_rank}")
+   torch.cuda.set_device(device)

+   dist.init_process_group(
+       backend="nccl",
+       rank=rank,
+       world_size=world_size,
+       device_id=device,
+   )
```

- `backend="nccl"`：GPU 上常用的分布式通信后端。
- `device_id`：告诉进程组当前进程对应哪张 GPU（需与项目使用的 PyTorch 版本一致；较新版本支持该参数）。

如果不调用 `torch.cuda.set_device`，进程可能不会使用正确的 CUDA 设备。

### <a id="including-rank-in-logging-statements"></a>在日志中包含 rank

所有进程可能写到同一个文件；即使只查看一份日志，每条记录带上 rank 也更方便：

```diff
 logging.basicConfig(
-    format=f"[%(asctime)s] %(levelname)s:%(message)s",
+    format=f"[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s",
     level=logging.INFO,
 )
```

### <a id="using-distributeddataparallel"></a>使用 DistributedDataParallel

各进程独立读本地 config、在对应 GPU 上建模型，再用 DDP 包装。本章是 `from_config` **从头训练**（随机初始化），不加载预训练权重；SFT / 续训需要另走权重加载路径。

```diff
+from torch.nn.parallel import DistributedDataParallel

 with device:
-    config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
+    config = AutoConfig.from_pretrained(
+        args.model_name, local_files_only=True, use_cache=False
+    )
     model = AutoModelForCausalLM.from_config(config, dtype=dtype)

+model = DistributedDataParallel(model, device_ids=[local_rank])
```

你可能会以为 DDP 模块负责把 batch 分给各个进程，其实完全不是。

它是一个模型包装类，确保**调用 `optimizer.step()` 之前先同步梯度**。建议阅读文档，内容很有帮助：https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html。

这个类还保证**构造时各进程的模型参数相同**。

它通过一些[特殊的模型钩子](https://github.com/pytorch/pytorch/blob/v2.4.1/torch/nn/parallel/distributed.py#L939)，对所有 rank 的梯度进行归约；下面先除以进程数再求和，得到的是平均梯度：

```python
# 注意：以下为 PyTorch 内部代码，见 https://github.com/pytorch/pytorch/blob/v2.4.1/torch/nn/parallel/distributed.py#L939
gradient = param.grad / self.process_group.size()
gradient = fcol.all_reduce(gradient, "sum", self.process_group)
```

### <a id="using-distributedsampler"></a>使用 DistributedSampler

每个进程仍持有**完整模型副本**，但不应再各自扫完整数据集。否则等价于把同一批样本算了 `world_size` 遍，全局有效 batch 和优化轨迹都会乱掉。

`DistributedSampler` 并不把数据集物理切成多份文件，而是为每个进程生成**不同的样本索引**。以 1 000 条样本、4 张 GPU 为例（示意，实际还会 shuffle / drop_last）：

```text
原始数据集索引：0 … 999

Rank 0 → 约 0–249
Rank 1 → 约 250–499
Rank 2 → 约 500–749
Rank 3 → 约 750–999
```

让它生效的代码改动非常少：

```diff
+from torch.utils.data.distributed import DistributedSampler

 dataloader = DataLoader(
     train_data,
     batch_size=args.batch_size,  # 每进程 batch size
-    shuffle=True,
-    drop_last=True,
     collate_fn=default_data_collator,
+    sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
 )
```

#### 全局 Batch Size

`--batch-size` / `args.batch_size` 是**每个进程**一次前向看到的样本数，不是整个任务的全局 batch。

```text
Global Batch Size ≈ per_device_batch × world_size × grad_accum_steps
```

例如 8 GPU、每卡 batch 4、梯度累积 2 步：`8 × 4 × 2 = 64`。增加 GPU 数而不改每卡 batch 时，全局 batch 会跟着变大；调学习率或对比实验时要意识到这一点。本章示例未做梯度累积，因此全局 batch ≈ `batch_size × world_size`。

还要调用 [DistributedSampler.set_epoch](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)。PyTorch 文档的说明是：

```diff
 for state["epoch"] in range(state["epoch"], args.num_epochs):
+    dataloader.sampler.set_epoch(state["epoch"])
     batches = iter(dataloader)
```

> 在分布式模式下，每个 epoch 开始、创建 DataLoader 迭代器之前必须调用 `set_epoch()`，多个 epoch 之间的打乱才能正常工作。否则会一直使用相同顺序。

### 数据预处理仍可能发生在首次启动

`local_files_only=True` 只保证不联网下载缺失文件；`load_from_cache_file=True` 只在 **map 缓存已存在** 时跳过 tokenization / grouping。缓存未命中时，每个 rank 仍会执行 `data.map(...)`。

另外，多进程同时把 `num_proc` 设成整机 CPU 核数会过度订阅。本章按 `cpu_count() // world_size` 限制每个 rank 的 map 并行度。更稳妥的工程做法是把 Tokenization / Packing 做成独立的数据准备阶段，训练启动只读已处理好的样本。

### <a id="downloading-model--data-in-rank-0-first"></a>进阶：未预置制品时再用 rank 协调

`rank0_first()` **不是** DDP 训练的必要步骤。DDP 主线是：绑 GPU → `init_process_group` → 各进程初始化模型 → DDP 包装 → `DistributedSampler` → 反向时同步梯度。

真实工程里，模型 config / tokenizer / 数据集通常在**训练任务准备阶段**就放到本地盘、共享存储或对象存储；`torchrun` 启动后各 Worker 只读已有文件。大模型制品可达数十 GB 以上，更不应让每个 Worker 在启动时竞争下载。

本章主线因此使用 `local_files_only=True`，并假定制品已就绪。只有在「任务启动时仍必须往共享缓存里写」的特殊场景，才需要让一个进程先写、其余进程再读，例如：

```python
@contextmanager
def rank0_first():
    rank = dist.get_rank()
    if rank == 0:
        yield
    dist.barrier()
    if rank > 0:
        yield
    dist.barrier()
```

这与 checkpoint 的「仅 rank 0 写盘 + barrier」不同：后者是训练过程中的落盘协调，前者是制品准备缺失时的应急手段。

### <a id="only-creating-experiment-directory-on-rank-0"></a>只在 rank 0 创建实验目录

注意创建目录前后的 `dist.barrier()`。**它们非常重要！**

创建实验目录之前会检查目录是否已经存在，因此必须保证**所有进程都已经完成这次检查**。第一个 `dist.barrier()` 就是为了等所有 worker 检查完。只有这时，rank 0 才能创建目录。

```diff
+if rank == 0:
     exp_dir.mkdir(parents=True, exist_ok=True)
+dist.barrier()
```

### <a id="save-checkpoint-on-rank-0"></a>只由 rank 0 保存检查点

只让一个 rank 保存检查点。否则多个 rank 可能写同一个文件并互相破坏。

```diff
 if state["global_step"] % args.ckpt_freq == 0:
+    if rank == 0:
         save_checkpoint(..., include_optimizer=False)
+    dist.barrier()
```

## <a id="optimizing-memory---zero-redundancy-optimizer"></a>6. 优化显存：Zero Redundancy Optimizer

DDP 在每张 GPU 上都保存完整模型和优化器。对优化器来说尤其浪费。好在可以直接加入 [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) 来降低显存占用：

```diff
+ from torch.distributed.optim import ZeroRedundancyOptimizer

-optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
+optimizer = ZeroRedundancyOptimizer(
+    model.parameters(), optimizer_class=torch.optim.AdamW, lr=args.lr
+)
```

保存 ZeRO 的优化器 state 很慢，本章因此 **`include_optimizer=False` / `load_optimizer=False`**：会恢复模型权重、学习率调度器与训练进度，但 **AdamW 的动量等优化器状态会重新初始化**。这不是严格意义上的完整断点续训，后续轨迹可能与从未中断的训练不同。若需要完整 ZeRO resume，应使用专门的分布式 checkpoint 机制（聚合优化器分片后再保存），而不是简单跳过。

## <a id="how-multi-node-works"></a>7. 多节点如何工作

它和单节点多 GPU 的工作方式基本相同。单节点时已经有多个进程，多节点只是把额外进程放到不同机器上。

主要需要考虑：

1. 如何让每个节点保持相同环境。
2. 节点之间如何互相发现（`torchrun` 命令中的 `rdzv` 参数）。
3. 每个节点如何访问数据。

超过一个节点后，错误报告和处理变得极其重要。网络问题很常见，还必须保证机器之间的一些细微配置完全一致。

## <a id="shared-storage---managing-your-python-virtual-environment-across-nodes"></a>8. 共享存储：在节点之间管理 Python 虚拟环境

最简单的方法是把 Python 虚拟环境创建在所有节点都能访问的共享网络盘上。这样所有节点使用完全相同的 Python 可执行文件和环境。

创建方式和普通虚拟环境一样，只是目录必须共享。

## <a id="shared-storage---mangaging-your-datasetmodel-checkpoints-across-nodes"></a>9. 共享存储：在节点之间管理数据集和模型检查点

同样，最简单的方法是把数据放在共享网络盘。需要注意，共享网络盘的读取通常比节点本地盘慢。如果数据加载变慢，可以把数据或模型复制到节点本地存储。

使用 `transformers` 或 `datasets` 时，设置 `$HF_HOME` 来控制 Hugging Face 下载数据集和模型权重的位置。

## <a id="hf_home---the-downloaded-modeldataset-directory"></a>10. `$HF_HOME`：已下载的模型与数据集目录

Hugging Face 的 `transformers` 和 `datasets` 默认把内容放到 `$HF_HOME`（默认为**节点本地**目录）。推荐做法仍是：训练启动前把需要的 config、tokenizer、数据集准备好；训练脚本用 `local_files_only=True` 只读。

若暂时无法预置、必须在任务内首次写入缓存，由**部署配置**声明缓存是否跨节点共享，再决定谁先写（不要用 `os.path.ismount` 去猜网络盘）：

```python
# 由训练任务 / 集群配置明确指定，而不是运行时探测文件系统
shared_cache = args.shared_cache  # True：跨节点共享缓存；False：每节点本地缓存
should_go_first = (rank == 0) if shared_cache else (local_rank == 0)

# 与上文 rank0_first 相同：先让 should_go_first 的进程跑完，再放行其余进程
if should_go_first:
    train_data = load_and_preprocess_data(args, config)
dist.barrier()
if not should_go_first:
    train_data = load_and_preprocess_data(args, config)
dist.barrier()
```

## 11. 本章小结

- 本章定位：单节点多 GPU 数据并行——讲透 `torchrun`、DDP、`DistributedSampler` 三件事。
- `torchrun` 启动进程并提供 `RANK` / `LOCAL_RANK` / `WORLD_SIZE`；DDP 在反向时同步梯度；`DistributedSampler` 按索引分片，每卡仍有完整模型副本。
- `--batch-size` 是每进程 batch；全局 batch ≈ `batch_size × world_size`（再乘梯度累积步数，若有）。
- 制品准备与训练启动分层：`local_files_only` 禁止启动时下载；tokenization 缓存未命中时仍可能在首次启动执行 `map`。
- 本章 ZeRO checkpoint 恢复权重与进度，但不恢复优化器状态，不是完整 resume。
- 单节点上用 `LOCAL_RANK` 绑定 GPU。多节点启动、会合和容错放到后面的章节。

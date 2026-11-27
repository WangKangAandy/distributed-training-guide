# 单节点多 GPU

本章在[第 1 章](../01-single-gpu)单卡脚本上，改成单节点多 GPU 的数据并行训练。训练循环本身还是前向、反向、更新；变化是由多个进程一起跑这份代码。

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

+   dist.init_process_group(rank=rank, world_size=world_size, device_id=device)
```

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

```diff
+from torch.nn.parallel import DistributedDataParallel

 with device: 
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

普通训练脚本用 `torch.utils.data.DataLoader` 组 batch。DataLoader 的一个参数是 `sampler`，构造 batch 时从数据集中取样。可以把 sampler 理解成：

```python
def simple_sampler():
    worker_len = len(dataset)
    return random.choice(range(worker_len))
```

DistributedSampler 的巧妙之处是按 worker 数量划分数据集长度。它甚至不需要真正切分数据集，只是从数据集的一个特定子集中选择要返回的整数：

```python
def distributed_sampler():
    worker_len = len(dataset) // dist.get_world_size()
    return dist.get_rank() * worker_len + random.choice(range(worker_len))
```

让它生效的代码改动非常少：

```diff
+from torch.utils.data.distributed import DistributedSampler

 dataloader = DataLoader(
     train_data,
     batch_size=args.batch_size,
-    shuffle=True,
-    drop_last=True,
     collate_fn=default_data_collator,
+    sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
 )
```

还要调用 [DistributedSampler.set_epoch](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)。PyTorch 文档的说明是：

```diff
 for state["epoch"] in range(state["epoch"], args.num_epochs):
+    dataloader.sampler.set_epoch(state["epoch"])
     batches = iter(dataloader)
```

> 在分布式模式下，每个 epoch 开始、创建 DataLoader 迭代器之前必须调用 `set_epoch()`，多个 epoch 之间的打乱才能正常工作。否则会一直使用相同顺序。

### <a id="downloading-model--data-in-rank-0-first"></a>先由 rank 0 下载模型和数据

这主要是因为加载数据时可能下载或预处理数据并写入磁盘。

如果不让 rank 0 先做，所有 rank 可能同时下载，整体会变慢。

**注意：更好的做法是提前把数据下载并预处理到共享网络盘。**

可以用一个简单的上下文管理器：

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

下载模型权重和分词器：

```diff
+with rank0_first():
     config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
     with device:
         model = AutoModelForCausalLM.from_config(config, dtype=dtype)
```

下载数据：

```diff
+with rank0_first():
     train_data = _load_and_preprocess_data(args, tokenizer, config)
```

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
         torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
         torch.save(model.state_dict(), exp_dir / "model.pt")
         torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
         with open(exp_dir / "state.json", "w") as fp:
              json.dump(state, fp)
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

可惜保存 ZeRO 状态字典的代码极其慢，所以还要去掉优化器状态字典的保存。

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

Hugging Face 的 `transformers` 和 `datasets` 默认把内容下载到 `$HF_HOME`。`$HF_HOME` 的默认值是**节点本地**目录。这里有两个选择：

1. 保持 `$HF_HOME` 为节点本地，并把 `with rank0_first()` 改成 `with local_rank0_first()`。
2. 把 `$HF_HOME` 改成共享网络盘。

第三种选择需要修改本仓库代码，在代码里自动判断：

```python
@contextmanager
def rank_ordered(first: bool):
    if first:
        yield
    dist.barrier()
    if not first:
        yield
    dist.barrier()

# 判断 HF_HOME 是节点本地目录还是共享目录
hf_home_is_networked = os.path.ismount(os.environ["HF_HOME"])

if hf_home_is_networked:
    # 目录共享时只让 rank 0 先执行，下载只发生在 rank 0
    should_go_first = rank == 0
else:
    # 目录是节点本地时，让该节点上的一个进程下载（local rank 0）
    should_go_first = local_rank == 0

with rank_ordered(should_go_first):
    train_data = _load_and_preprocess_data(args, tokenizer, config)
```

## 11. 本章小结

- 从单卡到多 GPU，训练循环没有换掉，而是变成多个进程各跑一份。
- `torchrun` 负责启动进程并提供 `RANK` / `LOCAL_RANK` / `WORLD_SIZE`；DDP 在反向时同步梯度；`DistributedSampler` 负责各进程读不同数据。
- 单节点上用 `LOCAL_RANK` 绑定 GPU。多节点启动、会合和容错放到后面的章节。

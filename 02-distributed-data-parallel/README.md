# 单节点多 GPU

**注意：本章代码基于[第 1 章](../01-single-gpu)。**

单节点命令：

```bash
cd distributed-training-guide/02-distributed-data-parallel
export TORCHELASTIC_ERROR_FILE=../error.json
export OMP_NUM_THREADS=1
torchrun \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir ../logs \
    train_llm.py \
    -d tatsu-lab/alpaca \
    -m openai-community/gpt2
```

多节点请看[任务启动器一章](../03-job-launchers/)。

快速跳转：
- [分布式训练如何工作](#how-distributed-training-works)
- [使用 torchrun](#using-torchrun-instead-of-python)
- [代码改动](#code-changes)
    - [调用 dist.init_process_group() 和 torch.cuda.set_device()](#calling-distinit_process_group-and-torchcudaset_device)
    - [在日志中包含 rank](#including-rank-in-logging-statements)
    - [DistributedDataParallel（DDP）](#using-distributeddataparallel)
    - [DistributedSampler](#using-distributedsampler)
    - 与 I/O 相关的保护
        - [先由 rank 0 下载模型和数据](#downloading-model--data-in-rank-0-first)
        - [只在 rank 0 操作文件系统](#only-creating-experiment-directory-on-rank-0)
        - [只由 rank 0 保存检查点](#save-checkpoint-on-rank-0)
- [优化显存：Zero Redundancy](#optimizing-memory---zero-redundancy-optimizer)
- [多节点如何工作](#how-multi-node-works)
- [共享存储：在节点之间管理 Python 虚拟环境](#shared-storage---managing-your-python-virtual-environment-across-nodes)
- [共享存储：在节点之间管理数据集和模型检查点](#shared-storage---mangaging-your-datasetmodel-checkpoints-across-nodes)
- [`$HF_HOME`：已下载的模型与数据集目录](#hf_home---the-downloaded-modeldataset-directory)

## 术语

- `world size`：参与训练的 GPU 总数。
- `rank`：这个 worker 的全局唯一编号，范围是 `0` 到 `world_size - 1`（含两端）。
- `local_rank`：这台机器内部的编号，范围是 `0` 到 `torch.cuda.device_count() - 1`（含两端）。

## 分布式训练如何工作

在讲分布式训练需要改什么之前，先想一下原理。做过并行计算的话就会知道，一种并行方式是把工作量分到所有核心上。当每份待处理任务大体相同时，这非常有效。Python 的 `multiprocessing.Pool.map` 就是这样做的。

GPU 上的分布式训练其实一样：我们把工作量（数据集中的 batch）分到多张 GPU。但还多一个问题：如何保证所有 GPU 上的模型相同？

可以用一种很巧妙的方式做到。为了简化，假设：

1. 模型和优化器都能完整放进每张 GPU。
2. 所有 GPU 上的模型初始化方式完全相同。
3. 所有 GPU 上的优化器设置完全相同。

现在看训练循环。PyTorch 中典型写法是：

```python
loss = model(**batch) # 1. 异步前向
optimizer.zero_grad() # 2. 异步清空梯度
loss.backward()       # 3. 异步计算梯度
optimizer.step()      # 4. 同步梯度并更新权重
```

前三行都可以异步完成。`loss.backward()` 会在每个训练进程里计算梯度。巧妙之处在于，`optimizer.step()` 会先在所有进程之间同步梯度，再真正更新模型参数。

下面是使用这种数据并行时，一个训练步骤的高层示意图：

<img width="800" alt="image" src="https://github.com/user-attachments/assets/a988f513-a5b3-4e5f-bf0d-019e796ee4e0" />

明确地说：**`optimizer.step()` 是所有进程的同步点**。

PyTorch 如何实现这一点？

1. [用 torchrun 运行训练脚本的 N 个副本。](#using-torchrun-instead-of-python)
2. [用 DistributedSampler 在 worker 之间切分数据。](#using-distributedsampler)
3. [用 DistributedDataParallel 同步梯度。](#using-distributeddataparallel)

## 用 `torchrun` 代替 `python`

用 `torchrun` 启动分布式训练时，它会**启动 N 个独立进程**（N 是 `--nproc-per-node` 指定的 GPU 数量），每个进程都运行同一份训练脚本：

```
> torchrun --nproc-per-node 3 train_llm.py ...
Launches subproc `$RANK=0 $WORLD_SIZE=3 train_llm.py ...`
Launches subproc `$RANK=1 $WORLD_SIZE=3 train_llm.py ...`
Launches subproc `$RANK=2 $WORLD_SIZE=3 train_llm.py ...`
```

它还会在进程之间建立同步。之后每个进程都运行相同的训练代码，并需要在若干位置同步。每个进程获得一个编号（`rank`），用来决定使用哪张设备。

在多节点上运行时，每台机器都要运行 `torchrun`，除此之外完全相同。具体做法见[任务启动器一章](../03-job-launchers/)。

本指南常用的 `torchrun` 命令行参数：

- `--standalone`：只在单节点运行时使用。
- `--nnodes`：使用的节点数。这里是 1；进入多节点后会大于 1。
- `--nproc-per-node`：进程数。`gpu` 表示使用全部可用 GPU。
- `--redirects 3`：把标准输出和标准错误重定向到文件。
- `--log-dir ../logs`：设置日志目录。

#### TORCHELASTIC_ERROR_FILE

**调试时一定要设置它！**

某个 worker（包括 worker 进程中的线程）出错时，`torchrun` 会把错误保存到这个环境变量指定的文件。

还要给 `main` 函数加上 `@record`（从 `torch.distributed.elastic.multiprocessing.errors` 导入 `record`）：

```diff
+@record
 def main():
```

#### OMP_NUM_THREADS

即使计算发生在 GPU 上，PyTorch 默认也会尽量使用所有可用 CPU 核心。现在有多个进程在运行 PyTorch，如果不把 `OMP_NUM_THREADS` 改成其他值，每个进程都会试图占用全部核心。

可以先查看可用核心数，再按进程分配。例如机器有 32 个核心、8 张 GPU，就可以把 `OMP_NUM_THREADS` 设为 4。

## 代码改动

### 调用 `dist.init_process_group()` 和 `torch.cuda.set_device()`

调用其他 `dist` API 之前，这两个都必须先调用。

`dist.init_process_group()` 会阻塞，直到 `WORLD_SIZE` 个进程都调用了它。

```diff
+from torch import distributed as dist

 def main():
     parser = _get_parser()
     args = parser.parse_args()

+   rank = int(os.getenv("RANK", "0"))
+   local_rank = rank % torch.cuda.device_count()
+   world_size = int(os.getenv("WORLD_SIZE", "1"))

-   device = torch.device(f"cuda") 
+   device = torch.device(f"cuda:{local_rank}")
+   torch.cuda.set_device(device)

+   dist.init_process_group(rank=rank, world_size=world_size, device_id=device)
```

如果不调用 `torch.cuda.set_device`，进程可能不会使用正确的 CUDA 设备。

### 在日志中包含 rank

所有进程可能写到同一个文件；即使只查看一份日志，每条记录带上 rank 也更方便：

```diff
 logging.basicConfig(
-    format=f"[%(asctime)s] %(levelname)s:%(message)s",
+    format=f"[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s",
     level=logging.INFO,
 )
```

### 使用 DistributedDataParallel

```diff
+from torch.nn.parallel import DistributedDataParallel

 with device: 
     model = AutoModelForCausalLM.from_config(config, dtype=dtype)

+model = DistributedDataParallel(model, device_ids=[local_rank])
```

你可能会以为 DDP 模块负责把 batch 分给各个进程，其实完全不是。

它是一个模型包装类，确保**调用 `optimizer.step()` 之前先同步梯度**。建议阅读文档，内容很有帮助：https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html。

这个类还保证**构造时各进程的模型参数相同**。

它通过一些[特殊的模型钩子](https://github.com/pytorch/pytorch/blob/v2.4.1/torch/nn/parallel/distributed.py#L939)，把所有 rank 的梯度在每个 rank 上相加：

```python
# NOTE: internal pytorch code found at https://github.com/pytorch/pytorch/blob/v2.4.1/torch/nn/parallel/distributed.py#L939
gradient = param.grad / self.process_group.size()
gradient = fcol.all_reduce(gradient, "sum", self.process_group)
```

### 使用 DistributedSampler

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

### 先由 rank 0 下载模型和数据

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

### 只在 rank 0 创建实验目录

注意创建目录前后的 `dist.barrier()`。**它们非常重要！**

创建实验目录之前会检查目录是否已经存在，因此必须保证**所有进程都已经完成这次检查**。第一个 `dist.barrier()` 就是为了等所有 worker 检查完。只有这时，rank 0 才能创建目录。

```diff
+if rank == 0:
     exp_dir.mkdir(parents=True, exist_ok=True)
+dist.barrier()
```

### 只由 rank 0 保存检查点

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

## 优化显存：Zero Redundancy Optimizer

DDP 在每张 GPU 上都保存完整模型和优化器。对优化器来说尤其浪费。好在可以直接加入 [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) 来降低显存占用：

```diff
+ from torch.distributed.optim import ZeroRedundancyOptimizer

-optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
+optimizer = ZeroRedundancyOptimizer(
+    model.parameters(), optimizer_class=torch.optim.AdamW, lr=args.lr
+)
```

可惜保存 ZeRO 状态字典的代码极其慢，所以还要去掉优化器状态字典的保存。

## 多节点如何工作

它和单节点多 GPU 的工作方式基本相同。单节点时已经有多个进程，多节点只是把额外进程放到不同机器上。

主要需要考虑：

1. 如何让每个节点保持相同环境。
2. 节点之间如何互相发现（`torchrun` 命令中的 `rdzv` 参数）。
3. 每个节点如何访问数据。

超过一个节点后，错误报告和处理变得极其重要。网络问题很常见，还必须保证机器之间的一些细微配置完全一致。

## 共享存储：在节点之间管理 Python 虚拟环境

最简单的方法是把 Python 虚拟环境创建在所有节点都能访问的共享网络盘上。这样所有节点使用完全相同的 Python 可执行文件和环境。

创建方式和普通虚拟环境一样，只是目录必须共享。

## 共享存储：在节点之间管理数据集和模型检查点

同样，最简单的方法是把数据放在共享网络盘。需要注意，共享网络盘的读取通常比节点本地盘慢。如果数据加载变慢，可以把数据或模型复制到节点本地存储。

使用 `transformers` 或 `datasets` 时，设置 `$HF_HOME` 来控制 Hugging Face 下载数据集和模型权重的位置。

## `$HF_HOME`：已下载的模型与数据集目录

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

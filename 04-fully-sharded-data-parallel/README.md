# 在 GPU 之间分片

**注意：本章代码基于[第 2 章](../02-distributed-data-parallel/)的代码。**

到目前为止，我们都假设模型和优化器都能完整放进单张 GPU。因此训练时每张 GPU 都保存模型和优化器的完整副本。

模型变大后这会成为问题：要么模型本身放不下，要么优化器放不下。优化器占用的显存通常是模型的 1 到 4 倍。

分片（sharding）是把优化器状态、梯度和/或模型参数的**存储**分散到各张 GPU 上。**层的执行方式不会改变**。

这意味着：

1. FSDP 单元执行前，需要通过 all-gather 收集该单元的**完整参数**；计算后按配置重新分片。反向传播通过 reduce-scatter 归约并分片梯度，优化器使用本地参数分片及对应状态更新。

   > 译注：原文把梯度和优化器状态也列为每层需要完整收集的对象，这不符合 [FSDP2 的工作方式](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)。启用 CPU offload 时，优化器更新在 CPU 上执行。
2. 每层前后的取消分片和重新分片都有同步成本。
3. 分片不会降低最大一层的峰值显存占用。

分片是一种**数据并行**技术，**不是**模型并行、张量并行或流水线并行。

## PyTorch FullyShardedDataParallel（FSDP）

<img width="800" alt="image" src="https://github.com/user-attachments/assets/1c470f25-2e48-4786-9588-1b28bc42344b" />

从高层看，FSDP 的工作方式是：

- 构造时：
    - 分片模型参数，每个 rank 只保留自己的分片。
- 前向路径：
    - 执行 all_gather，从所有 rank 收集分片，恢复这个 FSDP 单元的完整参数。
    - 执行前向计算。
    - 丢弃刚刚收集到的参数分片。
- 反向路径：
    - 执行 all_gather，从所有 rank 收集分片，恢复这个 FSDP 单元的完整参数。
    - 执行反向计算。
    - 执行 reduce_scatter 同步梯度。
    - 丢弃参数。

PyTorch 文档中的流程说明：

![image](https://pytorch.org/assets/images/fsdp_workflow.png)

参考资料：

- [FSDP 文档](https://pytorch.org/docs/stable/fsdp.html)
- [FSDP 教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

### 延迟初始化：`meta` 设备

[meta 设备文档](https://pytorch.org/docs/stable/meta.html)

meta 设备完全不分配内存，因此模型初始化极快。FSDP 使用它，是因为我们希望分片完成之后再初始化权重。

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, dtype=dtype)
```

### 给 meta 模型分片

这里使用 PyTorch 的 FSDP2 API：[fully_shard()](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)。

它会把仍位于 meta 设备上的模型拆分到所有 GPU。

```python
fsdp_config = dict(
    reshard_after_forward=True,
    offload_policy=CPUOffloadPolicy() if args.cpu_offload else None,
)
for decoder in model.model.layers:
    fully_shard(decoder, **fsdp_config)
fully_shard(model, **fsdp_config)
```

这里有一个重要细节：分片应用在**解码器层**这一级。这有什么影响？

**`fully_shard()` 会在你放置它的位置插入一次 all-gather。**

因此，每个解码器层执行前都会对该层做一次 all-gather。如果希望 all-gather 发生在其他位置，就需要以不同方式调用 `fully_shard()`。

#### 参数何时重新分片

如顶部图所示，参数通常在前向结束后立即重新分片。这是标准做法，因为前向过程中可以节省更多 GPU 显存。

如果设置 `reshard_after_forward=False`，则要等到反向传播结束后才重新分片。

#### <a id="cpu-offload"></a>CPU 卸载

摘自 [CPUOffloadPolicy 文档](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.CPUOffloadPolicy)：

> 这个卸载策略会把参数、梯度和优化器状态卸载到 CPU。

这个选项会**大幅**降低显存需求，但代价是大量计算和内存带宽。前向和反向仍在 GPU 上运行，然后梯度被卸载到 CPU，优化器也在 CPU 上运行：

> 反向传播时，分片梯度会从设备复制到主机，并且 **`optimizer.step()` 在 CPU 上运行**，使用的是 CPU 上的优化器状态。

注意：CPU offload **不会消除单个 FSDP 单元计算时完整参数和中间激活的显存需求**。不过，常驻 GPU 的分片状态减少后，总峰值显存仍可能下降。

> 译注：原文“不会降低峰值 GPU 显存”的说法过于绝对；这里区分单层计算所需显存与整个训练过程的总峰值。

### 初始化模型

先用 `to_empty` 分配内存，再用标准的 `reset_parameters()` 重置所有参数：

```python
model.to_empty(device="cpu" if args.cpu_offload else device)
model.apply(
    lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
)
```

注意：RotaryEmbedding 等模块拥有的是 buffer 而不是 parameter。`to_empty()` 会释放它们，但 `reset_parameters()` 不会重置它们，必须手动处理：

```python
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

def reset_rope(self: LlamaRotaryEmbedding):
    self.inv_freq, self.attention_scaling = self.rope_init_fn(
        self.config, self.inv_freq.device
    )
    self.original_inv_freq = self.inv_freq


LlamaRMSNorm.reset_parameters = lambda self: torch.nn.init.ones_(self.weight)
LlamaRotaryEmbedding.reset_parameters = reset_rope
```

### 分片检查点

模型参数可能分布在多张 GPU 上，因此保存检查点的方式也要改变。最快、显存开销最低的是分片检查点：每张 GPU 只保存自己当前持有的分片。我们推荐使用分片检查点。

需要的导入如下：

```python
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint import load, save
```

另外要配置 [StateDictOptions](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.StateDictOptions)，因为多处都会使用它：

```python
# NOTE: full_state_dict=False 表示保存分片检查点。
ckpt_opts = StateDictOptions(full_state_dict=False, cpu_offload=True)
```

如果设置 `full_state_dict=True`，保存的就是完整状态字典。

#### 保存分片检查点

与普通检查点明显不同的是，**每个 rank 都必须保存**。因为这是分片检查点，需要收集每个 rank 的分片。

```python
if state["global_step"] % args.ckpt_freq == 0:
    dist.barrier()
    # NOTE: 必须在所有 rank 上调用
    sharded_model_state, sharded_optimizer_state = get_state_dict(
        model, optimizer, options=ckpt_opts
    )
    save(
        dict(model=sharded_model_state, optimizer=sharded_optimizer_state),
        checkpoint_id=exp_dir / "checkpoint",
    )
```

[torch.distributed.checkpoint.state_dict.get_state_dict()](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict) 接收普通模型和优化器，并提取包含分片检查点的状态字典。

然后调用 [torch.distributed.checkpoint.state_dict_saver.save()](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_saver.save) 保存这个分片检查点。

运行结束后，指定目录中会为每个 rank 生成一个文件。

#### 加载分片检查点

加载分片检查点稍微复杂一些，因为需要在检查点的不同格式之间来回转换。

首先，像保存时一样调用 [torch.distributed.checkpoint.state_dict.get_state_dict()](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict)。它会按保存时的方式构造分片检查点字典：

```python
sharded_model_state, sharded_optimizer_state = get_state_dict(
    model, optimizer, options=ckpt_opts
)
```

接着调用与 save 相对的 [torch.distributed.checkpoint.state_dict_loader.load()](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_loader.load)。此时分片状态字典会包含之前保存的内容。

```python
load(
    dict(model=sharded_model_state, optimizer=sharded_optimizer_state),
    checkpoint_id=exp_dir / "checkpoint",
)
```

最后，用 [torch.distributed.checkpoint.state_dict.set_state_dict()](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_state_dict) 把这些分片检查点应用到实际模型参数：

```python
set_state_dict(
    model,
    optimizer,
    model_state_dict=sharded_model_state,
    optim_state_dict=sharded_optimizer_state,
    options=ckpt_opts,
)
```

#### 把分片检查点转换成完整状态字典检查点

如果要在格式之间转换，例如从分片格式转换为完整状态字典，PyTorch 已经提供了相应工具。指南在这里：

https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html#formats

## 运行命令

命令与平常相同：

```bash
cd distributed-training-guide/04-fully-sharded-data-parallel
export TORCHELASTIC_ERROR_FILE=../error.json
export OMP_NUM_THREADS=1
export HF_HOME=../.cache
torchrun --nproc-per-node gpu \
    --redirects 3 \
    --log-dir ../logs \
    train_llm.py \
    -d tatsu-lab/alpaca \
    -m openai-community/gpt2 \
    --cpu-offload
```

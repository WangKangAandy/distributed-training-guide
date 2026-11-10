# 训练 405B 模型

**注意：本章代码基于[第 4 章的 FSDP 代码](../04-fully-sharded-data-parallel/)。**

这里使用一个 8 节点集群（64 张 H100 GPU）训练 Llama 3.1 405B。**这里不使用 LoRA！** 我们确实是在普通 PyTorch 中完整训练一个 405B 模型的权重。

接下来几节说明，为了训练 405B 模型，需要对第 4 章 FSDP 代码做哪些修改。

快速跳转：

- [使用 Flash Attention](#use-flash-attention)
- [下载模型权重](#download-model-weights)
- [加载预训练权重](#loading-pretrained-weights)
- [给 Llama 405B 分片](#sharding-llama-405b)
- [梯度（也就是激活）检查点](#gradient-aka-activation-checkpointing)
- [CPU 卸载与融合优化器内核](#cpu-offload--fused-optimizer-kernels)
- [不要释放梯度](#not-de-allocating-gradients)
- [启动命令](#launch-command)
- [监控](#monitoring)
- [运行统计](#run-statistics)

## <a id="use-flash-attention"></a>使用 Flash Attention

Flash Attention 是缩放点积注意力的融合实现，能大幅降低显存占用。它的核心目标是尽量少访问内存，并减少临时内存。

更多信息见[代码仓库](https://github.com/Dao-AILab/flash-attention)和[论文](https://arxiv.org/abs/2205.14135)。

它最终能在前向和反向中节省数十 GB 显存。

安装：

```bash
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation
```

初始化模型时启用它：

```python
model = AutoModelForCausalLM.from_pretrained(
    ...
    attn_implementation="flash_attention_2",
)
```

## <a id="download-model-weights"></a>下载模型权重

实际模型权重非常大：共有 191 个独立文件，每个约 4GB，总计约 764GB。

存储这些权重有两种方式，而且差别很大：

1. 所有节点都能访问的共享网络盘。
2. 只放在主 rank 0 节点的本地存储。

初始化时，节点本地存储**快得多**。在 8 台 8×H100 80GB 节点上运行这个脚本时，共享网络盘初始化需要 50 分钟，节点本地存储只需要 3 分钟。

仓库里提供了一个下载脚本。在节点 0 上运行：

```bash
cd distributed-training-guide/05-training-llama-405b
python download.py
```

在其他节点上运行下面的命令，只下载配置和分词器：

```bash
cd distributed-training-guide/05-training-llama-405b
python download.py --skip-model
```

注意：通常需要先用 `hf auth login` 登录 Hugging Face 账号。

## <a id="loading-pretrained-weights"></a>加载预训练权重

### 把模型加载到 CPU 内存

首先在 **rank 0** 上把完整模型加载到 CPU 内存。稍后用它把所有权重广播给所有 rank。注意这里使用 `from_pretrained()`。

```python
if rank == 0:
    with torch.device("cpu"):
        full_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=dtype
        )
        full_sd = full_model.state_dict()
else:
    full_model = None
    full_sd = {}
dist.barrier()
```

真正加载权重时既耗时，也占用大量内存。完整大小约 764GB，因此必须确保有足够内存存放权重。

### 在所有 rank 上初始化 meta 模型

对于即将分片的实际模型，使用 `meta` 设备初始化。这是为了兼容马上要用的 `set_model_state_dict()` API：

```python
with rank0_first(), torch.device("meta"):
    config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
    model = AutoModelForCausalLM.from_config(
        config,
        dtype=dtype,
        attn_implementation="flash_attention_2",
    )
```

### 调用 `fully_shard()`

做法与上一章相同。一个细微差别是，顶层模型设置 `reshard_after_forward=False`。因为它是最后执行的层，反向传播可以立即使用这些权重。

```python
fsdp_config = dict(
    offload_policy=CPUOffloadPolicy() if args.cpu_offload else None,
    mp_policy=MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32),
)
for decoder in model.model.layers:
    fully_shard(decoder, reshard_after_forward=True, **fsdp_config)
fully_shard(model, reshard_after_forward=False, **fsdp_config)

model.to_empty(device="cpu" if args.cpu_offload else device)
dist.barrier()
```

### 从 rank 0 广播权重

[`set_model_state_dict()`](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.set_model_state_dict) 负责广播全部权重。`broadcast_from_rank0` 表示 rank 0 把 `full_sd` 广播给所有 rank。记住，`full_sd` 只存在于 rank 0。

```python
set_model_state_dict(
    model=model,
    model_state_dict=full_sd,
    options=StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=args.cpu_offload,
    ),
)
```

可惜这还不够，因为 PyTorch 有一个小特性：buffer 不包含在模型的 `state_dict()` 中。因此 RotaryEmbedding 等层不会得到正确数据，因为它们不在状态字典里。必须额外手动广播 rank 0 的 buffer：

```python
if rank == 0:
    for weight, buffer in zip(full_model.buffers(), model.buffers()):
        buffer.copy_(weight)
        dist.broadcast(buffer.to(device), src=0)
else:
    for buffer in model.buffers():
        device_buffer = buffer.to(device)
        dist.broadcast(device_buffer, src=0)
        buffer.copy_(device_buffer.to(buffer.device))
```

最后，为了清理 CPU 内存，可以释放不再需要的 `full_model`：

```python
del full_sd
# 强制释放模型的简便方法
if rank == 0:
    full_model.to(torch.device("meta"))
del full_model
```

## <a id="sharding-llama-405b"></a>给 Llama 405B 分片

上一节已经按解码器层调用了 `fully_shard()`。顶层模型使用 `reshard_after_forward=False`，解码器层使用 `reshard_after_forward=True`。

## <a id="gradient-aka-activation-checkpointing"></a>梯度（也就是激活）检查点

降低显存占用的另一项技术是梯度检查点，最早见于 [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)。正常前向时，必须把输入和输出保留到反向执行。保存这些中间张量会占用大量显存。使用梯度检查点后，反向期间会**重新执行**前向来再生输出。因此计算更多，但能节省大量显存。

这里使用的是 PyTorch 中一个相对不显眼的方法，但它正是 [Accelerate 底层使用的方法](https://github.com/huggingface/accelerate/blob/v0.34.2/src/accelerate/accelerator.py#L1492)，因此可以认为这是一种「标准」做法。

这段代码必须放在 `fully_shard` 相关代码**之后**。原因不完全明确，但放在 FSDP 初始化之前不会正常工作。

```python
fully_shard(...)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    auto_wrap_policy=functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )
)
```

## <a id="cpu-offload--fused-optimizer-kernels"></a>CPU 卸载与融合优化器内核

模型非常大，因此基本上必须启用 FSDP 的 [CPU 卸载](../04-fully-sharded-data-parallel/README.md#cpu-offload)。**使用 FSDP 的 CPUOffload 功能时，优化器完全在 CPU 上运行。** 这是因为执行 `optimizer.step()` 时，在 GPU 和 CPU 之间传输数据的成本很高。撰写本文时，如何让 `optimizer.step()` 与下一次 `forward()` 重叠仍有未解决的问题。

默认情况下，优化器在 CPU 上运行时使用非融合内核，会产生大量中间张量。显式使用融合内核可以获得很大加速；既然这一步运行在 CPU 上，这一点尤其重要：

```python
torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
```

如果想查看 PyTorch 源码：

1. [_single_tensor_adamw()](https://github.com/pytorch/pytorch/blob/v2.4.1/torch/optim/adamw.py#L322) 是默认实现。
2. [_fused_adamw()](https://github.com/pytorch/pytorch/blob/v2.4.1/torch/optim/adamw.py#L612) 是融合实现。

## torch.compile

这是一个容易获得吞吐量提升的改动：

```python
model = torch.compile(model)
model.loss_function = torch.compile(model.loss_function)
optimizer.step = torch.compile(optimizer.step)
```

## <a id="not-de-allocating-gradients"></a>不要释放梯度

你可能见过 [optimizer.zero_grad()](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html) 的 `set_to_none` 参数。文档说：

> 这通常会降低内存占用，并可能适度提升性能。

简单来说，`set_to_none=True` 会在梯度使用后**释放梯度**。在大多数想节省一点显存的 GPU 场景中，释放梯度是好事。但这里使用 CPU 卸载，所有梯度已经在 CPU 上。既然不占用 GPU 显存，设置 `set_to_none=True` 就只是反复付出分配和释放的成本。因此在这个场景中，把 `set_to_none=False` 反而会略微加速。

```python
optimizer.zero_grad(set_to_none=not args.cpu_offload)
```

## <a id="launch-command"></a>启动命令

相对基础的 [FSDP 代码](../04-fully-sharded-data-parallel/)，需要的改动就是这些。现在启动训练。

这里提供了定制的 [launch.sh](./launch.sh)，它基于在所有可用节点上启动 `torchrun` 的 Bash 命令：

```bash
cd distributed-training-guide/05-training-llama-405b
bash launch.sh # 注意：这条命令不会阻塞
```

另外，`launch.sh` 在 tmux 会话中把 `HF_HOME` 设为环境变量。如果没有使用默认值 `/home/ubuntu/.cache/huggingface`，请修改脚本。

可以修改本目录 [hosts](./hosts) 文件中的主机名。

## <a id="monitoring"></a>监控

[launch.sh](./launch.sh) 使用 `torchrun`，因此每个节点都会有一个输出目录，其中还有多个包含日志文件的子目录。逐个查看比较麻烦，可以用下面的 Bash 命令同时跟踪全部日志：

```bash
cd distributed-training-guide/05-training-llama-405b
find ../logs/ -name \*stderr.log | xargs tail -f
```

此外，仓库顶层有一个类似 `top` 的工具，用来监控整个集群：

```bash
cd distributed-training-guide/05-training-llama-405b
python ../top-cluster.py hosts
```

如果发现某个节点的进程数下降，或者功耗下降，就说明发生了错误。

要停止所有节点上的进程，直接关闭 tmux 会话：

```bash
xargs -a hosts -I{} ssh {} tmux kill-session -t torchrun-llama-405b
```

## <a id="run-statistics"></a>运行统计

在 64 张 H100 GPU（8 个独立节点）上，使用 `--seq-length 4096` 和 `--batch-size 1` 训练时，统计如下：

- 每次迭代（数据、前向、反向、更新）约 30 秒。细分是：
  - data：约 2ms
  - forward：约 7 秒
  - backward：约 19 秒
  - update：约 4 秒
- 峰值已分配显存：52.9GB
- 峰值预留显存：77.9GB

**注意：** 这些数字来自 PyTorch 2.5.1 和本指南的旧版本，因此可能已经过时。64 张 H100 也不容易随处找到 😄

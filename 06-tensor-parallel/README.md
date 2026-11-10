# 张量并行（Tensor Parallelism，TP）

到目前为止，我们只用了数据并行技术。你可能还听过其他并行技术。实际上，[Llama 405B 论文](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)在训练 405B 模型时使用了 4D 并行：

1. 数据并行（前面学过的 FSDP）
2. 张量并行（**本章**）
3. 上下文并行（用于长上下文）
4. 流水线并行 / 模型并行

本章先深入了解张量并行是什么，然后再考虑把它和其他并行方式组合。

## 基础：什么是张量并行？

TP 把模型权重**和**计算都拆分到多张 GPU 上。

FSDP 也会拆分模型权重，但计算前会把它们重新收集起来。是否把计算拆到多张 GPU，就是两者的区别。

结果是，world size 会按张量并行大小**缩小**，从而降低 all-gather 和 all-reduce 的成本。集群很大时，这会成为重要因素，TP 也是一种非常有效的扩展方式。

优点：

1. 峰值 GPU 显存降低。现在每张 GPU 不再为每一层加载完整权重，而只加载 `1/num_gpus` 的权重。
2. 每一层可用的显存变成 `每张 GPU 的显存 × GPU 数量`。
3. all-gather 和 all-reduce 的成本更低。

缺点：

1. 全局 batch size 变小。
2. 代码复杂度增加。

注意：它实际上只能应用于某些模块，但 LLM 中的大多数模块都适用。

## 确保一个节点上的所有 GPU 收到相同输入

因为计算被拆分到多张 GPU，同一个组中的所有 GPU 都必须收到相同输入。这也是全局 batch size 变小的原因。

首先创建 device mesh。device mesh 只是一种以 N 维方式查看设备的方法。如果有 8 张 GPU，可以组织成 `(2, 2, 2)`、`(2, 4)`、`(4, 2)`，甚至 `(1, 8)`。

它有用的原因是，我们可以像给张量维度命名一样给这些维度命名。类似于 batch 维度和 sequence 维度，这里的 device mesh 会有数据并行维度和张量并行维度。

```python
gpus_on_node = torch.cuda.device_count()
num_nodes = world_size // gpus_on_node
mesh = dist.device_mesh.init_device_mesh(
    "cuda",
    (num_nodes, gpus_on_node),
    mesh_dim_names=("dp", "tp"),
)
```

如果总共有 4 张 GPU，device mesh 是 `(2, 2)`，分配如下：

| | DP rank | TP rank |
| --- | --- | --- |
| GPU 0 | 0 | 0 |
| GPU 1 | 0 | 1 |
| GPU 2 | 1 | 0 |
| GPU 3 | 1 | 1 |

除非后续代码使用这些 device mesh，否则这个表本身没有任何作用。下面看看如何使用。很多 PyTorch 分布式 API 都接受可选参数 `mesh: Optional[DeviceMesh] = None`，只是前面还没有用过。

第一处是数据 sampler。它让张量并行组中的所有 GPU 得到相同输入：

```python
sampler=DistributedSampler(
    ...,
    num_replicas=mesh["dp"].size(),
    # NOTE: 一个节点上的每张 GPU 都有相同的 "dp" rank，
    # 因此它们都会收到相同输入！
    rank=mesh["dp"].get_local_rank(),
)
```

从上表中 GPU 0 的视角看，传给 DistributedSampler 的参数是：

| | num_replicas | rank |
| --- | --- | --- |
| GPU 0 | 2 | 0 |
| GPU 1 | 2 | 0 |
| GPU 2 | 2 | 1 |
| GPU 3 | 2 | 1 |

因为 DP 维度的大小是 2，而第一张表显示的正是传给 DistributedSampler 的 local rank。

## 并行化线性层和注意力模块

先看代码，后面的图会解释它如何工作。注意我们把 `mesh["tp"]` 传给 API，这意味着操作发生在张量并行组内。

```python
for layer in model.model.layers:
    tp.parallelize_module(
        layer,
        mesh["tp"],
        {
            "self_attn.q_proj": tp.ColwiseParallel(),
            "self_attn.k_proj": tp.ColwiseParallel(),
            "self_attn.v_proj": tp.ColwiseParallel(),
            "self_attn.o_proj": tp.RowwiseParallel(),

            "mlp.gate_proj": tp.ColwiseParallel(),
            "mlp.up_proj": tp.ColwiseParallel(),
            "mlp.down_proj": tp.RowwiseParallel(),
        },
    )
```

### 按列切分（colwise）

自注意力中的前三个线性层（q/k/v 投影）都是按列切分的线性层。由于权重以转置格式存储，这表示沿维度 0 切分层内的权重矩阵。注意力层的其余部分（包括自注意力）使用这个已切分的输出运行，因此注意力实际运行在更小的张量上。

<image src="https://storage.googleapis.com/lightning-avatars/litpages/01hyz8vg94nc6nk7t10rt8jpt1/a2fe38f3-4a73-4b0f-80da-c273d14cadd9.jpeg" width="640px" />

图片来源：[PyTorchLightning](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning#column-wise-parallel)

### 从按列切分进入按行切分

自注意力的最后一层是另一个线性层（`o_proj`）。这里使用按行并行。如下图所示，它可以在 TP 维度上「重新合并」：

<image src="https://storage.googleapis.com/lightning-avatars/litpages/01hyz8vg94nc6nk7t10rt8jpt1/a1bc6e8a-7146-44c6-b6cf-eec124cfbf74.jpeg" width="640px" />

图片来源：[PyTorchLightning](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning#combined-parallel-layers)

因此，自注意力的最终输出会再次成为副本。

### 并行化 Embedding 层

Embedding 权重沿维度 1 切分。也就是说，每张 GPU 保存每个 token 对应数据的不同切片：

| Embedding 权重形状 | 切分后的形状 |
| --- | --- |
| `(vocab_size, hidden_dim)` | `(vocab_size, hidden_dim / mesh["tp"].size())` |

普通 Embedding 层：

- 输入 token 的形状是 `shape=(batch, seq)`。
- 输出 embedding 的形状是 `shape=(batch, seq, hidden_dim)`。

切分 Embedding 权重张量后，这一层实际输出：

- 已切分的 embedding，形状是 `shape=(batch, seq, hidden_dim / mesh["tp"].size())`。

但这里有一个问题：`self_attn` 模块中的 **colwise** 部分会接收这一层的输出。ColwiseParallel 期望输入是**复制的**，而不是切分的。

因此需要对张量做一次 all-gather，让它在组内重新复制，也就是恢复为 `shape=(batch, seq, hidden_dim)`。幸好可以通过 `output_layouts` 参数指定这个额外变换：

```python
tp.parallelize_module(
    model,
    mesh["tp"],
    {"model.embed_tokens": tp.ColwiseParallel(output_layouts=Replicate())},
)
```

### 并行化模型最后的线性层

```python
tp.parallelize_module(
    model,
    mesh["tp"],
    {
        "lm_head": tp.ColwiseParallel(
            output_layouts=Replicate()
        ),
    },
)
```

这里必须包含 `Replicate()`，因为 loss 期望接收复制后的张量，而 colwise 默认会在最后一个维度上切分。

## 用 SequenceParallel 并行化 Norm 层

归一化层的工作方式略有不同。我们完全不切分层的权重，而是沿序列维度切分它的**输入**。

因此计算被拆开了，还需要做一些工作，把结果重新拼合给其他模块：

```diff
 for layer in model.model.layers:
     tp.parallelize_module(
         layer,
         mesh["tp"],
         {
+            "input_layernorm": tp.SequenceParallel(),
+            "self_attn": tp.PrepareModuleInput(
+                input_kwarg_layouts={"hidden_states": Shard(dim=1)},
+                desired_input_kwarg_layouts={"hidden_states": Replicate()},
+            ),
             "self_attn.q_proj": tp.ColwiseParallel(),
             "self_attn.k_proj": tp.ColwiseParallel(),
             "self_attn.v_proj": tp.ColwiseParallel(),
-            "self_attn.o_proj": tp.RowwiseParallel(),
+            "self_attn.o_proj": tp.RowwiseParallel(output_layouts=Shard(1)),
+            "post_attention_layernorm": tp.SequenceParallel(),
+            "mlp": tp.PrepareModuleInput(
+                input_layouts=Shard(dim=1),
+                desired_input_layouts=Replicate(),
+            ),
             "mlp.gate_proj": tp.ColwiseParallel(),
             "mlp.up_proj": tp.ColwiseParallel(),
-            "mlp.down_proj": tp.RowwiseParallel(),
+            "mlp.down_proj": tp.RowwiseParallel(output_layouts=Shard(1)),
         },
     )
```

`PrepareModuleInput` 对象改变张量的切分方式。例如，对 `self_attn` 来说，`hidden_states` 输入因为 `SequenceParallel` 而沿第 1 维切分，但所有 `ColwiseParallel` 都期望输入是复制的。

Embedding 层也要修改。现在它的输出进入 SequenceParallel 层，因此需要沿维度 1 切分：

```diff
 tp.parallelize_module(
     model,
     mesh["tp"],
-    {"model.embed_tokens": tp.ColwiseParallel(output_layouts=Replicate())},
+    {"model.embed_tokens": tp.ColwiseParallel(output_layouts=Shard(1))},
 )
```

由于 `transformers` 的特定代码，还需要一项额外修改。它根据 Embedding 层的输出计算序列长度；现在沿序列维度切分后，这个长度会是错的。显式传入 `position_ids` 可以修复，但**这非常依赖具体实现**：

```diff
 with timers["data"], torch.no_grad():
     batch = next(batches)
     batch = {k: v.to(device=device) for k, v in batch.items()}
+    batch["position_ids"] = torch.arange(
+        0, args.seq_length, device=device, dtype=torch.long
+    ).unsqueeze(0)
```

网络最终输出的差异如下：

```diff
 tp.parallelize_module(
     model,
     mesh["tp"],
     {
+        "model.norm": tp.SequenceParallel(),
         "lm_head": tp.ColwiseParallel(
+            input_layouts=Shard(1),
             output_layouts=Replicate(),
         ),
     },
 )
```

## 并行化 loss 计算

还有一个 API 可以沿**类别**维度并行化 loss 计算（撰写时只支持交叉熵）。首先要在 loss 计算周围使用这个上下文管理器：

```python
with tp.loss_parallel(), timers["forward"]:
    outputs = model(**batch)

with tp.loss_parallel(), timers["backward"]:
    outputs.loss.backward()
```

然后还要更新 `lm_head` 的输出，因为 `loss_parallel` 需要不同的切分格式和 DTensor：

```diff
 tp.parallelize_module(
     model,
     mesh["tp"],
     {
         "model.norm": tp.SequenceParallel(),
         "lm_head": tp.ColwiseParallel(
             input_layouts=Shard(1),
-            output_layouts=Replicate(),
+            output_layouts=Shard(-1),
+            use_local_output=False,
         ),
     },
 )
```

`use_local_output=False` 告诉 PyTorch，该操作返回 `DTensor`，而不是普通 `Tensor`。

## 用新的 world size 计算吞吐量

现在单张 GPU 不再是计算单位，因此吞吐量计算要改用 device mesh：

```diff
 if state["global_step"] % args.log_freq == 0:
-    tok_per_step = world_size * args.batch_size * args.seq_length
+    tok_per_step = mesh["dp"].size() * args.batch_size * args.seq_length
     ms_per_step = sum(t.avg_elapsed_ms() for t in timers.values())
```

## 结果

下面是在单台 8×H100 节点上启动 Llama 8B 训练的结果。

命令：

```bash
HF_HOME=/home/ubuntu/.cache/huggingface OMP_NUM_THREADS=26 torchrun --standalone --nproc-per-node gpu train_llm.py --experiment-name tp-llama-8b --dataset-name tatsu-lab/alpaca --model-name meta-llama/Llama-3.1-8B --log-freq 10 --batch-size 16 --seq-length 1024 --num-epochs 1
```

<img src="https://github.com/user-attachments/assets/4bafada2-beea-4e37-a341-62d7f4639014" width="480px" />

<img src="https://github.com/user-attachments/assets/c6e67666-db7c-4b67-bae4-98480382557f" width="480px" />

## 有用的参考资料

为了完整起见，这里列出 PyTorch 中实现这些功能的相关文档和指南：

- [TP API 文档](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#tensor-parallelism-torch-distributed-tensor-parallel)
- [2D 并行教程](https://pytorch.org/tutorials/intermediate/TP_tutorial.html#large-scale-transformer-model-training-with-tensor-parallel-tp)
- [Device Mesh 教程](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html)
- [PyTorch Lightning TP 教程](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning)

## PyTorch API 参考

这里简要说明即将使用的 API 如何工作。

- [tp.RowwiseParallel()](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.RowwiseParallel) 按行切分模块权重。
    - 输入默认沿最后一个维度切分。
    - 输出默认在所有 worker 上复制。
- [tp.ColwiseParallel()](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.ColwiseParallel) 按列切分模块权重。
    - 输入默认在所有 worker 上复制。
    - 输出默认沿最后一个维度切分。
- [tp.SequenceParallel()](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.SequenceParallel) 沿维度 1 切分输入和输出。模块权重**不会**被切分。
- [tp.PrepareModuleInput()](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#torch.distributed.tensor.parallel.PrepareModuleInput) 用于改变输入张量的切分配置。
- `torch.distributed._tensor.Shard(dim=X)` 表示张量应沿维度 X 切分。
- `torch.distributed._tensor.Replicate()` 表示张量应在所有 worker 上复制。

这些对象之间的交互实际上非常微妙和复杂，这也是本指南有用的原因。

还可以通过这些类的参数改变大多数默认行为。例如，可以让 RowwiseParallel 假设输入是复制的，而不是切分的。

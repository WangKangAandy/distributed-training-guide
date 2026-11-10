# 2D 并行（TP + DP）

同时使用 [FSDP](../04-fully-sharded-data-parallel) 和 [TP](../06-tensor-parallel)，从代码角度看其实很简单。起点是[第 6 章的 TP 脚本](../06-tensor-parallel/train_llm.py)。

两者一起使用究竟是什么意思？下面用 6 张 GPU、2 路 FSDP 和 3 路 TP 作为例子：

<img width="894" alt="image" src="https://github.com/user-attachments/assets/c384756a-66a9-4056-be1b-018fb046e275" />

一开始，每张 GPU 都持有完整模型。然后把模型切成 3 份，也就是 TP 维度。上图中的 3 个分片是红+橙、黄+绿、蓝+紫。注意 GPU 0 和 GPU 3 **持有完全相同的分片**。这是因为它们的张量并行 rank 相同，但数据并行 rank 不同。这意味着模型沿数据并行维度被**复制**了一份。

下一步应用 FSDP 时，这些重复的分片会被继续拆开。红+橙这个分片同时存在于 GPU 0 和 GPU 3，会被拆成两块：红色分片和橙色分片。

最终，模型被拆成 6 个不同分片，每张 GPU 持有一块。

回忆一下 FSDP：前向之前，它会对所有分片做 all-gather。GPU 0 和 GPU 3 执行前向时，会把红色分片和橙色分片收集到本地显存，组成红+橙分片，这样每张 GPU 在计算时都能使用完整分片。

## 在 TP 之后应用 FSDP

起点是已经支持 TP 的[第 6 章代码](../06-tensor-parallel/train_llm.py)，因此只需要把 FSDP 加进脚本。

这个 API 比 FSDP 1 简单得多。在 TP 代码**之后**，只需要加入：

```python
from torch.distributed.fsdp import fully_shard

for layer in model.model.layers:
    fully_shard(layer, mesh=mesh["dp"])
fully_shard(model, mesh=mesh["dp"])
```

注意这里传入 `mesh["dp"]`，表示操作发生在数据并行维度上。

## 控制 TP 大小

创建 mesh 时，根据命令行参数设置 TP 大小：

```python
assert world_size % args.tp == 0
mesh = dist.device_mesh.init_device_mesh(
    "cuda",
    (world_size // args.tp, args.tp),
    mesh_dim_names=("dp", "tp"),
)
```

并把它加入参数解析器：

```python
parser.add_argument("--tp", default=8, type=int)
```

## 不同配置的性能

下面是 4 种 TP 大小配置的训练结果：

- 1×8 是 8 路 TP，没有数据并行：`--batch-size 18 --tp 8`
- 2×4 是 4 路 TP，有 2 组 FSDP：`--batch-size 14 --tp 4`
- 4×2 是 2 路 TP，有 4 组 FSDP：`--batch-size 10 --tp 2`
- 8×1 是 FSDP：`--batch-size 7 --tp 1`

注意：这些运行使用相同的 `--lr`，但 batch size 不同，因此 loss 曲线会略有差异。

<img src="https://github.com/user-attachments/assets/8645b7d2-992f-4f49-9214-f6c5d4d42c37" width="480px" />

<img src="https://github.com/user-attachments/assets/1b9269ce-c1db-43e4-9fd7-0bbf11871b11" width="480px" />

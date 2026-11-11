# 优化数据加载

**注意：本章代码基于[第 2 章](../../02-distributed-data-parallel/)的代码。**

分布式训练要获得高吞吐量，一个重要部分是让所有进程的速度大致相同。如果某个进程快得多，它会花大量时间等待其他进程赶上。数据加载实际上是其中非常重要的一部分。

## 动机示例

撰写本指南时，我发现从单节点换成多节点后，**所有节点**的 GPU 利用率都下降了。单节点训练时 GPU 功耗约为 80%，进入多节点后，所有节点都降到了 60%。

原因是某个节点的数据加载持续更慢，导致**所有节点**都在等待它。

在本指南中，数据加载相对较快，因此只需要调整 worker 数量和预取因子就能解决。在更复杂的例子中，可能还需要其他优化或预处理。

## 并行加载数据

这种情况下，大多数变慢都来自数据规模：

1. 如果某些进程读取数据更慢，它们一开始就会落后。原因可能是磁盘读取被阻塞、打开文件描述符数量受限等。
2. 如果 batch 大小不同，模型前向和反向所需的时间也会不同。

大多数问题都可以通过在另一个进程中加载数据来处理（使用 `num_workers` 参数）：

```diff
 dataloader = DataLoader(
     train_data,
     batch_size=args.batch_size,
     collate_fn=default_data_collator,
+    num_workers=1,
+    prefetch_factor=2,
     # NOTE: 这个 sampler 会把数据集均匀分给各个 worker
     sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
 )
```

这样数据加载会在后台进行，并且**与 batch 处理并行**。

需要根据以下几项调整 `num_workers` 和预取因子：

1. batch size 有多大。
2. 数据集中单行数据的加载和预处理需要多久。
3. 处理一个 batch 有多快。

如果 `num_workers>0`，目标就是让完整加载一个 batch 的时间少于处理这个 batch 的时间。

## 测量等待时间

可以在代码中显式加入 `dist.barrier()`，并用计时器包住它们，从而测量这种现象：

```diff --git a/03-multi-node/train_llm.py b/06-data-loading/train_llm.py
index d5cb05c..26cadb8 100644
--- a/03-multi-node/train_llm.py
+++ b/06-data-loading/train_llm.py
@@ -146,7 +148,10 @@ def main():
         },
     )
 
-    timers = {k: LocalTimer(device) for k in ["data", "forward", "backward", "update"]}
+    timers = {
+        k: LocalTimer(device)
+        for k in ["data", "forward", "backward", "update", "waiting"]
+    }
 
     for state["epoch"] in range(state["epoch"], args.num_epochs):
         LOGGER.info(
@@ -168,13 +173,22 @@ def main():
                 # NOTE: 用于恢复训练
                 continue
 
+            with timers["waiting"]:
+                dist.barrier()
+
             with timers["forward"]:
                 outputs = model(**batch)
 
+            with timers["waiting"]:
+                dist.barrier()
+
             with timers["backward"]:
                 optimizer.zero_grad(set_to_none=True)
                 outputs.loss.backward()
 
+            with timers["waiting"]:
+                dist.barrier()
+
             with timers["update"]:
                 optimizer.step()
                 lr_scheduler.step()
```

## 更快的存储

一种非常常见的配置是把所有数据放在网络存储上。这对代码来说很方便，但对读取数据并不是最高效的。

类似于缓存快于内存、内存快于磁盘，节点本地存储也比网络存储快得多：

1. 缓存（最快）
2. 内存
3. 机器本地磁盘
4. 网络磁盘（最慢）

把所有数据复制到每个节点，可以提高数据加载速度，但代价是占用更多存储。

> 译注：diff 中的旧目录名来自原文历史版本，主要展示应修改的位置；请对照当前第 2 章脚本应用改动。

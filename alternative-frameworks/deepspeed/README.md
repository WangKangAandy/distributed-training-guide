# DeepSpeed ZeRO

安装 DeepSpeed：`pip install deepspeed`

<img width="583" alt="image" src="https://github.com/user-attachments/assets/c2966a46-1807-4a56-92d4-977798087dd6">

它实际上是一组逐步分片更多内存的模式：

> ZeRO Stage 1：优化器状态（例如 Adam 优化器的 32 位权重，以及一阶矩和二阶矩估计）在进程之间划分，因此每个进程只更新自己的那一部分。

> ZeRO Stage 2：用于更新模型权重的已归约 32 位梯度也被划分，每个进程只保留与自己那部分优化器状态对应的梯度。

> ZeRO Stage 3：16 位模型参数在进程之间划分。ZeRO-3 会在前向和反向期间自动收集并重新划分它们。

参考资料：

- [DeepSpeed 文档](https://deepspeed.readthedocs.io/en/latest/zero3.html)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)

## 把 DeepSpeed 集成进训练代码

### 参数解析

```diff
@@ -305,11 +302,10 @@ def _get_parser() -> argparse.ArgumentParser:
     parser.add_argument("--log-freq", default=10, type=int)
     parser.add_argument("--ckpt-freq", default=500, type=int)
+    parser.add_argument("--local_rank", type=int, default=None)
+    deepspeed.add_config_arguments(parser)
     return parser
```

### 初始化

这里有两个主要区别：

1. 调用 `deepspeed.init_distributed`，而不是使用 PyTorch 的 `init_process_group`。
2. 构造模型后调用 `deepspeed.initialize`，**代替**用 DDP 包装模型。

**注意**：`deepspeed.initialize` 会根据传入的配置构造优化器和学习率调度器。

```diff
@@ -14,6 +14,7 @@ from torch.nn.parallel import DistributedDataParallel
 from torch import distributed as dist
 from torch.distributed.elastic.multiprocessing.errors import record
 
+import deepspeed
 import numpy
 import wandb
 import tqdm
@@ -42,10 +43,15 @@ def main(): 
-    dist.init_process_group()
+    deepspeed.init_distributed()
 
     rank = dist.get_rank()
-    local_rank = rank % torch.cuda.device_count()
+    local_rank = args.local_rank or (rank % torch.cuda.device_count())
     world_size = dist.get_world_size()
 
     LOGGER.info(f"local_rank={local_rank} rank={rank} world size={world_size}")
 
@@ -73,10 +73,6 @@ def main():
     if len(tokenizer) > embedding_size:
         model.resize_token_embeddings(len(tokenizer))
 
-    model = DistributedDataParallel(
-        model, device_ids=[local_rank], output_device=local_rank
-    )
-
@@ -89,9 +95,11 @@ def main():
     )
     LOGGER.info(f"{len(dataloader)} batches per epoch")
 
-    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
-    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
-        optimizer, T_max=1000, eta_min=args.lr * 1e-2
+    model_engine: deepspeed.DeepSpeedEngine
+    model_engine, _, _, lr_scheduler = deepspeed.initialize(
+        args,
+        model=model,
+        model_parameters=(p for p in model.parameters() if p.requires_grad),
     )
```

### 训练循环

这里只是把 PyTorch 调用替换成 DeepSpeed 调用。注意，我们不再直接访问优化器和学习率调度器，因为 DeepSpeed 会处理它们。

```diff
             with timers["forward"]:
-                outputs = model(**batch)
+                outputs = model_engine(**batch)
 
             with timers["backward"]:
-                optimizer.zero_grad(set_to_none=True)
-                outputs.loss.backward()
+                model_engine.backward(outputs.loss)
 
             with timers["update"]:
-                optimizer.step()
-                lr_scheduler.step()
+                model_engine.step()
 
             state["global_step"] += 1
             state["epoch_step"] += 1
```

### 检查点

加载变成：

```diff
     resumed = False
-    if (exp_dir / "state.json").exists():
-        model.load_state_dict(_load_to_device(exp_dir / "model.pt"))
-        optimizer.load_state_dict(_load_to_device(exp_dir / "optimizer.pt"))
-        lr_scheduler.load_state_dict(_load_to_device(exp_dir / "lr_scheduler.pt"))
-        with open(exp_dir / "state.json") as fp:
-            state = json.load(fp)
-        resumed = True
+    if (exp_dir / "pytorch_model.bin").exists():
+        load_path, state = model_engine.load_checkpoint(exp_dir)
+        resumed = load_path is not None
```

保存变成（**注意**：由于分片，必须在所有 rank 上保存，而不能只在 rank 0 上保存）：

```diff
             if state["global_step"] % args.ckpt_freq == 0:
-                if rank == 0:
-                    torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
-                    torch.save(model.state_dict(), exp_dir / "model.pt")
-                    torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
-                    with open(exp_dir / "state.json", "w") as fp:
-                        json.dump(state, fp)
+                model_engine.save_checkpoint(exp_dir, client_state=state)
                 dist.barrier()
```

## 配置

```json
{
    "train_micro_batch_size_per_gpu": 64,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 3e-5
        }
    },
    "scheduler": {
        "type": "WarmupCosineLR",
        "params": {
            "total_num_steps": 1000,
            "warmup_num_steps": 0,
            "cos_min_ratio": 1e-2
        }
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": false,
        "offload_optimizer": false
    }
}
```

## 命令

```bash
cd distributed-training-guide/05-sharding-deepspeed
export TORCHELASTIC_ERROR_FILE=../error.json
export OMP_NUM_THREADS=1
export HF_HOME=../.cache
deepspeed \
    --enable_each_rank_log ../logs \
    train_llm.py \
    --experiment-name deepspeed-multi-node-$(date +%Y-%m-%dT%H-%M-%S) \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2 \
    --deepspeed_config ds_config.json
```

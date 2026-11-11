# 梯度累积

梯度累积是一种增大模型更新有效 batch size 的方法。

通常在模型很大、前向和反向必须使用较小 batch size 时使用它。

如果单张 GPU 的 batch size 是 4，梯度累积步数是 2，那么有效 batch size 是 8。

但是，以标准方式应用梯度累积会在分布式训练中造成变慢，因为梯度需要同步。

> 译注：以下保留原文的简化示意，假定 `i_step` 从 **1** 开始，且每组恰有 `grad_accum` 个 microbatch。若步号从 0 开始，应使用 `(i_step + 1) % grad_accum`；循环开始前要清空梯度。若要匹配大 batch 的平均梯度，等大小、等权重 microbatch 应先将 loss 除以 `grad_accum`。最后不足一组时需显式更新，并按实际样本数或有效 token 数正确加权。原示例未展示这些细节，不能直接视为完整训练循环。

## 标准实现

```python
outputs = model(**batch)
outputs.loss.backward()
if i_step % grad_accum == 0:
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
```

## DistributedDataParallel 实现

在默认 DDP 中，每次反向传播都会触发梯度同步。累积期间可以跳过中间 microbatch 的梯度同步，在本轮累积的最后一次反向传播时同步，再更新参数。

> 译注：原文写成“前向期间同步梯度”，这里改为反向传播的梯度归约。`no_sync()` 必须同时包住前向和反向；前向仍可能涉及 buffer 广播等其他通信。

可以使用 [torch.nn.parallel.DistributedDataParallel.no_sync](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync)：

```python
from contextlib import nullcontext
maybe_sync_grads = model.no_sync if i_step % grad_accum != 0 else nullcontext
with maybe_sync_grads():
    outputs = model(**batch)
    outputs.loss.backward()
if i_step % grad_accum == 0:
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
```

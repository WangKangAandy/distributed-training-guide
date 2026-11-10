# 梯度累积

梯度累积是一种增大模型更新有效 batch size 的方法。

通常在模型很大、前向和反向必须使用较小 batch size 时使用它。

如果单张 GPU 的 batch size 是 4，梯度累积步数是 2，那么有效 batch size 是 8。

但是，以标准方式应用梯度累积会在分布式训练中造成变慢，因为梯度需要同步。

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

在分布式环境中，前向期间的多个位置都会同步梯度。实际上，必须把这次同步推迟到执行完整模型更新时。

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

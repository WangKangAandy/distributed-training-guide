# 弹性训练

弹性训练是指启动器可以在训练过程中的不同时刻重启一部分（或全部）worker。

与你可能的预期相反，通常只要 1 个 worker 遇到错误，**所有 worker 都会重启**（见 https://pytorch.org/docs/stable/elastic/run.html#membership-changes）。

`torchrun` 通过[弹性启动](https://pytorch.org/docs/stable/elastic/run.html#elastic-min-1-max-4-tolerates-up-to-3-membership-changes-or-failures)支持这一点：

```bash
torchrun
    --nnodes=1:4
    --max-restarts=3
    ...
```

这表示 `torchrun` 最多重启所有 worker 3 次；如果某些节点离线，它最少可以使用 1 个节点。

注意：

- worker 重启后，`rank`、`local_rank` 和 `world_size` 都不稳定。
- 有些节点问题无法只靠重启解决，例如代码本身有 bug。

## 代码改动

现有代码做弹性训练不需要修改。更有帮助的是运行一个 worker 随机崩溃的示例，感受它如何工作。

```bash
cd distributed-training-guide/96-elastic-training
torchrun \
    --nnodes 1 \
    --nproc_per_node 8 \
    --max-restarts 3 \
    --redirects 3 \
    --log-dir ../logs \
    toy.py
```

这个示例脚本会从各个 rank 随机抛出错误。**试用这条命令不需要 GPU。**

运行后查看日志目录。每次尝试都会有一个 worker 子目录包含 `error.json`。也可以查看每个 worker 的标准输出和标准错误。

# wandb 配置

训练运行期间配置 wandb 有很多方式。哪种最合适取决于集群有多大，以及你想跟踪什么。

## rank 0

这是标准方法。你只能看到 rank 0 进程所在节点的系统信息，也只会记录 rank 0 的数据。信息量最小，但仍然可以跟踪实验进度。

```python
if rank == 0:
    wandb.init(
        project="distributed-training-guide",
        dir=exp_dir,
        id=args.experiment_name,
        name=args.experiment_name,
        resume="must" if resumed else None,
        save_code=True,
        config=...,
    )
```

## local_rank 0（每个节点）

使用这种方法可以看到所有节点的系统信息，记录数量随节点数线性增长。它使用 [wandb 分组运行](https://docs.wandb.ai/guides/runs/grouping/)。

```python
if local_rank == 0:
    wandb.init(
        project="distributed-training-guide",
        dir=exp_dir / f"rank-{rank}",
        group=args.experiment_name,
        name=f"rank-{rank}",
        id=f"{args.experiment_name}-{rank}",
        resume="must" if resumed else None,
        save_code=True,
        config=...,
    )
```

如果希望名称显示为节点编号，可以设置：

```python
name=f"node-{rank // world_size}"
```

## 每个 rank

[分组文档](https://docs.wandb.ai/guides/runs/grouping)

这种配置非常适合尽可能多地跟踪集群信息。缺点是，集群非常大时可能触发 wandb 的速率限制，而且 wandb 图表会变得无法使用。

```python
wandb.init(
    project="distributed-training-guide",
    dir=exp_dir / f"rank-{rank}",
    group=args.experiment_name,
    name=f"rank-{rank}",
    id=f"{args.experiment_name}-{rank}",
    resume="must" if resumed else None,
    save_code=True,
    config=...,
)
```

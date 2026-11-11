# 诊断错误

挂起和死锁可能由很多原因引起，甚至包括你自己的代码。下面这些诊断工具可以帮助判断发生了什么。

## 诊断挂起时要观察的系统指标

`GPU Power Usage` 是最主要的指标。如果训练进程挂起，所有 worker 的功耗都会降到大约 10%：

```bash
> nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader
69.75 W, 700.00 W
75.10 W, 700.00 W
70.82 W, 700.00 W
69.29 W, 700.00 W
69.19 W, 700.00 W
68.72 W, 700.00 W
70.80 W, 700.00 W
70.87 W, 700.00 W
```

使用提供的 [top-cluster.py](../top-cluster.py) 脚本会输出类似下面的内容：

从仓库根目录运行；`hosts` 文件每行填写一个主机名。可选参数 `--poll-freq` 控制轮询间隔，单位为毫秒，默认 `1000`。输出列 `util`、`power`、`memory`、`nprocs` 分别表示 GPU 利用率、功耗比例、显存占用比例和计算进程数。

```bash
> python top-cluster.py <hosts file>
===2024-10-02 19:55:02.553039
    name	      util	     power	    memory	    nprocs
 cluster	    100.0%	     99.1%	     96.9%	        64
node-001	    100.0%	     99.7%	     96.1%	         8
node-002	    100.0%	     97.8%	     96.9%	         8
node-003	    100.0%	     99.2%	     97.2%	         8
node-004	    100.0%	     99.1%	     97.4%	         8
node-005	    100.0%	     98.1%	     97.1%	         8
node-006	    100.0%	     99.0%	     97.7%	         8
node-007	    100.0%	     99.8%	     96.9%	         8
node-008	    100.0%	    100.0%	     96.2%	         8
===
```

## 获取堆栈跟踪转储

使用 [py-spy](https://github.com/benfred/py-spy) 可以转储正在运行的 Python 程序中所有 Python 线程的堆栈跟踪。从每个 worker 获取转储的方法是：

```
sudo env "PATH=$PATH" py-spy dump --locals --pid <PID of the torchrun process>
```

## 基准测试与性能分析

可以使用 `py-spy top --pid <>`，获得类似 `top` / `htop` 的视图，查看 Python 进程正在调用哪些函数。

## 记录错误

Python 有一个很好的内置库 [faulthandler](https://docs.python.org/3/library/faulthandler.html)，可以获取 Python 程序任意线程中发生的错误。当 DataLoader 的 `num_workers > 0` 时，它尤其有用。

实际上，PyTorch 已经内置了使用它的方法。只需设置环境变量 `TORCHELASTIC_ERROR_FILE=../error.json`，并给 `main` 函数加上 `@record` 注解。

```python
from torch.distributed.elastic.multiprocessing.errors import record

# NOTE: 把错误记录到 $TORCHELASTIC_ERROR_FILE
@record
def main():
    ...
```

好在本指南中的所有代码都已经这样做了，你也应该这样做。**一定要设置 `$TORCHELASTIC_ERROR_FILE`。**

## 系统问题检查清单

1. 每台机器的系统日期和时间相同（不一致可能导致 NCCL 超时）。
2. NVLink 拓扑有效：`nvidia-smi topo -m`。
3. NVLink 状态：`nvidia-smi topo -p2p n`（也可以把 `n` 换成 `w` 或 `r`）。
4. 打开文件描述符上限：`ulimit -aH`，然后查看包含 `open files` 的那一行。
5. `dist.init_process_group(timeout=...)` 中的 `timeout` 足够大。

> 译注：使用 `py-spy` 时应传入目标训练 worker 的 Python PID；只查看 `torchrun` 启动器的 PID 可能只能看到等待子进程的调用栈。`@record` 主要记录装饰入口中传播出的异常，不能保证捕获所有线程或 DataLoader 子进程中的任意错误；单卡示例也未使用该装饰器。功耗下降是排障线索，不是判断挂起的充分条件。

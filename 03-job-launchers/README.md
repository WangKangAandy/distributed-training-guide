# 任务启动器

**注意：本章代码与[第 2 章](../02-distributed-data-parallel/)完全相同，因此命令使用第 2 章的训练脚本。** 如果某种启动器必须修改代码才能工作，文中会单独标出。

> 译注：本章 MPI 和 DeepSpeed 小节要求的代码改动，应分别应用到第 2 章脚本或它的副本，再运行对应命令；以下路径修正并不表示未经修改的脚本支持所有启动器。

手动 SSH 到每台节点再启动训练很麻烦，所以有多种方法可以从一台节点启动分布式训练任务。

快速跳转：
- [每节点一条 Bash 命令](#bash-commands-xargssshtmux)
- [slurm](#slurm)
- [mpi](#mpirun)
- [deepspeed](#deepspeed)

## <a id="bash-commands-xargssshtmux"></a>Bash 命令（xargs/ssh/tmux）

我们主要需要在其他机器上创建进程，因此可以把几个 Bash 工具组合起来。这是最轻量的方法之一，也方便按需要修改命令。理解这些 Bash 命令如何配合需要一点时间，但它们也能用到其他问题上。

把主机名或 IP 列表放进名为 `hosts` 的文件。每一行代表一台要启动 `torchrun` 的节点。

```
<hostname 1>
<hostname 2>
...
<hostname n>
```

然后用 SSH 在每台主机上启动 `torchrun`。这条命令和前面的 Bash 命令很像，只是不再直接运行 Python 脚本，而是使用 `torchrun`（`python -m torch.distributed.run`）。

```bash
cd distributed-training-guide/03-job-launchers
JOB_NAME=multi-node-tmux
xargs -a hosts -I {} \
    ssh {} tmux new-session -d -s $JOB_NAME -c $(pwd) \
    -e TORCHELASTIC_ERROR_FILE=../error.json \
    -e OMP_NUM_THREADS=1 \
    -e HF_HOME=../.cache \
    $(which python) -m torch.distributed.run \
    --rdzv-id $JOB_NAME \
    --rdzv-backend c10d \
    --rdzv-endpoint $(head -n 1 hosts):5001 \
    --nnodes $(grep -c '^' hosts) \
    --nproc-per-node gpu \
    --redirects 3 \
    --log-dir ../logs \
    ../02-distributed-data-parallel/train_llm.py \
    --experiment-name $JOB_NAME \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2
```

查看输出：

```bash
find ../logs/ -name \*stderr.log | xargs tail -f
```

停止任务：

```bash
xargs -a hosts -I{} ssh {} tmux kill-session -t $JOB_NAME
```

这些部分的作用：

1. `xargs -a hosts -I {}` 读取 `hosts` 的每一行，并用该行替换后面命令中的 `{}`。
2. `ssh {} tmux new-session -d -s $JOB_NAME -c $(pwd)` 在 `hosts` 中的每台主机上创建一个 tmux 会话。
    1. `-d` 表示分离会话，因此创建时不会阻塞。
    2. `-s $JOB_NAME` 用任务名命名会话，便于统一停止。
    3. `-c $(pwd)` 让每个进程使用执行这条命令时所在的工作目录。
    4. `-e <环境变量名>=<值>` 为 tmux 启动的进程设置环境变量。

后面直接接上原来的 Python 命令。注意用 `$(which python)` 取得当前解释器可执行文件的绝对路径。

## slurm

slurm 是集群上非常常用的任务调度软件。

用提供的 `job.sbatch` 脚本提交训练任务：

```bash
cd distributed-training-guide/03-job-launchers
sbatch --nodes 2 --gpus 16 --cpus-per-task 8 job.sbatch
```

slurm 默认每个节点分配一个 task。这正好适合我们，因为每个节点只需调用一次 `torchrun`。

上面的命令总共向 2 个节点申请 16 张 GPU。

### slurm 文件

它和目前一直使用的 `torchrun` 命令基本相同，只是各项设置由 slurm 控制。

由于指定了 `--ntasks-per-node=1`，下面的命令会在每个节点上运行。

```bash
# SBATCH --ntasks-per-node=1

source $(pwd)/../venv/bin/activate

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$(expr 5000 + $(echo -n ${SLURM_JOBID} | tail -c 4))
export TORCHELASTIC_ERROR_FILE=./error-${SLURM_JOBID}-${SLURM_NODEID}.json
export OMP_NUM_THREADS=1
export HF_HOME=../.cache

printenv

srun torchrun \
    --rdzv-id "slurm-${SLURM_JOBID}" \
    --rdzv-backend c10d \
    --rdzv-endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes ${SLURM_NNODES} \
    --nproc-per-node ${SLURM_GPUS_ON_NODE} \
    --redirects 3 \
    --log-dir ${SLURM_SUBMIT_DIR}/logs \
    ../02-distributed-data-parallel/train_llm.py \
    --experiment-name gpt2-alpaca-slurm-$(date +%Y-%m-%dT%H-%M-%S) \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2
```

## mpirun

MPI 主要有 OpenMPI 和 MPICH 两种实现。两者都可以，本文使用 OpenMPI。**需要先安装 OpenMPI。**

### 代码改动

初始化进程组时使用 MPI 环境变量：

```diff
-    dist.init_process_group()
+    dist.init_process_group(
+        rank=int(os.environ["OMPI_COMM_WORLD_RANK"]),
+        world_size=int(os.environ["OMPI_COMM_WORLD_SIZE"]),
+    )
```

### 命令

```bash
cd distributed-training-guide/03-job-launchers
mpirun \
    -H <host 1>:<gpus on 1>,...,<host n>:<gpus on n> \
    -x MASTER_ADDR=<host 1> \
    -x MASTER_PORT=5001 \
    -x TORCHELASTIC_ERROR_FILE=../error.json \
    -x OMP_NUM_THREADS=1 \
    -x HF_HOME=../.cache \
    -bind-to none \
    -map-by slot \
    -wdir $(pwd) \
    -output-filename ../logs/mpi-multi-node \
    $(which python) ../02-distributed-data-parallel/train_llm.py \
    --experiment-name mpi-multi-node \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2
```

参数：

- `-H` 指定要启动的主机，以及每台主机上的进程数。
- `-x` 给所有启动的进程设置环境变量。
- `-wdir` 设置这些进程的工作目录。
- `-bind-to none` 让 Open MPI 不要把训练进程绑定到单个 CPU 核心，否则会损害性能。
- `-map-by slot` 允许混合不同的 NUMA 配置，因为默认行为是绑定到 socket。

说明：

- 必须指定 `MASTER_ADDR` 和 `MASTER_PORT`，PyTorch 才知道进程之间如何通信。
- 代码中必须根据 `$OMPI_COMM_WORLD_RANK` 和 `$OMPI_COMM_WORLD_SIZE` 传入 rank 和 world size。
- 用 `$(which python)` 获取 Python 解释器的绝对路径。如果从管理节点而不是 worker 节点启动，就需要修改这一项。

## deepspeed

deepspeed 是一个带有许多优化的分布式训练库。后面章节会更详细地介绍其中一些优化；这里只使用它自带的启动器。

**注意：使用 deepspeed 启动器时，不必把 deepspeed 集成进训练代码。**

安装：`pip install deepspeed`

### 代码改动

在命令行解析中加入 `--local_rank`：

```diff
     parser.add_argument("--log-freq", default=10, type=int)
     parser.add_argument("--ckpt-freq", default=500, type=int)
+    parser.add_argument("--local_rank", type=int, default=None)
     return parser
```

初始化 `local_rank` 时使用它：

```diff
-    local_rank = rank % torch.cuda.device_count()
+    local_rank = args.local_rank or (rank % torch.cuda.device_count())
```

### 命令

```bash
cd distributed-training-guide/03-job-launchers
export HF_HOME=../.cache
export TORCHELASTIC_ERROR_FILE=../error.json
export OMP_NUM_THREADS=1
deepspeed \
    --include <ip of node 1>@<ip of node 2> \
    --enable_each_rank_log ../logs \
    ../02-distributed-data-parallel/train_llm.py \
    --experiment-name deepspeed-multi-node \
    --dataset-name tatsu-lab/alpaca \
    --model-name openai-community/gpt2
```

# 分布式训练指南

本分支提供完整简体中文阅读版，覆盖全部教程、相关主题及示例脚本的教学注释。英文原版见 [LambdaLabsML/distributed-training-guide](https://github.com/LambdaLabsML/distributed-training-guide)，翻译基于上游 `aed7489`。正文中的“译注”用于澄清原文或说明示例边界。

API、配置键、命令参数、论文名称和日志输出保留原文，便于检索和排障。[LICENSE](./LICENSE) 保留 MIT 许可证原文。教程命令以 Linux/Bash 和相应的 GPU/集群环境为基础。

🔥 新消息 🔥

- [PyTorch Conference 2025 幻灯片](https://docs.google.com/presentation/d/126BQcCJti_z3NQ7cGbm2iMBDVG7-0ekitxU3pvJ_WQw/edit?usp=sharing)
- [NeurIPS 2024 幻灯片](https://docs.google.com/presentation/d/1ANMmkOGaruYKTvhnsAbZgI9GrdMliNvibWGuNYw6HX8/edit?usp=sharing)

<img src="https://lambdalabs.com/hubfs/distriubuted-training-guide.png" width="400px" />

想知道如何在大型集群上训练一个大神经网络？看这篇就够了。

这是一份关于分布式训练最佳实践、错误诊断以及充分利用现有资源的完整指南。内容按顺序分成若干章，每一章都有一份 `README.md`，多数还有一份 `train_llm.py` 脚本。README 会同时讲分布式训练的高层概念，以及该章引入的代码改动。

指南全部使用非常精简的标准 PyTorch 编写，模型用 `transformers`，数据用 `datasets`。分布式相关代码不再依赖其他库，全部写在 PyTorch 里。

1. [第 1 章](./01-single-gpu/)：标准的因果语言模型训练脚本，在**单张 GPU** 上运行。
2. [第 2 章](./02-distributed-data-parallel/)：把训练脚本升级为支持**多张 GPU，并使用 DDP**。
3. [第 3 章](./03-job-launchers/)：介绍如何在多节点集群上**启动训练任务**。
4. [第 4 章](./04-fully-sharded-data-parallel/)：把训练脚本从 DDP 升级为 **FSDP**，以更合理地使用显存。
5. [第 5 章](./05-training-llama-405b/)：把训练脚本升级为可以**训练 Llama-405B**。
6. [第 6 章](./06-tensor-parallel/)：把单卡训练脚本升级为支持**张量并行**。
7. [第 7 章](./07-2d-parallel/)：把张量并行脚本升级为 **2D 并行（FSDP + TP）**。
8. [其他框架](./alternative-frameworks/)：介绍底层仍基于 PyTorch 的不同框架。
9. [错误诊断](./diagnosing-errors/)：在集群中**快速诊断错误**的最佳实践和方法。
10. [相关主题](./related-topics/)：做分布式训练时应该了解的主题。

本指南回答的问题：

- 如何把单卡训练 / 微调脚本改成在多张 GPU 或多台机器上运行？
- 训练过程中出现挂起或报错时，如何诊断？
- 模型或优化器放不进单张 GPU，如何在集群上训练 / 微调？
- 如何在集群上调度并启动训练？
- 增加 worker 数量时，如何缩放超参数？

其中也包含标准输出、标准错误以及 wandb 日志的最佳实践。在集群上诊断和调试训练时，日志至关重要。

每份训练脚本的目标都是训练一个因果语言模型（例如 GPT / Llama）。

## 环境准备

### 克隆仓库

```bash
git clone -b zh-cn https://github.com/WangKangAandy/distributed-training-guide.git
```

### 虚拟环境

```bash
cd distributed-training-guide
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip
pip install -U setuptools wheel
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### wandb

本教程使用 `wandb` 作为实验跟踪工具。

```bash
wandb login
```

<p align="center">
🦄 Lambda 的其他机器学习项目：<a href="https://news.lambda.ai/news/today">ML Times</a>、<a href="https://lambdalabsml.github.io/Open-Sora/introduction/">文生视频</a>、<a href="https://lambdalabs.com/gpu-benchmarks">GPU 基准测试</a>。
</p>

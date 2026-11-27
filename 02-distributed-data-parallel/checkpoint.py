"""实验目录、checkpoint 保存与断点续训。与第 1 章 checkpoint.py 同构；谁写盘、何时 barrier 由 train_llm.py 决定。"""

import json
import logging
from pathlib import Path

import torch

LOGGER = logging.getLogger(__name__)


def resolve_experiment_dir(args) -> tuple[bool, Path]:
    """
    「实验」= 给这次 run 起名，用来存 checkpoint / 断点续训。
    不传 --experiment-name：只训练、不落盘；传了：目录为 save_dir/实验名/
    """
    is_experiment = False  # 默认：纯训练，不写盘
    exp_dir = Path(args.save_dir)  # 先落到 save_dir 根
    if args.experiment_name is not None:
        is_experiment = True  # 开启存盘 / 续训
        exp_dir = exp_dir / args.experiment_name  # 每个实验名一个子目录
    return is_experiment, exp_dir


def resume_if_available(
    exp_dir,
    device,
    model,
    optimizer,
    lr_scheduler,
    state,
    *,
    load_optimizer=True,
):
    """
    同名实验目录里已有 state.json → 视为续训：权重 + 进度一并读回。
    返回 (state, resumed)。
    load_optimizer=False：跳过优化器（例如本章 ZeRO 未保存优化器状态时）。
    此时只恢复权重 / scheduler / 进度，Adam 动量等会重新初始化，不是完整 resume。
    """
    resumed = False
    # 没有 state.json 就当作新实验，原样返回调用方传入的 state
    if not (exp_dir / "state.json").exists():
        return state, resumed

    # map_location：把保存时的张量加载到当前 device
    # weights_only=True：只反序列化张量，降低恶意 pickle 风险
    def _load_to_device(p):
        return torch.load(p, map_location=device, weights_only=True)

    # 模型权重（DDP 包装下 state_dict 键通常带 module. 前缀，保存/加载需一致）
    model.load_state_dict(_load_to_device(exp_dir / "model.pt"))
    # 可选：恢复优化器；本章 ZeRO 路径传 load_optimizer=False
    if load_optimizer:
        optimizer.load_state_dict(_load_to_device(exp_dir / "optimizer.pt"))
    # 学习率调度器（当前步数、余弦相位等）
    lr_scheduler.load_state_dict(_load_to_device(exp_dir / "lr_scheduler.pt"))
    # 训练进度：epoch / global_step / epoch_step
    with open(exp_dir / "state.json") as fp:
        state = json.load(fp)
    # 旧 checkpoint 可能带 running_loss；累加器不落盘，丢掉以免语义混乱
    state.pop("running_loss", None)
    resumed = True
    return state, resumed


def ensure_experiment_dir(exp_dir: Path) -> None:
    LOGGER.info("Creating experiment root directory")
    # 确保存盘目录存在（续训时目录已在，exist_ok 避免报错）
    # 多进程时由 train_llm 保证只让 rank 0 调用，并在前后 barrier
    exp_dir.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    exp_dir,
    model,
    optimizer,
    lr_scheduler,
    state,
    *,
    include_optimizer=True,
) -> None:
    """
    写入模型 /（可选）优化器 / 调度器 / 进度。
    有 state.json 启动时才会走恢复逻辑。
    include_optimizer=False：跳过 optimizer.pt（本章 ZeRO 默认如此）。
    """
    LOGGER.info("Saving checkpoint.")
    # 可选写优化器；关闭时续训无法恢复 Adam 动量等
    if include_optimizer:
        torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
    # 模型权重（若 model 是 DDP，保存的是包装后的 state_dict）
    torch.save(model.state_dict(), exp_dir / "model.pt")
    # 学习率调度器状态
    torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
    # 进度 JSON：供下次启动知道从哪一步接着跑
    with open(exp_dir / "state.json", "w") as fp:
        json.dump(state, fp)

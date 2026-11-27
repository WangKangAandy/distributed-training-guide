"""实验目录、checkpoint 保存与断点续训。与第 1 章 checkpoint.py 同构；rank0 / barrier 由 train_llm.py 调用。"""

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
    is_experiment = False
    exp_dir = Path(args.save_dir)
    if args.experiment_name is not None:
        is_experiment = True
        exp_dir = exp_dir / args.experiment_name
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
    load_optimizer=False：跳过优化器（例如 ZeRO 未保存优化器状态时）。
    """
    resumed = False
    if not (exp_dir / "state.json").exists():
        return state, resumed

    # map_location：把保存时的张量加载到当前 device
    # weights_only=True：只反序列化张量，降低恶意 pickle 风险
    def _load_to_device(p):
        return torch.load(p, map_location=device, weights_only=True)

    model.load_state_dict(_load_to_device(exp_dir / "model.pt"))
    if load_optimizer:
        optimizer.load_state_dict(_load_to_device(exp_dir / "optimizer.pt"))
    lr_scheduler.load_state_dict(_load_to_device(exp_dir / "lr_scheduler.pt"))
    with open(exp_dir / "state.json") as fp:
        state = json.load(fp)
    state.pop("running_loss", None)  # 旧 checkpoint 可能带该字段；累加器不落盘
    resumed = True
    return state, resumed


def ensure_experiment_dir(exp_dir: Path) -> None:
    LOGGER.info("Creating experiment root directory")
    # 确保存盘目录存在（续训时目录已在，exist_ok 避免报错）
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
    """写入模型 / 优化器 / 调度器 / 进度；有 state.json 启动时才会走恢复逻辑。"""
    LOGGER.info("Saving checkpoint.")
    if include_optimizer:
        torch.save(optimizer.state_dict(), exp_dir / "optimizer.pt")
    torch.save(model.state_dict(), exp_dir / "model.pt")
    torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
    with open(exp_dir / "state.json", "w") as fp:
        json.dump(state, fp)

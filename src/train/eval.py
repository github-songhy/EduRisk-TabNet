"""评估流程实现。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from src.models.model_full import EduRiskTabNet模型
from src.train.train import 评估模型
from src.utils.logger import 创建日志器


def 评估检查点(
    模型: EduRiskTabNet模型,
    加载器: torch.utils.data.DataLoader,
    checkpoint路径: Path,
    结果目录: Path,
    设备: torch.device,
) -> Dict[str, float]:
    """加载检查点并评估。"""

    if torch is None:
        raise ModuleNotFoundError("未检测到torch，请先安装依赖")
    日志器 = 创建日志器("评估", 结果目录)
    if not checkpoint路径.exists():
        raise FileNotFoundError("检查点文件不存在")

    模型.load_state_dict(torch.load(checkpoint路径, map_location=设备))
    模型.to(设备)

    指标 = 评估模型(模型, 加载器, 设备)
    日志器.info(f"评估完成：{指标}")
    return 指标

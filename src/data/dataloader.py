"""数据加载器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    DataLoader = object

from src.data.dataset import 数据集配置, 表格数据集


@dataclass
class 加载器配置:
    """加载器配置。"""

    batch_size: int
    shuffle: bool
    num_workers: int


def 创建数据加载器(
    数据配置: 数据集配置,
    加载配置: 加载器配置,
    训练模式: bool = True,
) -> DataLoader:
    """创建数据加载器。"""

    if torch is None:
        raise ModuleNotFoundError("未检测到torch，请先安装依赖")
    数据集 = 表格数据集(数据配置, 训练模式=训练模式)
    return DataLoader(
        数据集,
        batch_size=加载配置.batch_size,
        shuffle=加载配置.shuffle if 训练模式 else False,
        num_workers=加载配置.num_workers,
    )

"""随机种子工具。"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@dataclass
class 种子摘要:
    """随机种子配置摘要。"""

    种子: int
    cudnn确定性: bool
    cudnn基准模式: Optional[bool]
    python哈希种子: Optional[int]


def 设置随机种子(种子: int, cudnn确定性: bool = True) -> 种子摘要:
    """设置全局随机种子，并返回当前配置摘要。"""

    random.seed(种子)

    if np is not None:
        np.random.seed(种子)
    else:
        print("未检测到numpy，已跳过numpy随机种子设置")

    if torch is not None:
        torch.manual_seed(种子)
        torch.cuda.manual_seed_all(种子)
    else:
        print("未检测到torch，已跳过torch随机种子设置")

    os.environ["PYTHONHASHSEED"] = str(种子)

    cudnn基准模式 = None
    if torch is not None:
        torch.backends.cudnn.deterministic = cudnn确定性
        torch.backends.cudnn.benchmark = not cudnn确定性
        cudnn基准模式 = torch.backends.cudnn.benchmark

    return 种子摘要(
        种子=种子,
        cudnn确定性=cudnn确定性,
        cudnn基准模式=cudnn基准模式,
        python哈希种子=int(os.environ.get("PYTHONHASHSEED", str(种子))),
    )

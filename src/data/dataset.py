"""数据集定义。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    Dataset = object

from src.data.preprocess import 应用预处理, 拟合预处理, 读取数据


@dataclass
class 数据集配置:
    """数据集配置。"""

    数据路径: str
    数据格式: str
    标签列: str
    类别列: Optional[List[str]]
    填充策略: str
    分组列: Optional[str]
    标准化: bool
    类别编码: str
    统计路径: str


class 表格数据集(Dataset):
    """表格数据集，输出(x_hat, m, y)。"""

    def __init__(self, 配置: 数据集配置, 训练模式: bool = True) -> None:
        self.配置 = 配置
        self.训练模式 = 训练模式
        if torch is None:
            raise ModuleNotFoundError("未检测到torch，请先安装依赖")
        if pd is None:
            raise ModuleNotFoundError("未检测到pandas，请先安装依赖")
        self.数据表 = 读取数据(Path(配置.数据路径), 配置.数据格式)

        if 配置.标签列 not in self.数据表.columns:
            raise ValueError(f"标签列不存在：{配置.标签列}")
        if np is None:
            raise ModuleNotFoundError("未检测到numpy，请先安装依赖")

        if 训练模式:
            _, 缺失指示, _, 特征矩阵 = 拟合预处理(
                self.数据表,
                标签列=配置.标签列,
                类别列=配置.类别列,
                填充策略=配置.填充策略,
                分组列=配置.分组列,
                标准化=配置.标准化,
                类别编码=配置.类别编码,
                统计保存路径=Path(配置.统计路径),
            )
        else:
            _, 缺失指示, _, 特征矩阵 = 应用预处理(
                self.数据表,
                标签列=配置.标签列,
                统计路径=Path(配置.统计路径),
                填充策略=配置.填充策略,
                分组列=配置.分组列,
                标准化=配置.标准化,
                类别编码=配置.类别编码,
            )

        self.缺失指示 = 缺失指示
        self.标签 = self.数据表[配置.标签列].to_numpy(dtype=np.int64)

        特征列 = [列 for 列 in self.数据表.columns if 列 != 配置.标签列]
        self.原始缺失位置 = self.数据表[特征列].isna().to_numpy()

        self.特征矩阵 = 特征矩阵

    def __len__(self) -> int:
        return len(self.数据表)

    def __getitem__(self, 索引: int):
        x_hat = torch.tensor(self.特征矩阵[索引], dtype=torch.float32)
        m = torch.tensor(self.缺失指示[索引], dtype=torch.float32)
        y = torch.tensor(self.标签[索引], dtype=torch.long)
        return x_hat, m, y

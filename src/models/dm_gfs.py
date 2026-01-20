"""双Mask组级特征选择模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from src.models.sparsemax import sparsemax


@dataclass
class DMGFS配置:
    """DM-GFS配置。"""

    输入维度: int
    组数: int
    上下文维度: int


class 组注意力变换器(nn.Module):
    """组级注意力变换器。"""

    def __init__(self, 组数: int, 上下文维度: int) -> None:
        super().__init__()
        self.组线性 = nn.Linear(组数, 组数)
        self.上下文线性 = nn.Linear(上下文维度, 组数)

    def forward(self, 组特征: torch.Tensor, 上下文: torch.Tensor) -> torch.Tensor:
        # 组特征: (B, G)，上下文: (B, C)
        return self.组线性(组特征) + self.上下文线性(上下文)


class 特征注意力变换器(nn.Module):
    """特征级注意力变换器。"""

    def __init__(self, 输入维度: int, 上下文维度: int) -> None:
        super().__init__()
        self.特征线性 = nn.Linear(输入维度, 输入维度)
        self.上下文线性 = nn.Linear(上下文维度, 输入维度)

    def forward(self, 特征: torch.Tensor, 上下文: torch.Tensor) -> torch.Tensor:
        # 特征: (B, D)，上下文: (B, C)
        return self.特征线性(特征) + self.上下文线性(上下文)


class 双Mask组特征选择(nn.Module):
    """组级mask与特征级mask的联合生成。"""

    def __init__(self, 配置: DMGFS配置) -> None:
        super().__init__()
        self.配置 = 配置
        self.组注意力 = 组注意力变换器(配置.组数, 配置.上下文维度)
        self.特征注意力 = 特征注意力变换器(配置.输入维度, 配置.上下文维度)

    def forward(
        self,
        x_tilde: torch.Tensor,
        上下文: torch.Tensor,
        组矩阵: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入:
        - x_tilde: (B, D)
        - 上下文: (B, C)
        - 组矩阵: (D, G)
        输出:
        - 特征mask: (B, D)
        - 组mask: (B, G)
        """

        if 组矩阵.dim() != 2:
            raise ValueError("组矩阵必须为二维矩阵")
        if 组矩阵.size(0) != x_tilde.size(1):
            raise ValueError("组矩阵行数必须等于特征维度")

        # f_g: (B, G)
        组特征 = torch.matmul(x_tilde, 组矩阵)
        组logits = self.组注意力(组特征, 上下文)
        组mask = sparsemax(组logits, dim=1)

        # m_tilde: (B, D)
        组投影 = torch.matmul(组mask, 组矩阵.t())

        特征logits = self.特征注意力(x_tilde, 上下文)
        特征mask = sparsemax(特征logits * 组投影, dim=1)

        return 特征mask, 组mask

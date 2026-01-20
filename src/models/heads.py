"""分类头定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class Head配置:
    """分类头配置。"""

    输入维度: int
    类别数: int
    类型: str = "softmax"  # softmax 或 ordinal


class SoftmaxHead(nn.Module):
    """softmax分类头。"""

    def __init__(self, 输入维度: int, 类别数: int) -> None:
        super().__init__()
        self.线性 = nn.Linear(输入维度, 类别数)

    def forward(self, 输入: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.线性(输入)
        prob = torch.softmax(logits, dim=1)
        return logits, prob


class OrdinalHead(nn.Module):
    """有序分类头。"""

    def __init__(self, 输入维度: int, 类别数: int) -> None:
        super().__init__()
        self.类别数 = 类别数
        self.线性 = nn.Linear(输入维度, 类别数 - 1)

    def forward(self, 输入: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.线性(输入)
        q = torch.sigmoid(logits)
        prob = self._还原概率(q)
        return logits, prob

    def _还原概率(self, q: torch.Tensor) -> torch.Tensor:
        # q形状: (B, K-1)
        批大小 = q.size(0)
        K = self.类别数
        prob = torch.zeros(批大小, K, device=q.device)
        prob[:, 0] = 1 - q[:, 0]
        for k in range(1, K - 1):
            prob[:, k] = q[:, k - 1] - q[:, k]
        prob[:, K - 1] = q[:, K - 2]
        return prob.clamp(min=1e-8, max=1.0)


def 创建分类头(配置: Head配置) -> nn.Module:
    """根据配置创建分类头。"""

    if 配置.类型 == "softmax":
        return SoftmaxHead(配置.输入维度, 配置.类别数)
    if 配置.类型 == "ordinal":
        return OrdinalHead(配置.输入维度, 配置.类别数)
    raise ValueError("不支持的分类头类型")

"""样本级自适应步长控制模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class SASC配置:
    """SASC配置。"""

    决策维度: int
    决策步数: int
    启用: bool = True
    early_exit: bool = False
    阈值: float = 0.95
    代价阈值: float = 1e-4


class 样本级步长控制(nn.Module):
    """样本级自适应步长控制。"""

    def __init__(self, 配置: SASC配置) -> None:
        super().__init__()
        self.配置 = 配置
        self.停止线性 = nn.Linear(配置.决策维度, 1)

    def forward(
        self,
        决策序列: torch.Tensor,
        期望代价序列: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        输入:
        - 决策序列: (B, T, D)
        - 期望代价序列: (B, T)，可选
        输出:
        - 聚合决策: (B, D)
        - alpha序列: (B, T)
        - 期望步数正则: 标量
        - 实际步数: (B,)
        """

        批大小, 步数, _ = 决策序列.shape
        if 步数 != self.配置.决策步数:
            raise ValueError("决策序列步数与配置不一致")

        r = torch.ones(批大小, device=决策序列.device)
        alpha序列 = []
        r序列 = []

        for t in range(步数):
            p = torch.sigmoid(self.停止线性(决策序列[:, t, :]).squeeze(-1))
            p = p.clamp(min=1e-6, max=1 - 1e-6)
            alpha = r * p
            alpha序列.append(alpha)
            r序列.append(r)
            r = r * (1 - p)

        alpha张量 = torch.stack(alpha序列, dim=1)
        聚合决策 = torch.sum(alpha张量.unsqueeze(-1) * 决策序列, dim=1)

        r张量 = torch.stack(r序列, dim=1)
        期望步数正则 = r张量.sum(dim=1).mean()

        if not self.配置.early_exit:
            实际步数 = torch.full((批大小,), 步数, device=决策序列.device, dtype=torch.long)
            return 聚合决策, alpha张量, 期望步数正则, 实际步数

        累计 = torch.cumsum(alpha张量, dim=1)
        触发阈值 = 累计 >= self.配置.阈值

        if 期望代价序列 is not None:
            代价下降 = torch.zeros_like(期望代价序列)
            代价下降[:, 1:] = 期望代价序列[:, :-1] - 期望代价序列[:, 1:]
            触发阈值 = 触发阈值 | (代价下降 < self.配置.代价阈值)

        实际步数 = torch.argmax(触发阈值.to(torch.int64), dim=1) + 1
        实际步数 = torch.clamp(实际步数, min=1, max=步数)

        return 聚合决策, alpha张量, 期望步数正则, 实际步数

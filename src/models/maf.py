"""缺失感知融合模块。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MAF配置:
    """MAF配置。"""

    输入维度: int
    缺失嵌入维度: int = 1
    上下文维度: int = 8
    使用_film: bool = True
    使用缺失嵌入: bool = True


class 缺失感知融合(nn.Module):
    """缺失感知融合模块，输出x_tilde。"""

    def __init__(self, 配置: MAF配置) -> None:
        super().__init__()
        self.配置 = 配置

        self.缺失嵌入 = nn.Embedding(2, 配置.缺失嵌入维度)
        self.缺失上下文编码 = nn.Sequential(
            nn.Linear(配置.输入维度, 配置.上下文维度),
            nn.ReLU(),
            nn.Linear(配置.上下文维度, 配置.上下文维度),
        )
        self.film生成器 = nn.Sequential(
            nn.Linear(配置.上下文维度, 配置.输入维度 * 2),
        )
        self.缺失嵌入映射 = nn.Linear(配置.缺失嵌入维度, 1)

    def forward(self, x_hat: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        输入:
        - x_hat: (B, D)
        - m: (B, D) 二值缺失指示
        输出:
        - x_tilde: (B, D)
        """

        if x_hat.shape != m.shape:
            raise ValueError("x_hat与m的形状必须一致")

        输出 = x_hat

        if self.配置.使用_film:
            # c_m: (B, C)
            c_m = self.缺失上下文编码(m)
            # film参数: (B, 2D) -> gamma,beta: (B, D)
            film参数 = self.film生成器(c_m)
            gamma, beta = torch.chunk(film参数, 2, dim=1)
            输出 = gamma * 输出 + beta

        if self.配置.使用缺失嵌入:
            # e_m: (B, D, E)
            e_m = self.缺失嵌入(m.long())
            # g(e_m): (B, D)
            g_em = self.缺失嵌入映射(e_m).squeeze(-1)
            输出 = 输出 + g_em

        return 输出

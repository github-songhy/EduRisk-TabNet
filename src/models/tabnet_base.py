"""TabNet基座实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn

from src.models.sparsemax import sparsemax


@dataclass
class TabNet配置:
    """TabNet配置。"""

    输入维度: int
    决策维度: int
    上下文维度: int
    决策步数: int
    类别数: int
    稀疏正则权重: float = 1.0
    gamma: float = 1.5


class 注意力变换器(nn.Module):
    """注意力变换器，用于生成mask。"""

    def __init__(self, 输入维度: int, 输出维度: int) -> None:
        super().__init__()
        self.线性层 = nn.Linear(输入维度, 输出维度)

    def forward(self, 输入: torch.Tensor) -> torch.Tensor:
        return self.线性层(输入)


class 特征变换器(nn.Module):
    """特征变换器，输出决策向量与上下文。"""

    def __init__(self, 输入维度: int, 决策维度: int, 上下文维度: int) -> None:
        super().__init__()
        self.决策维度 = 决策维度
        self.上下文维度 = 上下文维度
        self.层1 = nn.Linear(输入维度, 决策维度 + 上下文维度)
        self.激活 = nn.ReLU()
        self.层2 = nn.Linear(决策维度 + 上下文维度, 决策维度 + 上下文维度)

    def forward(self, 输入: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        隐变量 = self.激活(self.层1(输入))
        输出 = self.激活(self.层2(隐变量))
        决策, 上下文 = torch.split(输出, [self.决策维度, self.上下文维度], dim=1)
        return 决策, 上下文


class TabNet基座(nn.Module):
    """简化版TabNet基座。"""

    def __init__(self, 配置: TabNet配置) -> None:
        super().__init__()
        self.配置 = 配置
        self.注意力变换器 = 注意力变换器(配置.上下文维度, 配置.输入维度)
        self.特征变换器 = 特征变换器(配置.输入维度, 配置.决策维度, 配置.上下文维度)
        self.分类头 = nn.Linear(配置.决策维度, 配置.类别数)

    def forward(self, 输入: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], Dict[str, torch.Tensor]]:
        批大小 = 输入.size(0)
        prior = torch.ones_like(输入)
        上下文 = torch.zeros(批大小, self.配置.上下文维度, device=输入.device)

        mask列表: List[torch.Tensor] = []
        决策列表: List[torch.Tensor] = []
        稀疏项列表: List[torch.Tensor] = []

        累积决策 = torch.zeros(批大小, self.配置.决策维度, device=输入.device)

        for _ in range(self.配置.决策步数):
            注意力 = self.注意力变换器(上下文)
            mask = sparsemax(prior * 注意力, dim=1)
            mask列表.append(mask)

            特征输入 = mask * 输入
            决策, 上下文 = self.特征变换器(特征输入)
            决策列表.append(决策)

            累积决策 = 累积决策 + torch.relu(决策)

            prior = prior * (self.配置.gamma - mask)

            稀疏项 = (mask * torch.log(mask.clamp(min=1e-6))).sum(dim=1)
            稀疏项列表.append(稀疏项)

        logits = self.分类头(累积决策)
        稀疏正则 = torch.stack(稀疏项列表).mean() * self.配置.稀疏正则权重

        中间量 = {
            "prior": prior,
            "sparse_loss": 稀疏正则,
        }

        return logits, mask列表, 决策列表, 中间量

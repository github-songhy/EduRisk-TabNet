"""DACOS代价敏感损失与推理工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class DACOS配置:
    """DACOS配置。"""

    类别数: int
    eta: float = 1.0
    alpha: float = 2.0
    beta: float = 1.0
    tau: float = 1.0
    lambda_risk: float = 1.0
    warmup_steps: int = 0


def 构建代价矩阵(配置: DACOS配置, 设备: torch.device) -> torch.Tensor:
    """构建代价矩阵C，形状为(K, K)。"""

    K = 配置.类别数
    y = torch.arange(K, device=设备).view(-1, 1)
    a = torch.arange(K, device=设备).view(1, -1)
    距离 = (y - a).float()
    代价 = 配置.eta * 距离.pow(2)
    代价 = 代价 + 配置.alpha * torch.clamp(距离, min=0)
    代价 = 代价 + 配置.beta * torch.clamp(-距离, min=0)
    return 代价


def 期望代价(prob: torch.Tensor, 代价矩阵: torch.Tensor) -> torch.Tensor:
    """计算期望代价R，返回形状(B, K)。"""

    return prob @ 代价矩阵


def 风险对齐损失(
    prob: torch.Tensor,
    标签: torch.Tensor,
    代价矩阵: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """计算风险对齐损失。"""

    风险 = 期望代价(prob, 代价矩阵)
    温度 = max(tau, 1e-6)
    pi = torch.softmax(-风险 / 温度, dim=1)
    return torch.nn.functional.nll_loss(torch.log(pi.clamp(min=1e-12)), 标签)


def 风险权重(当前步: int, 配置: DACOS配置) -> float:
    """计算风险损失权重，支持warmup。"""

    if 配置.warmup_steps <= 0:
        return 配置.lambda_risk
    比例 = min(1.0, 当前步 / 配置.warmup_steps)
    return 配置.lambda_risk * 比例


def Bayes最小代价预测(prob: torch.Tensor, 代价矩阵: torch.Tensor) -> torch.Tensor:
    """Bayes最小代价决策，返回预测类别索引。"""

    风险 = 期望代价(prob, 代价矩阵)
    return torch.argmin(风险, dim=1)


def ExpectedCost指标(prob: torch.Tensor, 标签: torch.Tensor, 代价矩阵: torch.Tensor) -> torch.Tensor:
    """计算期望代价指标。"""

    风险 = 期望代价(prob, 代价矩阵)
    标签风险 = 风险.gather(1, 标签.view(-1, 1)).squeeze(1)
    return 标签风险.mean()

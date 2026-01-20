"""统一模型封装。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn

from src.models.maf import MAF配置, 缺失感知融合
from src.models.tabnet_base import TabNet基座, TabNet配置
from src.models.sasc import SASC配置, 样本级步长控制


@dataclass
class 模型配置:
    """统一模型配置。"""

    maf: MAF配置
    tabnet: TabNet配置
    sasc: SASC配置


class EduRiskTabNet模型(nn.Module):
    """MAF + TabNet基座的统一模型。"""

    def __init__(self, 配置: 模型配置) -> None:
        super().__init__()
        self.配置 = 配置
        self.maf = 缺失感知融合(配置.maf)
        self.tabnet = TabNet基座(配置.tabnet)
        self.sasc = 样本级步长控制(配置.sasc)

    def forward(
        self,
        x_hat: torch.Tensor,
        m: torch.Tensor,
        组矩阵: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], Dict[str, torch.Tensor]]:
        x_tilde = self.maf(x_hat, m)
        logits, mask列表, 决策列表, 中间量 = self.tabnet(x_tilde, 组矩阵=组矩阵)
        if not self.配置.sasc.启用:
            return logits, mask列表, 决策列表, 中间量

        决策序列 = torch.stack(决策列表, dim=1)
        聚合决策, alpha, 期望步数, 实际步数 = self.sasc(决策序列)
        logits = self.tabnet.分类头(聚合决策)
        中间量.update(
            {
                "alpha": alpha,
                "step_loss": 期望步数,
                "used_steps": 实际步数,
            }
        )
        return logits, mask列表, 决策列表, 中间量

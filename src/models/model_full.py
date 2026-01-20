"""统一模型封装。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn

from src.models.maf import MAF配置, 缺失感知融合
from src.models.tabnet_base import TabNet基座, TabNet配置


@dataclass
class 模型配置:
    """统一模型配置。"""

    maf: MAF配置
    tabnet: TabNet配置


class EduRiskTabNet模型(nn.Module):
    """MAF + TabNet基座的统一模型。"""

    def __init__(self, 配置: 模型配置) -> None:
        super().__init__()
        self.配置 = 配置
        self.maf = 缺失感知融合(配置.maf)
        self.tabnet = TabNet基座(配置.tabnet)

    def forward(
        self,
        x_hat: torch.Tensor,
        m: torch.Tensor,
        组矩阵: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], Dict[str, torch.Tensor]]:
        x_tilde = self.maf(x_hat, m)
        return self.tabnet(x_tilde, 组矩阵=组矩阵)

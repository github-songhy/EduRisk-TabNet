"""统一模型封装。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None

from src.losses.dacos import Bayes最小代价预测, DACOS配置, ExpectedCost指标, 构建代价矩阵
from src.models.heads import Head配置, 创建分类头
from src.models.maf import MAF配置, 缺失感知融合
from src.models.sasc import SASC配置, 样本级步长控制
from src.models.tabnet_base import TabNet基座, TabNet配置


@dataclass
class 模型配置:
    """统一模型配置。"""

    maf: MAF配置
    tabnet: TabNet配置
    sasc: SASC配置
    head: Head配置
    dacos: DACOS配置


class EduRiskTabNet模型(nn.Module):
    """MAF + TabNet基座的统一模型。"""

    def __init__(self, 配置: 模型配置) -> None:
        if nn is None or torch is None:
            raise ModuleNotFoundError("未检测到torch，请先安装依赖")
        super().__init__()
        self.配置 = 配置
        self.maf = 缺失感知融合(配置.maf)
        self.tabnet = TabNet基座(配置.tabnet)
        self.sasc = 样本级步长控制(配置.sasc)
        self.head = 创建分类头(配置.head)
        self.register_buffer("代价矩阵", 构建代价矩阵(配置.dacos, torch.device("cpu")))

    def forward(
        self,
        x_hat: torch.Tensor,
        m: torch.Tensor,
        组矩阵: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], Dict[str, torch.Tensor]]:
        x_tilde = self.maf(x_hat, m)
        _, mask列表, 决策列表, 中间量 = self.tabnet(x_tilde, 组矩阵=组矩阵)
        决策序列 = torch.stack(决策列表, dim=1)
        基座聚合 = torch.sum(torch.relu(决策序列), dim=1)

        if not self.配置.sasc.启用:
            logits, prob = self.head(基座聚合)
            中间量.update({"prob": prob})
            return prob, mask列表, 决策列表, 中间量

        聚合决策, alpha, 期望步数, 实际步数 = self.sasc(决策序列)
        logits, prob = self.head(聚合决策)
        中间量.update(
            {
                "alpha": alpha,
                "step_loss": 期望步数,
                "used_steps": 实际步数,
                "prob": prob,
            }
        )
        return prob, mask列表, 决策列表, 中间量

    def predict(self, prob: torch.Tensor) -> torch.Tensor:
        """使用Bayes最小代价进行预测。"""

        代价矩阵 = self.代价矩阵.to(prob.device)
        return Bayes最小代价预测(prob, 代价矩阵)

    def 计算期望代价(self, prob: torch.Tensor, 标签: torch.Tensor) -> torch.Tensor:
        """计算期望代价指标。"""

        代价矩阵 = self.代价矩阵.to(prob.device)
        return ExpectedCost指标(prob, 标签, 代价矩阵)

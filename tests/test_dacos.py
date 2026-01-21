"""DACOS模块测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

项目根目录 = Path(__file__).resolve().parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

from src.losses.dacos import DACOS配置, ExpectedCost指标, Bayes最小代价预测, 构建代价矩阵, 风险对齐损失
from src.models.heads import Head配置, 创建分类头


def test_dacos代价矩阵与风险() -> None:
    配置 = DACOS配置(类别数=3, eta=1.0, alpha=2.0, beta=1.0, tau=1.0)
    代价矩阵 = 构建代价矩阵(配置, torch.device("cpu"))
    assert 代价矩阵.shape == (3, 3)

    prob = torch.tensor([[0.2, 0.5, 0.3]])
    标签 = torch.tensor([1])
    损失 = 风险对齐损失(prob, 标签, 代价矩阵, 配置.tau)
    assert 损失.ndim == 0

    期望代价 = ExpectedCost指标(prob, 标签, 代价矩阵)
    assert 期望代价.ndim == 0

    预测 = Bayes最小代价预测(prob, 代价矩阵)
    assert 预测.shape == (1,)


def test_dacos输出头() -> None:
    head = 创建分类头(Head配置(输入维度=4, 类别数=3, 类型="ordinal"))
    输入 = torch.randn(2, 4)
    logits, prob = head(输入)
    assert logits.shape == (2, 2)
    assert prob.shape == (2, 3)
    行和 = prob.sum(dim=1)
    assert torch.allclose(行和, torch.ones_like(行和), atol=1e-4)

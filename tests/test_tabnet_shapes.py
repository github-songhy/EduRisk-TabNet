"""TabNet基座形状测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

项目根目录 = Path(__file__).resolve().parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

from src.models.tabnet_base import TabNet基座, TabNet配置


def test_tabnet输出形状() -> None:
    配置 = TabNet配置(
        输入维度=8,
        决策维度=4,
        上下文维度=4,
        决策步数=3,
        类别数=5,
        稀疏正则权重=1.0,
        gamma=1.5,
    )
    模型 = TabNet基座(配置)
    输入 = torch.randn(2, 8)
    logits, mask列表, 决策列表, 中间量 = 模型(输入)

    assert logits.shape == (2, 5)
    assert len(mask列表) == 配置.决策步数
    assert len(决策列表) == 配置.决策步数
    for mask in mask列表:
        assert mask.shape == (2, 8)
        行和 = mask.sum(dim=1)
        assert torch.allclose(行和, torch.ones_like(行和), atol=1e-4)
    for 决策 in 决策列表:
        assert 决策.shape == (2, 4)

    assert "sparse_loss" in 中间量
    assert 中间量["sparse_loss"].ndim == 0

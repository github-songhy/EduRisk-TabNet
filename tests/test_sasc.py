"""SASC模块测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

项目根目录 = Path(__file__).resolve().parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

from src.models.sasc import SASC配置, 样本级步长控制


def test_sasc软聚合() -> None:
    配置 = SASC配置(决策维度=4, 决策步数=3, early_exit=False)
    模块 = 样本级步长控制(配置)
    决策序列 = torch.randn(2, 3, 4)
    聚合决策, alpha, 期望步数, 实际步数 = 模块(决策序列)

    assert 聚合决策.shape == (2, 4)
    assert alpha.shape == (2, 3)
    assert 期望步数.ndim == 0
    assert 实际步数.shape == (2,)
    assert torch.all(实际步数 == 3)


def test_sasc早退() -> None:
    配置 = SASC配置(决策维度=4, 决策步数=3, early_exit=True, 阈值=0.5)
    模块 = 样本级步长控制(配置)
    决策序列 = torch.randn(2, 3, 4)
    聚合决策, alpha, 期望步数, 实际步数 = 模块(决策序列)

    assert 聚合决策.shape == (2, 4)
    assert alpha.shape == (2, 3)
    assert 实际步数.min() >= 1
    assert 实际步数.max() <= 3

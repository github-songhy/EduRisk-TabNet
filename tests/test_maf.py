"""MAF模块测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

项目根目录 = Path(__file__).resolve().parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

from src.models.maf import MAF配置, 缺失感知融合


def test_maf输出形状与稳定性() -> None:
    配置 = MAF配置(
        输入维度=6,
        缺失嵌入维度=1,
        上下文维度=4,
        使用_film=True,
        使用缺失嵌入=True,
    )
    模块 = 缺失感知融合(配置)
    x_hat = torch.randn(3, 6)
    m = torch.randint(0, 2, (3, 6)).float()
    输出 = 模块(x_hat, m)

    assert 输出.shape == x_hat.shape
    assert not torch.isnan(输出).any()


def test_maf仅film() -> None:
    配置 = MAF配置(
        输入维度=4,
        缺失嵌入维度=1,
        上下文维度=4,
        使用_film=True,
        使用缺失嵌入=False,
    )
    模块 = 缺失感知融合(配置)
    x_hat = torch.zeros(2, 4)
    m = torch.ones(2, 4)
    输出 = 模块(x_hat, m)
    assert 输出.shape == (2, 4)


def test_maf仅嵌入() -> None:
    配置 = MAF配置(
        输入维度=4,
        缺失嵌入维度=1,
        上下文维度=4,
        使用_film=False,
        使用缺失嵌入=True,
    )
    模块 = 缺失感知融合(配置)
    x_hat = torch.zeros(2, 4)
    m = torch.zeros(2, 4)
    输出 = 模块(x_hat, m)
    assert 输出.shape == (2, 4)

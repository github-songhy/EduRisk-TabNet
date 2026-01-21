"""DM-GFS模块测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

项目根目录 = Path(__file__).resolve().parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

from src.models.dm_gfs import DMGFS配置, 双Mask组特征选择


def test_dm_gfs输出形状() -> None:
    配置 = DMGFS配置(输入维度=4, 组数=2, 上下文维度=3)
    模块 = 双Mask组特征选择(配置)
    x_tilde = torch.randn(2, 4)
    上下文 = torch.randn(2, 3)
    组矩阵 = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    特征mask, 组mask = 模块(x_tilde, 上下文, 组矩阵)
    assert 特征mask.shape == (2, 4)
    assert 组mask.shape == (2, 2)
    assert torch.allclose(特征mask.sum(dim=1), torch.ones(2), atol=1e-4)
    assert torch.allclose(组mask.sum(dim=1), torch.ones(2), atol=1e-4)

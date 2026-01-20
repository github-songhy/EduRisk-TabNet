"""稀疏化函数测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

项目根目录 = Path(__file__).resolve().parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

from src.models.sparsemax import sparsemax


def test_sparsemax求和为1() -> None:
    输入 = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    输出 = sparsemax(输入, dim=1)
    行和 = 输出.sum(dim=1)
    assert torch.allclose(行和, torch.ones_like(行和), atol=1e-6)


def test_sparsemax产生稀疏() -> None:
    输入 = torch.tensor([[10.0, 0.0, -1.0]])
    输出 = sparsemax(输入, dim=1)
    assert torch.isclose(输出[0, 0], torch.tensor(1.0))
    assert torch.isclose(输出[0, 1], torch.tensor(0.0))
    assert torch.isclose(输出[0, 2], torch.tensor(0.0))


def test_sparsemax极端输入稳定() -> None:
    输入 = torch.tensor([[1000.0, -1000.0, 0.0]])
    输出 = sparsemax(输入, dim=1)
    assert not torch.isnan(输出).any()
    assert torch.allclose(输出.sum(dim=1), torch.ones(1), atol=1e-6)

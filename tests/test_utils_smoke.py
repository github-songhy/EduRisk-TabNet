"""工具模块最小测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

项目根目录 = Path(__file__).resolve().parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

from src.utils.config import 加载并合并配置, 读取配置
from src.utils.logger import 创建日志器
from src.utils.seed import 设置随机种子


def test_设置随机种子不报错() -> None:
    摘要 = 设置随机种子(1234)
    assert 摘要.种子 == 1234


def test_logger能写文件(tmp_path: Path) -> None:
    日志器 = 创建日志器("测试日志", tmp_path)
    日志器.info("测试日志输出")

    日志文件 = tmp_path / "运行日志.log"
    assert 日志文件.exists()
    内容 = 日志文件.read_text(encoding="utf-8")
    assert "测试日志输出" in 内容


def test_config能读写(tmp_path: Path) -> None:
    配置路径 = tmp_path / "配置.json"
    配置路径.write_text(json.dumps({"a": 1, "b": {"c": 2}}), encoding="utf-8")

    默认配置 = {"b": {"d": 3}}
    合并配置 = 加载并合并配置(配置路径, 默认配置=默认配置, 结果目录=tmp_path)

    assert 合并配置["a"] == 1
    assert 合并配置["b"]["c"] == 2
    assert 合并配置["b"]["d"] == 3

    保存路径 = tmp_path / "最终配置.yaml"
    assert 保存路径.exists()
    读取内容 = 读取配置(保存路径)
    assert 读取内容["a"] == 1

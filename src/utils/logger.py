"""日志工具。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def 创建日志器(
    名称: str,
    结果目录: str | Path,
    级别: int = logging.INFO,
    文件名: str = "运行日志.log",
) -> logging.Logger:
    """创建同时输出到控制台与文件的日志器。"""

    结果路径 = Path(结果目录)
    结果路径.mkdir(parents=True, exist_ok=True)

    日志器 = logging.getLogger(名称)
    日志器.setLevel(级别)
    日志器.propagate = False

    if 日志器.handlers:
        return 日志器

    格式 = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    控制台处理器 = logging.StreamHandler()
    控制台处理器.setLevel(级别)
    控制台处理器.setFormatter(格式)

    文件处理器 = logging.FileHandler(结果路径 / 文件名, encoding="utf-8")
    文件处理器.setLevel(级别)
    文件处理器.setFormatter(格式)

    日志器.addHandler(控制台处理器)
    日志器.addHandler(文件处理器)

    日志器.info("日志器初始化完成")

    return 日志器

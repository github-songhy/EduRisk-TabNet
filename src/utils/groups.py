"""分组矩阵构建工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


def 读取分组配置(路径: Path) -> Dict[str, List[str]]:
    """读取分组配置文件。"""

    if not 路径.exists():
        raise FileNotFoundError(f"分组配置文件不存在：{路径}")

    if 路径.suffix.lower() in {".yaml", ".yml"}:
        内容 = yaml.safe_load(路径.read_text(encoding="utf-8")) or {}
        return 内容.get("groups", {})

    if 路径.suffix.lower() == ".json":
        内容 = json.loads(路径.read_text(encoding="utf-8"))
        return 内容.get("groups", {})

    raise ValueError("分组配置文件格式不支持，仅支持yaml/json")


def 从schema读取分组(schema路径: Path) -> Optional[Dict[str, List[str]]]:
    """从schema.json读取分组信息。"""

    if not schema路径.exists():
        return None

    内容 = json.loads(schema路径.read_text(encoding="utf-8"))
    分组 = 内容.get("特征分组")
    if not isinstance(分组, dict):
        return None
    return {k: list(v) for k, v in 分组.items()}


def 默认分组策略(特征列: List[str]) -> Dict[str, List[str]]:
    """根据列名前缀生成默认分组。"""

    分组: Dict[str, List[str]] = {}
    for 列名 in 特征列:
        if "_" in 列名:
            前缀 = 列名.split("_")[0]
        else:
            前缀 = "默认组"
        分组.setdefault(前缀, []).append(列名)
    return 分组


def 构建组矩阵(
    特征列: List[str],
    分组配置路径: Optional[Path] = None,
    schema路径: Optional[Path] = None,
) -> Tuple[np.ndarray, List[str]]:
    """构建特征到组的归属矩阵。"""

    if 分组配置路径 is not None and 分组配置路径.exists():
        分组 = 读取分组配置(分组配置路径)
    else:
        分组 = None

    if 分组 is None and schema路径 is not None:
        分组 = 从schema读取分组(schema路径)

    if 分组 is None:
        分组 = 默认分组策略(特征列)

    组名列表 = list(分组.keys())
    组数 = len(组名列表)
    特征数 = len(特征列)

    if 组数 == 0:
        raise ValueError("分组信息为空，无法构建组矩阵")

    矩阵 = np.zeros((特征数, 组数), dtype=np.float32)
    特征到索引 = {名称: idx for idx, 名称 in enumerate(特征列)}

    for 组索引, 组名 in enumerate(组名列表):
        for 特征名 in 分组.get(组名, []):
            if 特征名 not in 特征到索引:
                raise ValueError(f"分组配置包含未知特征：{特征名}")
            矩阵[特征到索引[特征名], 组索引] = 1.0

    if not np.allclose(矩阵.sum(axis=1), 1.0):
        raise ValueError("每个特征必须且只能属于一个组")

    return 矩阵, 组名列表

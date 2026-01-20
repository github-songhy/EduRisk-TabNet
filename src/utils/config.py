"""配置工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


def _读取yaml(路径: Path) -> Dict[str, Any]:
    if yaml is None:
        with 路径.open("r", encoding="utf-8") as 文件:
            return json.load(文件)
    with 路径.open("r", encoding="utf-8") as 文件:
        return yaml.safe_load(文件) or {}


def _保存yaml(配置: Dict[str, Any], 路径: Path) -> None:
    if yaml is None:
        with 路径.open("w", encoding="utf-8") as 文件:
            json.dump(配置, 文件, ensure_ascii=False, indent=2)
        return
    with 路径.open("w", encoding="utf-8") as 文件:
        yaml.safe_dump(配置, 文件, allow_unicode=True, sort_keys=False)


def 读取配置(路径: str | Path) -> Dict[str, Any]:
    """读取YAML或JSON配置。"""

    路径对象 = Path(路径)
    if not 路径对象.exists():
        raise FileNotFoundError(f"配置文件不存在：{路径对象}")

    if 路径对象.suffix.lower() in {".yaml", ".yml"}:
        return _读取yaml(路径对象)

    if 路径对象.suffix.lower() == ".json":
        with 路径对象.open("r", encoding="utf-8") as 文件:
            return json.load(文件)

    raise ValueError(f"不支持的配置格式：{路径对象.suffix}")


def 递归合并配置(默认配置: Dict[str, Any], 覆盖配置: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并默认配置与覆盖配置。"""

    结果 = dict(默认配置)
    for 键, 值 in 覆盖配置.items():
        if (
            键 in 结果
            and isinstance(结果[键], dict)
            and isinstance(值, dict)
        ):
            结果[键] = 递归合并配置(结果[键], 值)
        else:
            结果[键] = 值
    return 结果


def 保存配置(配置: Dict[str, Any], 结果目录: str | Path, 文件名: str = "最终配置.yaml") -> Path:
    """保存配置到结果目录。"""

    结果路径 = Path(结果目录)
    结果路径.mkdir(parents=True, exist_ok=True)
    目标路径 = 结果路径 / 文件名
    _保存yaml(配置, 目标路径)
    return 目标路径


def 打印配置摘要(配置: Dict[str, Any]) -> str:
    """生成配置摘要文本。"""

    if yaml is None:
        摘要文本 = json.dumps(配置, ensure_ascii=False, indent=2)
    else:
        摘要文本 = yaml.safe_dump(配置, allow_unicode=True, sort_keys=False)
    return "当前配置摘要：\n" + 摘要文本


def 加载并合并配置(
    配置路径: str | Path,
    默认配置: Optional[Dict[str, Any]] = None,
    结果目录: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """读取配置、合并默认值，并可选保存最终配置。"""

    配置内容 = 读取配置(配置路径)
    合并后配置 = 递归合并配置(默认配置 or {}, 配置内容)

    if 结果目录 is not None:
        保存配置(合并后配置, 结果目录)

    print(打印配置摘要(合并后配置))
    return 合并后配置

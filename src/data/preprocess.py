"""数据预处理工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class 预处理统计:
    """预处理统计信息。"""

    数值列: List[str]
    类别列: List[str]
    均值: Dict[str, float]
    中位数: Dict[str, float]
    分组统计: Dict[str, Dict[str, Dict[str, float]]]
    类别映射: Dict[str, Dict[str, int]]
    标准化均值: List[float]
    标准化尺度: List[float]


def _保存统计(统计: 预处理统计, 路径: Path) -> None:
    路径.parent.mkdir(parents=True, exist_ok=True)
    内容 = {
        "数值列": 统计.数值列,
        "类别列": 统计.类别列,
        "均值": 统计.均值,
        "中位数": 统计.中位数,
        "分组统计": 统计.分组统计,
        "类别映射": 统计.类别映射,
        "标准化均值": 统计.标准化均值,
        "标准化尺度": 统计.标准化尺度,
    }
    路径.write_text(json.dumps(内容, ensure_ascii=False, indent=2), encoding="utf-8")


def _读取统计(路径: Path) -> 预处理统计:
    内容 = json.loads(路径.read_text(encoding="utf-8"))
    return 预处理统计(
        数值列=内容.get("数值列", []),
        类别列=内容.get("类别列", []),
        均值=内容.get("均值", {}),
        中位数=内容.get("中位数", {}),
        分组统计=内容.get("分组统计", {}),
        类别映射=内容.get("类别映射", {}),
        标准化均值=内容.get("标准化均值", []),
        标准化尺度=内容.get("标准化尺度", []),
    )


def _推断列(df: pd.DataFrame, 标签列: str, 类别列: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    特征列 = [列 for 列 in df.columns if 列 != 标签列]
    if 类别列 is None:
        类别列 = [列 for 列 in 特征列 if df[列].dtype == "object"]
    数值列 = [列 for 列 in 特征列 if 列 not in 类别列]
    return 数值列, 类别列


def _计算分组统计(df: pd.DataFrame, 数值列: List[str], 分组列: str, 统计方式: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    分组统计: Dict[str, Dict[str, Dict[str, float]]] = {}
    if 分组列 not in df.columns:
        return 分组统计
    分组结果 = df.groupby(分组列)
    for 组名, 子表 in 分组结果:
        分组统计[str(组名)] = {}
        for 列 in 数值列:
            if 统计方式 == "median":
                分组统计[str(组名)][列] = {"value": float(子表[列].median())}
            else:
                分组统计[str(组名)][列] = {"value": float(子表[列].mean())}
    return 分组统计


def _填充数值(df: pd.DataFrame, 数值列: List[str], 策略: str, 分组列: Optional[str], 统计: 预处理统计) -> pd.DataFrame:
    df = df.copy()
    if 策略 == "group" and 分组列 is not None and 分组列 in df.columns:
        for 索引, 行 in df.iterrows():
            组键 = str(行[分组列])
            for 列 in 数值列:
                if pd.isna(行[列]):
                    if 组键 in 统计.分组统计 and 列 in 统计.分组统计[组键]:
                        df.at[索引, 列] = 统计.分组统计[组键][列]["value"]
                    elif 列 in 统计.均值:
                        df.at[索引, 列] = 统计.均值[列]
        return df

    if 策略 == "median":
        for 列 in 数值列:
            df[列] = df[列].fillna(统计.中位数.get(列, 0.0))
        return df

    for 列 in 数值列:
        df[列] = df[列].fillna(统计.均值.get(列, 0.0))
    return df


def _处理类别(df: pd.DataFrame, 类别列: List[str], 编码方式: str, 统计: 预处理统计) -> pd.DataFrame:
    df = df.copy()
    if not 类别列:
        return df

    if 编码方式 == "onehot":
        return pd.get_dummies(df, columns=类别列, dummy_na=True)

    if 编码方式 == "label":
        for 列 in 类别列:
            映射 = 统计.类别映射.get(列, {})
            df[列] = df[列].astype(str).map(映射).fillna(-1).astype(int)
        return df

    return df


def 拟合预处理(
    df: pd.DataFrame,
    标签列: str,
    类别列: Optional[List[str]] = None,
    填充策略: str = "mean",
    分组列: Optional[str] = None,
    标准化: bool = True,
    类别编码: str = "none",
    统计保存路径: Optional[Path] = None,
) -> Tuple[pd.DataFrame, np.ndarray, 预处理统计, np.ndarray]:
    """拟合预处理并返回处理后的特征与缺失指示。"""

    数值列, 类别列 = _推断列(df, 标签列, 类别列)

    均值 = {列: float(df[列].mean()) for 列 in 数值列}
    中位数 = {列: float(df[列].median()) for 列 in 数值列}
    分组统计 = _计算分组统计(df, 数值列, 分组列, 填充策略)

    类别映射: Dict[str, Dict[str, int]] = {}
    if 类别编码 == "label":
        for 列 in 类别列:
            唯一值 = sorted(set(df[列].astype(str).tolist()))
            类别映射[列] = {值: 索引 for 索引, 值 in enumerate(唯一值)}

    统计 = 预处理统计(
        数值列=数值列,
        类别列=类别列,
        均值=均值,
        中位数=中位数,
        分组统计=分组统计,
        类别映射=类别映射,
        标准化均值=[],
        标准化尺度=[],
    )

    特征列 = 数值列 + 类别列
    缺失指示 = (~df[特征列].isna()).astype(int).to_numpy(dtype=np.float32)

    df_filled = _填充数值(df, 数值列, 填充策略, 分组列, 统计)
    df_encoded = _处理类别(df_filled, 类别列, 类别编码, 统计)

    数值矩阵 = df_encoded[数值列].to_numpy(dtype=np.float32) if 数值列 else np.empty((len(df), 0))
    其他列 = [列 for 列 in df_encoded.columns if 列 not in 数值列 + [标签列]]
    其他矩阵 = df_encoded[其他列].to_numpy(dtype=np.float32) if 其他列 else np.empty((len(df), 0))

    if 标准化 and 数值列:
        scaler = StandardScaler()
        数值矩阵 = scaler.fit_transform(数值矩阵)
        统计.标准化均值 = scaler.mean_.tolist()
        统计.标准化尺度 = scaler.scale_.tolist()

    特征矩阵 = np.concatenate([数值矩阵, 其他矩阵], axis=1).astype(np.float32)

    if 统计保存路径 is not None:
        _保存统计(统计, 统计保存路径)

    return df_encoded, 缺失指示, 统计, 特征矩阵


def 应用预处理(
    df: pd.DataFrame,
    标签列: str,
    统计路径: Path,
    填充策略: str = "mean",
    分组列: Optional[str] = None,
    标准化: bool = True,
    类别编码: str = "none",
) -> Tuple[pd.DataFrame, np.ndarray, 预处理统计, np.ndarray]:
    """使用已保存统计进行预处理。"""

    统计 = _读取统计(统计路径)
    数值列 = 统计.数值列
    类别列 = 统计.类别列

    特征列 = 数值列 + 类别列
    缺失指示 = (~df[特征列].isna()).astype(int).to_numpy(dtype=np.float32)

    df_filled = _填充数值(df, 数值列, 填充策略, 分组列, 统计)
    df_encoded = _处理类别(df_filled, 类别列, 类别编码, 统计)

    数值矩阵 = df_encoded[数值列].to_numpy(dtype=np.float32) if 数值列 else np.empty((len(df), 0))
    其他列 = [列 for 列 in df_encoded.columns if 列 not in 数值列 + [标签列]]
    其他矩阵 = df_encoded[其他列].to_numpy(dtype=np.float32) if 其他列 else np.empty((len(df), 0))

    if 标准化 and 数值列 and 统计.标准化均值:
        均值 = np.array(统计.标准化均值, dtype=np.float32)
        尺度 = np.array(统计.标准化尺度, dtype=np.float32)
        数值矩阵 = (数值矩阵 - 均值) / 尺度

    特征矩阵 = np.concatenate([数值矩阵, 其他矩阵], axis=1).astype(np.float32)

    return df_encoded, 缺失指示, 统计, 特征矩阵


def 读取数据(路径: Path, 格式: str) -> pd.DataFrame:
    """读取CSV或Parquet数据。"""

    if 格式.lower() == "csv":
        return pd.read_csv(路径)
    if 格式.lower() == "parquet":
        return pd.read_parquet(路径)
    raise ValueError("不支持的数据格式，只允许csv或parquet")

"""生成合成数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成合成表格数据")
    parser.add_argument("--输出目录", type=str, default="data/synth", help="输出目录")
    parser.add_argument("--样本数", type=int, default=200, help="样本数")
    parser.add_argument("--数值特征数", type=int, default=10, help="数值特征数")
    parser.add_argument("--类别特征数", type=int, default=2, help="类别特征数")
    parser.add_argument("--缺失率", type=float, default=0.1, help="缺失率")
    parser.add_argument("--类别数", type=int, default=4, help="有序标签类别数")
    return parser.parse_args()


def 主函数() -> None:
    参数 = 解析参数()
    输出目录 = Path(参数.输出目录)
    输出目录.mkdir(parents=True, exist_ok=True)

    if np is None or pd is None:
        print("未检测到numpy或pandas，改用标准库生成合成数据")
        _使用标准库生成(参数, 输出目录)
        return

    rng = np.random.default_rng(42)
    数值特征 = rng.normal(size=(参数.样本数, 参数.数值特征数))

    类别特征 = []
    类别列名 = []
    for i in range(参数.类别特征数):
        类别列名.append(f"cat_{i}")
        类别特征.append(rng.integers(0, 3, size=参数.样本数))

    数值列名 = [f"num_{i}" for i in range(参数.数值特征数)]
    数据 = pd.DataFrame(数值特征, columns=数值列名)

    for i, 列名 in enumerate(类别列名):
        数据[列名] = 类别特征[i].astype(str)

    权重 = rng.normal(size=参数.数值特征数)
    连续分数 = 数值特征 @ 权重
    if 参数.类别特征数 > 0:
        类别加成 = sum((类别特征[i] * (i + 1)) for i in range(参数.类别特征数))
        连续分数 += 类别加成

    阈值 = np.quantile(连续分数, np.linspace(0, 1, 参数.类别数 + 1)[1:-1])
    标签 = np.digitize(连续分数, 阈值)
    数据["label"] = 标签.astype(int)

    缺失掩码 = rng.random(size=数据[数值列名 + 类别列名].shape) < 参数.缺失率
    数据.loc[:, 数值列名 + 类别列名] = 数据.loc[:, 数值列名 + 类别列名].mask(缺失掩码)

    数据路径 = 输出目录 / "data.csv"
    数据.to_csv(数据路径, index=False)

    分组 = {
        "group_0": 数值列名[: max(1, 参数.数值特征数 // 2)],
        "group_1": 数值列名[max(1, 参数.数值特征数 // 2):],
        "group_2": 类别列名,
    }

    说明 = {
        "数值列": 数值列名,
        "类别列": 类别列名,
        "标签列": "label",
        "类别数": 参数.类别数,
        "特征分组": 分组,
    }

    说明路径 = 输出目录 / "schema.json"
    说明路径.write_text(json.dumps(说明, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"合成数据已生成：{数据路径}")
    print(f"字段说明已生成：{说明路径}")


def _使用标准库生成(参数: argparse.Namespace, 输出目录: Path) -> None:
    import csv
    import random

    random.seed(42)
    数值列名 = [f"num_{i}" for i in range(参数.数值特征数)]
    类别列名 = [f"cat_{i}" for i in range(参数.类别特征数)]
    列名 = 数值列名 + 类别列名 + ["label"]

    数值特征 = [
        [random.gauss(0, 1) for _ in range(参数.数值特征数)]
        for _ in range(参数.样本数)
    ]
    类别特征 = [
        [str(random.randint(0, 2)) for _ in range(参数.类别特征数)]
        for _ in range(参数.样本数)
    ]

    权重 = [random.gauss(0, 1) for _ in range(参数.数值特征数)]
    连续分数 = [
        sum(数值特征[i][j] * 权重[j] for j in range(参数.数值特征数))
        for i in range(参数.样本数)
    ]
    if 参数.类别特征数 > 0:
        for i in range(参数.样本数):
            连续分数[i] += sum(int(类别特征[i][j]) * (j + 1) for j in range(参数.类别特征数))

    排序分数 = sorted(连续分数)
    阈值列表 = []
    for i in range(1, 参数.类别数):
        索引 = int(len(排序分数) * i / 参数.类别数)
        阈值列表.append(排序分数[索引])

    标签 = []
    for 分数 in 连续分数:
        等级 = 0
        for 阈值 in 阈值列表:
            if 分数 > 阈值:
                等级 += 1
        标签.append(等级)

    数据路径 = 输出目录 / "data.csv"
    with 数据路径.open("w", encoding="utf-8", newline="") as 文件:
        writer = csv.writer(文件)
        writer.writerow(列名)
        for i in range(参数.样本数):
            行 = []
            for 值 in 数值特征[i]:
                if random.random() < 参数.缺失率:
                    行.append("")
                else:
                    行.append(值)
            for 值 in 类别特征[i]:
                if random.random() < 参数.缺失率:
                    行.append("")
                else:
                    行.append(值)
            行.append(标签[i])
            writer.writerow(行)

    分组 = {
        "group_0": 数值列名[: max(1, 参数.数值特征数 // 2)],
        "group_1": 数值列名[max(1, 参数.数值特征数 // 2):],
        "group_2": 类别列名,
    }

    说明 = {
        "数值列": 数值列名,
        "类别列": 类别列名,
        "标签列": "label",
        "类别数": 参数.类别数,
        "特征分组": 分组,
    }
    说明路径 = 输出目录 / "schema.json"
    说明路径.write_text(json.dumps(说明, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"合成数据已生成：{数据路径}")
    print(f"字段说明已生成：{说明路径}")


if __name__ == "__main__":
    主函数()

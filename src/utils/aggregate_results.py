"""实验结果汇总工具。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None


def 汇总结果(结果目录: Path, 输出路径: Path) -> None:
    """汇总各实验结果到CSV。"""

    rows: List[Dict[str, str]] = []
    for 实验目录 in sorted(结果目录.glob("*/")):
        summary_path = 实验目录 / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for 记录 in summary:
            行 = {"experiment": 实验目录.name, "seed": str(记录.get("seed", ""))}
            for k, v in 记录.items():
                if k == "seed":
                    continue
                行[k] = str(v)
            rows.append(行)

        if summary:
            keys = [k for k in summary[0].keys() if k != "seed"]
            mean_row = {"experiment": 实验目录.name, "seed": "mean"}
            std_row = {"experiment": 实验目录.name, "seed": "std"}
            for k in keys:
                values = [float(record[k]) for record in summary]
                if np is not None:
                    mean_row[k] = str(float(np.mean(values)))
                    std_row[k] = str(float(np.std(values)))
                else:
                    mean_row[k] = str(sum(values) / max(1, len(values)))
                    std_row[k] = "nan"
            rows.append(mean_row)
            rows.append(std_row)

    输出路径.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        输出路径.write_text("", encoding="utf-8")
        return

    字段 = sorted({key for row in rows for key in row.keys()})
    with 输出路径.open("w", newline="", encoding="utf-8") as 文件:
        writer = csv.DictWriter(文件, fieldnames=字段)
        writer.writeheader()
        writer.writerows(rows)

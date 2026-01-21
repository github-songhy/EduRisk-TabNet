"""批量运行实验脚本。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

项目根目录 = Path(__file__).resolve().parents[1]
if str(项目根目录) not in sys.path:
    sys.path.insert(0, str(项目根目录))

from src.utils.aggregate_results import 汇总结果


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行实验")
    parser.add_argument("--config_dir", type=str, default="configs/experiments", help="实验配置目录")
    parser.add_argument("--max_experiments", type=int, default=0, help="最多运行实验数，0表示全部")
    parser.add_argument("--skip_completed", action="store_true", help="跳过已完成实验")
    return parser.parse_args()


def 发现实验配置(目录: Path) -> List[Path]:
    return sorted(目录.glob("*.yaml"))


def 已完成(输出目录: Path) -> bool:
    return (输出目录 / "summary.json").exists()


def 主函数() -> None:
    参数 = 解析参数()
    配置目录 = Path(参数.config_dir)
    实验列表 = 发现实验配置(配置目录)

    if 参数.max_experiments > 0:
        实验列表 = 实验列表[: 参数.max_experiments]

    时间戳 = datetime.now().strftime("%Y%m%d_%H%M%S")
    根结果目录 = Path("results")

    for 配置路径 in 实验列表:
        实验名 = 配置路径.stem
        输出目录 = 根结果目录 / 实验名 / 时间戳
        if 参数.skip_completed and 已完成(输出目录):
            print(f"跳过已完成实验：{实验名}")
            continue
        print(f"开始运行实验：{实验名}")
        subprocess.run(
            [
                "python",
                "-m",
                "src.train.runner",
                "--config",
                str(配置路径),
            ],
            check=False,
        )

    汇总结果(根结果目录, 根结果目录 / "summary_all.csv")
    print("实验汇总已完成")


if __name__ == "__main__":
    主函数()

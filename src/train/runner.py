"""训练入口脚本。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import typing

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from src.utils.config import 加载并合并配置
from src.utils.logger import 创建日志器
from src.utils.seed import 设置随机种子

if typing.TYPE_CHECKING:  # pragma: no cover
    from src.models.model_full import EduRiskTabNet模型, 模型配置


def 解析参数() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练入口")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--eval_only", action="store_true", help="仅评估模式")
    parser.add_argument("--checkpoint", type=str, default="", help="评估时的检查点路径")
    return parser.parse_args()


def 构建模型配置(配置: Dict) -> "模型配置":
    from src.losses.dacos import DACOS配置
    from src.models.heads import Head配置
    from src.models.maf import MAF配置
    from src.models.model_full import 模型配置
    from src.models.sasc import SASC配置
    from src.models.tabnet_base import TabNet配置

    maf_cfg = 配置["model"]["maf"]
    tabnet_cfg = 配置["model"]["tabnet"]
    sasc_cfg = 配置["model"]["sasc"]
    head_cfg = 配置["model"]["head"]
    dacos_cfg = 配置["model"]["dacos"]

    return 模型配置(
        maf=MAF配置(
            输入维度=maf_cfg["input_dim"],
            缺失嵌入维度=maf_cfg["embed_dim"],
            上下文维度=maf_cfg["context_dim"],
            使用_film=maf_cfg["use_film"],
            使用缺失嵌入=maf_cfg["use_embedding"],
        ),
        tabnet=TabNet配置(
            输入维度=tabnet_cfg["input_dim"],
            决策维度=tabnet_cfg["decision_dim"],
            上下文维度=tabnet_cfg["attention_dim"],
            决策步数=tabnet_cfg["steps"],
            类别数=tabnet_cfg["num_classes"],
            稀疏正则权重=tabnet_cfg["sparse_weight"],
            gamma=tabnet_cfg["gamma"],
            使用_dm_gfs=tabnet_cfg.get("use_dm_gfs", False),
            组数=tabnet_cfg.get("group_dim", 0),
        ),
        sasc=SASC配置(
            决策维度=sasc_cfg["decision_dim"],
            决策步数=sasc_cfg["steps"],
            启用=sasc_cfg["enabled"],
            early_exit=sasc_cfg["early_exit"],
            阈值=sasc_cfg["threshold"],
            代价阈值=sasc_cfg["cost_epsilon"],
        ),
        head=Head配置(
            输入维度=head_cfg["input_dim"],
            类别数=head_cfg["num_classes"],
            类型=head_cfg["type"],
        ),
        dacos=DACOS配置(
            类别数=dacos_cfg["num_classes"],
            eta=dacos_cfg["eta"],
            alpha=dacos_cfg["alpha"],
            beta=dacos_cfg["beta"],
            tau=dacos_cfg["tau"],
            lambda_risk=dacos_cfg["lambda_risk"],
            warmup_steps=dacos_cfg["warmup_steps"],
        ),
    )


def 构建数据加载器(配置: Dict) -> "torch.utils.data.DataLoader":
    from src.data.dataset import 数据集配置
    from src.data.dataloader import 创建数据加载器, 加载器配置
    data_cfg = 配置["data"]
    loader_cfg = 配置["loader"]
    数据配置 = 数据集配置(
        数据路径=data_cfg["path"],
        数据格式=data_cfg["format"],
        标签列=data_cfg["label_col"],
        类别列=data_cfg.get("categorical_cols"),
        填充策略=data_cfg["impute"],
        分组列=data_cfg.get("group_col"),
        标准化=data_cfg["standardize"],
        类别编码=data_cfg["categorical_encoding"],
        统计路径=data_cfg["stats_path"],
    )
    加载配置 = 加载器配置(
        batch_size=loader_cfg["batch_size"],
        shuffle=loader_cfg["shuffle"],
        num_workers=loader_cfg["num_workers"],
    )
    return 创建数据加载器(数据配置, 加载配置, 训练模式=True)


def 主函数() -> None:
    参数 = 解析参数()
    if torch is None:
        raise ModuleNotFoundError("未检测到torch，请先安装依赖")
    配置 = 加载并合并配置(参数.config)

    时间戳 = datetime.now().strftime("%Y%m%d_%H%M%S")
    输出目录 = Path(配置["experiment"]["output_dir"].format(timestamp=时间戳))
    输出目录.mkdir(parents=True, exist_ok=True)

    日志器 = 创建日志器("主流程", 输出目录)
    日志器.info("开始运行训练流程")

    设备 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    训练加载器 = 构建数据加载器(配置)
    验证加载器 = 构建数据加载器(配置)

    from src.models.model_full import EduRiskTabNet模型
    from src.train.eval import 评估检查点
    from src.train.train import 训练单次

    模型配置 = 构建模型配置(配置)
    模型 = EduRiskTabNet模型(模型配置)

    if 参数.eval_only:
        if not 参数.checkpoint:
            raise ValueError("评估模式下必须指定checkpoint路径")
        评估检查点(模型, 验证加载器, Path(参数.checkpoint), 输出目录, 设备)
        return

    seeds = 配置["experiment"].get("seeds", [42])
    结果汇总 = []

    for seed in seeds:
        设置随机种子(seed)
        日志器.info(f"开始种子{seed}的训练")
        结果 = 训练单次(
            模型,
            训练加载器,
            验证加载器,
            配置,
            输出目录 / f"seed_{seed}",
            设备,
        )
        结果汇总.append(结果.指标)

    统计路径 = 输出目录 / "summary.json"
    统计路径.write_text(json.dumps(结果汇总, ensure_ascii=False, indent=2), encoding="utf-8")
    日志器.info("训练流程结束")


if __name__ == "__main__":
    主函数()

"""训练流程实现。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from src.losses.dacos import ExpectedCost指标, 风险对齐损失, 风险权重
from src.models.model_full import EduRiskTabNet模型
from src.utils.logger import 创建日志器


@dataclass
class 训练输出:
    """训练输出信息。"""

    指标: Dict[str, float]
    最佳模型路径: Path
    最终模型路径: Path


def 计算指标(
    概率: np.ndarray,
    标签: np.ndarray,
    期望代价: float,
    平均步数: float,
) -> Dict[str, float]:
    """计算评估指标。"""

    预测 = 概率.argmax(axis=1)
    return {
        "accuracy": accuracy_score(标签, 预测),
        "macro_f1": f1_score(标签, 预测, average="macro"),
        "weighted_f1": f1_score(标签, 预测, average="weighted"),
        "mae": mean_absolute_error(标签, 预测),
        "expected_cost": 期望代价,
        "avg_steps": 平均步数,
    }


def 训练单次(
    模型: EduRiskTabNet模型,
    训练加载器: torch.utils.data.DataLoader,
    验证加载器: torch.utils.data.DataLoader,
    配置: Dict,
    结果目录: Path,
    设备: torch.device,
) -> 训练输出:
    """执行单次训练。"""

    if torch is None:
        raise ModuleNotFoundError("未检测到torch，请先安装依赖")
    日志器 = 创建日志器("训练", 结果目录)
    模型.to(设备)
    模型.train()

    学习率 = 配置["optimizer"]["lr"]
    优化器 = torch.optim.Adam(模型.parameters(), lr=学习率)
    基础损失函数 = torch.nn.CrossEntropyLoss()

    最佳损失 = float("inf")
    最佳模型路径 = 结果目录 / "best.pt"
    最终模型路径 = 结果目录 / "last.pt"

    训练轮数 = 配置["trainer"]["epochs"]
    lambda_sparse = 配置["loss"]["lambda_sparse"]
    lambda_group = 配置["loss"]["lambda_group"]
    lambda_step = 配置["loss"]["lambda_step"]

    for epoch in range(训练轮数):
        epoch损失 = 0.0
        for x_hat, m, y in 训练加载器:
            x_hat = x_hat.to(设备)
            m = m.to(设备)
            y = y.to(设备)

            prob, _, _, 中间量 = 模型(x_hat, m)
            logits = torch.log(prob.clamp(min=1e-12))
            基础损失 = 基础损失函数(logits, y)

            sparse_loss = 中间量.get("sparse_loss", torch.tensor(0.0, device=设备))
            group_loss = 中间量.get("group_loss", torch.tensor(0.0, device=设备))
            step_loss = 中间量.get("step_loss", torch.tensor(0.0, device=设备))

            dacos配置 = 模型.配置.dacos
            代价矩阵 = 模型.代价矩阵.to(设备)
            risk_loss = 风险对齐损失(prob, y, 代价矩阵, dacos配置.tau)
            risk_weight = 风险权重(epoch + 1, dacos配置)

            总损失 = (
                基础损失
                + lambda_sparse * sparse_loss
                + lambda_group * group_loss
                + lambda_step * step_loss
                + risk_weight * risk_loss
            )

            优化器.zero_grad()
            总损失.backward()
            torch.nn.utils.clip_grad_norm_(模型.parameters(), 5.0)
            优化器.step()

            epoch损失 += 总损失.item()

        日志器.info(f"第{epoch + 1}轮训练损失：{epoch损失:.4f}")
        验证损失 = 评估损失(模型, 验证加载器, 基础损失函数, 设备)
        日志器.info(f"第{epoch + 1}轮验证损失：{验证损失:.4f}")

        if 验证损失 < 最佳损失:
            最佳损失 = 验证损失
            torch.save(模型.state_dict(), 最佳模型路径)

    torch.save(模型.state_dict(), 最终模型路径)

    指标 = 评估模型(模型, 验证加载器, 设备)
    保存指标(指标, 结果目录 / "metrics.csv")

    return 训练输出(指标=指标, 最佳模型路径=最佳模型路径, 最终模型路径=最终模型路径)


def 评估损失(
    模型: EduRiskTabNet模型,
    加载器: torch.utils.data.DataLoader,
    损失函数: torch.nn.Module,
    设备: torch.device,
) -> float:
    """评估验证损失。"""

    if torch is None:
        raise ModuleNotFoundError("未检测到torch，请先安装依赖")
    模型.eval()
    总损失 = 0.0
    with torch.no_grad():
        for x_hat, m, y in 加载器:
            x_hat = x_hat.to(设备)
            m = m.to(设备)
            y = y.to(设备)

            prob, _, _, _ = 模型(x_hat, m)
            logits = torch.log(prob.clamp(min=1e-12))
            总损失 += 损失函数(logits, y).item()
    模型.train()
    return 总损失 / max(1, len(加载器))


def 评估模型(
    模型: EduRiskTabNet模型,
    加载器: torch.utils.data.DataLoader,
    设备: torch.device,
) -> Dict[str, float]:
    """评估模型并返回指标。"""

    if torch is None:
        raise ModuleNotFoundError("未检测到torch，请先安装依赖")
    模型.eval()
    概率列表: List[np.ndarray] = []
    标签列表: List[np.ndarray] = []
    步数列表: List[np.ndarray] = []
    期望代价列表: List[float] = []

    with torch.no_grad():
        for x_hat, m, y in 加载器:
            x_hat = x_hat.to(设备)
            m = m.to(设备)
            y = y.to(设备)

            prob, _, _, 中间量 = 模型(x_hat, m)
            概率列表.append(prob.cpu().numpy())
            标签列表.append(y.cpu().numpy())
            used_steps = 中间量.get("used_steps")
            if used_steps is not None:
                步数列表.append(used_steps.cpu().numpy())
            期望代价 = 模型.计算期望代价(prob, y).item()
            期望代价列表.append(期望代价)

    概率 = np.concatenate(概率列表, axis=0)
    标签 = np.concatenate(标签列表, axis=0)
    平均步数 = float(np.mean(np.concatenate(步数列表))) if 步数列表 else float("nan")
    期望代价均值 = float(np.mean(期望代价列表)) if 期望代价列表 else float("nan")

    指标 = 计算指标(概率, 标签, 期望代价均值, 平均步数)
    return 指标


def 保存指标(指标: Dict[str, float], 路径: Path) -> None:
    """保存指标为CSV。"""

    路径.parent.mkdir(parents=True, exist_ok=True)
    with 路径.open("w", newline="", encoding="utf-8") as 文件:
        writer = csv.writer(文件)
        writer.writerow(["metric", "value"])
        for k, v in 指标.items():
            writer.writerow([k, v])

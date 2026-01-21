# EduRisk-TabNet

## 项目简介
- 面向学业预警表格数据的可解释模型框架，包含缺失感知、组级特征选择、自适应步长与代价对齐决策。

## 目录概览
- docs/：需求与设计文档。
- configs/：实验与模块配置。
- src/：核心实现。
- scripts/：数据与绘图脚本。
- tests/：测试用例。
- results/、figures/：结果与图表输出目录。

## 快速开始
1. 安装依赖
   - `pip install -r requirements.txt`
2. 生成合成数据
   - `python scripts/make_synth_data.py`
3. 冒烟训练
   - `python -m src.train.runner --config configs/experiments/smoke.yaml`
4. 绘图与报告
   - `python scripts/plot_all.py --config configs/plot.yaml`

## 全量实验与消融
- 一键运行全部实验矩阵
  - `python scripts/run_all_experiments.py`
- 快速验证仅跑一个实验
  - `python scripts/run_all_experiments.py --max_experiments 1`
- 汇总结果
  - 汇总结果将输出到 `results/summary_all.csv`，包含每个实验与各seed的指标及均值方差。

## 配置说明
- configs/experiments/：主实验、消融与冒烟训练配置。
- configs/model.yaml：模型超参与模块开关配置。
- configs/data.yaml：数据与缺失处理配置。
- configs/groups.yaml：分组配置示例。

## 里程碑
- 里程碑1：工程骨架与文档完善。
- 里程碑2：合成数据、模型骨架与冒烟训练。
- 里程碑3：完整模块与主实验/消融实验。
- 里程碑4：绘图与报告输出。

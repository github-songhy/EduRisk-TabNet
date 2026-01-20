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

## 快速开始（占位）
1. 安装依赖
   - `pip install -r requirements.txt`
2. 生成合成数据
   - `python scripts/make_synth_data.py --config configs/data/synth.yaml`
3. 冒烟训练
   - `python -m src.train.runner --config configs/experiments/smoke.yaml`
4. 绘图与报告
   - `python scripts/plot_all.py --config configs/plot.yaml`

## 配置说明（占位）
- configs/experiments/：主实验、消融与冒烟训练配置。
- configs/model/：各模块超参数与开关。
- configs/data/：数据与缺失处理配置。

## 里程碑（占位）
- 里程碑1：工程骨架与文档完善。
- 里程碑2：合成数据、模型骨架与冒烟训练。
- 里程碑3：完整模块与主实验/消融实验。
- 里程碑4：绘图与报告输出。

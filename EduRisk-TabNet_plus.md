# EduRisk-TabNet 方法框架（面向学业预警的 TabNet 结构与学习策略增强）

> 说明：本文档仅组织“方法框架”所需内容（论文主张、总体技术路线、4 个创新点写作骨架、实验设计总方案、分创新点参考文献）。

---

## 1）一句话论文主张

面向具有**高缺失、强相关指标簇、等级有序且错判代价不对称**特征的学生学业预警表格数据，本文提出 **EduRisk-TabNet**：在 TabNet 上引入**缺失感知融合**、**双 Mask 的组级特征选择**、**样本级自适应决策步长**与**代价-序级敏感决策对齐损失**，以期在保持可解释性的同时获得更稳健且更符合预警决策需求的预测输出。

---

## 2）总体技术路线

### 2.1 问题定义与符号体系

- 数据集：$\mathcal{D}=\{(\mathbf{x}_n,y_n)\}_{n=1}^{N}$，其中 $\mathbf{x}_n\in\mathbb{R}^{D}$ 为表格特征，$y_n\in\{0,1,\dots,K-1\}$ 为预警等级（天然有序）。
- 缺失指示：$\mathbf{m}_n\in\{0,1\}^{D}$，$m_{n,d}=1$ 表示特征 $d$ 观测到，$m_{n,d}=0$ 表示缺失。
- 特征组：将 $D$ 维特征划分为 $G$ 个组（例如“成绩域/考勤域/行为域/经济域”等），用组归属矩阵 $\mathbf{G}\in\{0,1\}^{D\times G}$ 表示，满足每个特征只属于一个组：$\sum_{g=1}^{G}G_{d,g}=1$。

### 2.2 基础 TabNet 回顾（改进基座）

令输入特征（经嵌入/归一化后）为 $\mathbf{f}\in\mathbb{R}^{D}$。TabNet 通过 $T$ 个 decision steps 逐步选择特征并累积决策表征。

**(1) 第 $t$ 步特征选择 mask（Attentive Transformer）**

用 $\mathbf{a}^{(t-1)}$ 表示上一步的注意力上下文，用 $\mathbf{P}^{(t-1)}\in\mathbb{R}^{D}$ 表示先验（prior）以抑制重复选择。则特征选择 mask 记为 $\mathbf{M}^{(t)}\in[0,1]^D$：

$$
\mathbf{M}^{(t)}=\mathrm{sparsemax}\Big(\mathbf{P}^{(t-1)}\odot \mathrm{Att}^{(t)}(\mathbf{a}^{(t-1)})\Big), \quad t=1,\dots,T.
$$

其中 $\odot$ 为逐元素乘，$\mathrm{sparsemax}(\cdot)$ 产生稀疏分布以提升可解释性。

**(2) 第 $t$ 步特征变换（Feature Transformer）与决策累积**
$$
\mathbf{z}^{(t)} = \mathbf{M}^{(t)}\odot \mathbf{f},
$$
$$
[\mathbf{d}^{(t)},\mathbf{a}^{(t)}]=\mathrm{FT}^{(t)}(\mathbf{z}^{(t)}),
$$
$$
\mathbf{d}_{\text{out}}=\sum_{t=1}^{T}\mathrm{ReLU}(\mathbf{d}^{(t)}).
$$

**(3) prior 更新（鼓励跨步多样化选择）**

$$
\mathbf{P}^{(t)}=\mathbf{P}^{(t-1)}\odot\big(\gamma-\mathbf{M}^{(t)}\big),\quad \gamma>1.
$$

**(4) 稀疏正则（保持 mask 可解释性）**

常用做法是对各步 mask 的“熵类”项做惩罚（记为 $L_{\text{sparse}}$），以鼓励更稀疏/更集中的选择分布。

### 2.3 EduRisk-TabNet 总体结构（把四个策略串联成一条可实现的流水线）

从“数据 → 表征 → 选择 → 决策 → 损失/推理规则”串联四个模块：

```text
原始样本 x (含缺失) 
   │  生成缺失指示 m
   ▼
[模块1] 缺失感知融合 MAF: (x, m) →  ᵜ x
   ▼
TabNet Encoder（被两处增强）
   ├─ [模块2] 双Mask组级特征选择 DM-GFS: 组mask → 特征mask
   └─ [模块3] 样本级step调节 SASC: 逐步输出 + 样本自适应加权/早停
   ▼
分类头（softmax 或 ordinal head）得到 p(y|x)
   ▼
[模块4] 代价-序级敏感 DACOS: 决策对齐训练 + Bayes最小期望代价推理
```

### 2.4 统一训练目标与推理规则

令模型输出类别分布为 $\mathbf{p}(\mathbf{x})\in\Delta^{K}$（$K$ 维概率单纯形）。EduRisk-TabNet 的总体训练目标建议写为：

$$
L = L_{\text{base}} + \lambda_{\text{sparse}}L_{\text{sparse}} + \lambda_{\text{group}}L_{\text{group}} + \lambda_{\text{step}}L_{\text{step}} + \lambda_{\text{risk}}L_{\text{risk}}.
$$

- $L_{\text{base}}$：基础分类损失（softmax CE 或 ordinal head 的阈值式损失）。
- $L_{\text{sparse}}$：TabNet 原生 mask 稀疏正则。
- $L_{\text{group}}$：组级选择约束/多样性约束（来自模块2）。
- $L_{\text{step}}$：鼓励少步计算的“计算代价项/ponder cost”（来自模块3）。
- $L_{\text{risk}}$：按代价矩阵对齐部署决策的风险损失（来自模块4）。

推理阶段不再仅使用 $\arg\max$，而采用 Bayes 最小期望代价决策（模块4）：

$$
\hat{y}(\mathbf{x})=\arg\min_{a\in\{0,\dots,K-1\}}\;R(a\mid \mathbf{x}),\quad R(a\mid \mathbf{x})=\sum_{y=0}^{K-1}C(y,a)\,p(y\mid \mathbf{x}).
$$

---

## 3）每个创新点的小节写作骨架（动机→方法→实现→实验→小结）

> 统一要求：每个创新点都回答“为什么需要/做了什么/怎么做/凭什么有效/与谁比/有什么代价”。


### 3.1 创新点1：缺失感知融合策略（Missingness-Aware Fusion, MAF）

#### 3.1.1 动机（为什么需要）

1. **学业预警数据的缺失并非纯噪声**：学生信息系统中“未提交/未记录/不参与”往往与风险水平存在相关性，属于“缺失本身携带信号”的情形。
2. **常规填充（均值/众数/0）会抹平缺失机制**：将缺失当作随机噪声处理，可能把“制度性缺失/行为性缺失”的结构信息变成伪数值，导致模型学习到不可控偏差。
3. **TabNet 的可解释性依赖 mask**：若缺失被粗暴填充，mask 可能解释的是“填充值”而非“真实观测+缺失模式”，从而削弱解释可信度。

> 研究问题表述建议：在不改变 TabNet 主体结构的前提下，如何让模型将“缺失模式”作为条件信息参与特征表示学习，从而更符合学业预警数据生成机制。

#### 3.1.2 方法（做了什么 + 关键公式）

给定原始输入 $\mathbf{x}$ 与缺失指示 $\mathbf{m}$：

**(1) 轻量填充得到可计算输入**

$$
\hat{\mathbf{x}}=\mathrm{Impute}(\mathbf{x},\mathbf{m}).
$$

其中 $\mathrm{Impute}$ 可为均值/中位数/分组统计等（论文中需声明：填充仅为数值可计算，缺失语义由后续模块显式建模）。

**(2) 缺失模式编码与条件调制（FiLM式融合）**

- 缺失嵌入（逐特征缺失信号）：$\mathbf{e}_{\mathbf{m}}=\mathrm{Emb}(\mathbf{m})$。
- 缺失模式上下文（全局缺失结构）：$\mathbf{c}_{\mathbf{m}}=f_{\text{enc}}(\mathbf{m})$。
- 生成调制参数：

$$
[\boldsymbol{\gamma},\boldsymbol{\beta}]=f_{\text{film}}(\mathbf{c}_{\mathbf{m}}).
$$

- 融合输出（送入 TabNet 的最终输入）：

$$
\tilde{\mathbf{x}} = \boldsymbol{\gamma}\odot \hat{\mathbf{x}} + \boldsymbol{\beta} + g(\mathbf{e}_{\mathbf{m}}).
$$

其中 $f_{\text{enc}}, f_{\text{film}}, g(\cdot)$ 均可取轻量 MLP，$\tilde{\mathbf{x}}$ 作为 TabNet 的输入特征表征。

> 写作强调点：该设计将“缺失”从数据清洗问题提升为“条件信息”，实现“缺失机制→特征表征”的可微映射。

#### 3.1.3 实现（怎么做）

- 输入侧实现：在数据加载阶段同时输出 $(\hat{\mathbf{x}},\mathbf{m})$。
- 参数规模控制：
  - $\mathrm{Emb}(\mathbf{m})$ 可为按特征的两值嵌入（观测/缺失），避免维度膨胀。
  - $f_{\text{enc}}$ 对 $\mathbf{m}$ 可用 1–2 层 MLP，输出维度与 $\hat{\mathbf{x}}$ 对齐或作为共享条件向量。
- 与 TabNet 的接口：将 $\tilde{\mathbf{x}}$ 替换原始输入 $\mathbf{f}$（或替换归一化后特征），其余 TabNet 前向保持一致。

#### 3.1.4 实验（与谁比 + 怎么验证 + 凭什么有效）

- **对比对象（与谁比）**
  1. TabNet + 常规填充（均值/众数）。
  2. TabNet + “缺失指示拼接”（$[\hat{\mathbf{x}},\mathbf{m}]$ 直接拼接）作为强基线。
  3. 其他表格模型（如 XGBoost/LightGBM/CatBoost）在相同缺失处理下的对照。

- **消融设计（怎么验证）**
  - 去掉 $g(\mathbf{e}_{\mathbf{m}})$ 仅保留 FiLM：检验“逐特征缺失嵌入”的贡献。
  - 去掉 FiLM 仅保留 $g(\mathbf{e}_{\mathbf{m}})$：检验“全局缺失模式调制”的贡献。

- **有效性依据（凭什么有效）**
  - 当缺失与标签相关（informative missingness）时，显式输入 $\mathbf{m}$ 并进行条件调制，可使模型学习到“缺失模式→风险”的稳定关联，而不是依赖填充值的偶然统计。

#### 3.1.5 小结（贡献与代价）

- **贡献**：在不破坏 TabNet 主体可解释机制（mask）的前提下，把缺失模式纳入表征学习，使 mask 的解释更贴近“真实观测+缺失机制”。
- **代价/风险**：
  - 增加少量参数与超参（嵌入维度、调制网络宽度）。
  - 若缺失近似 MCAR（完全随机缺失），该模块可能收益有限，需通过消融实证。


### 3.2 创新点2：双 Mask 的组级特征选择策略（Dual-Mask Group-wise Feature Selection, DM-GFS）

#### 3.2.1 动机（为什么需要）

1. **学业指标天然存在“域/组”结构**：例如课程成绩、考勤、消费、心理量表、学习行为等，同组内特征高相关、同质性强。
2. **原始 TabNet 的逐特征选择在强相关场景易“来回徘徊”**：不同 step 可能在同一指标域内反复选择相近特征，造成（i）解释冗余，（ii）选择不稳定，（iii）跨域交互被弱化。
3. **组级先验有助于解释与治理**：预警系统更关心“哪个指标域导致风险上升”，而非单个特征的偶然波动。

> 研究问题表述建议：如何在 TabNet 的逐步注意力选择中注入“组结构先验”，实现“先选组、再选特征”的层级化可解释选择。

#### 3.2.2 方法（做了什么 + 关键公式）

令输入为 $\tilde{\mathbf{x}}\in\mathbb{R}^{D}$（来自模块1），组归属矩阵为 $\mathbf{G}\in\{0,1\}^{D\times G}$。

**(1) 组级表征**

将特征聚合到组空间：

$$
\mathbf{f}_{g}=\tilde{\mathbf{x}}\mathbf{G}\in\mathbb{R}^{G}.
$$

**(2) 组级 mask（第 $t$ 步）**

$$
\mathbf{m}_{g}^{(t)}=\mathrm{sparsemax}\big(\mathrm{Att}_{g}^{(t)}(\mathbf{f}_{g},\mathbf{a}^{(t-1)})\big),\quad \mathbf{m}_{g}^{(t)}\in[0,1]^{G}.
$$

**(3) 组 mask 映射回特征空间，得到组门控向量**
$$
\tilde{\mathbf{m}}^{(t)}=\mathbf{G}\mathbf{m}_{g}^{(t)}\in\mathbb{R}^{D}.
$$

**(4) 特征级 logits 与双 mask 融合得到最终特征 mask**

先计算特征选择 logits：

$$
\mathbf{s}_{f}^{(t)}=\mathrm{Att}_{f}^{(t)}(\tilde{\mathbf{x}},\mathbf{a}^{(t-1)}),
$$

再进行组门控重加权并 sparsemax：

$$
\mathbf{m}_{f}^{(t)}=\mathrm{sparsemax}\big(\mathbf{s}_{f}^{(t)}\odot \tilde{\mathbf{m}}^{(t)}\big),
$$

最终将 $\mathbf{M}^{(t)}:=\mathbf{m}_{f}^{(t)}$ 送入 Feature Transformer。

**(5) 组多样性/去冗余正则（可选，但建议写进论文）**

为了减少连续步骤反复选择同一组，可加入相邻步的组重叠惩罚：

$$
L_{\text{group}}=\frac{1}{T-1}\sum_{t=2}^{T}\left(\mathbf{m}_{g}^{(t)}\cdot \mathbf{m}_{g}^{(t-1)}\right).
$$

> 写作强调点：DM-GFS 并非简单“加 group embedding”，而是把组信息显式地参与每一步 mask 的生成与约束，从而改变了 TabNet 的选择动力学。

#### 3.2.3 实现（怎么做）

- $\mathbf{G}$ 的构建：由专家知识/字段字典/统计聚类（如相关系数聚类）得到，需在论文中说明构建原则与稳定性。
- 计算图实现：
  - $\mathrm{Att}_{g}$ 与 $\mathrm{Att}_{f}$ 可共享部分参数或完全独立（需在消融中验证）。
  - 为避免 $\mathbf{f}_g$ 过于粗糙，可使用组内池化（mean/sum）或可学习投影。
- 解释输出：同时输出组级重要性 $\mathbf{m}_g^{(t)}$ 与特征级重要性 $\mathbf{m}_f^{(t)}$，支持“域级-指标级”两层解释。

#### 3.2.4 实验（与谁比 + 怎么验证 + 凭什么有效）

- **对比对象（与谁比）**
  1. 原始 TabNet（单 mask）。
  2. TabNet + group one-hot 拼接（仅提供组标识但不改变选择机制）。
  3. 组稀疏模型（如 Group Lasso / 稳定选择）在同数据上的对照（作为“组级选择”思想的传统基线）。

- **消融设计（怎么验证）**
  - 去掉组多样性正则 $L_{\text{group}}$。
  - 仅使用组 mask（不再细化到特征）：检验“先选组再选特征”的必要性。
  - 仅使用特征 mask（退化回原 TabNet）：检验组门控的贡献。

- **有效性依据（凭什么有效）**
  - 强相关特征域内，组门控相当于对特征空间施加结构化先验，使注意力在更高层级先完成“域的筛选”，降低在同质特征间随机游走导致的不稳定与解释冗余。

#### 3.2.5 小结（贡献与代价）

- **贡献**：实现“组→特征”的层级化可解释选择，增强跨域信息整合的可控性。
- **代价/风险**：
  - 需要稳定合理的分组方案；分组错误可能引入偏置。
  - 增加一套组级注意力计算与正则项，训练更依赖超参（如 $\lambda_{\text{group}}$）。


### 3.3 创新点3：样本级 step 调节策略（Sample-wise Adaptive Step Control, SASC）

#### 3.3.1 动机（为什么需要）

1. **样本难度异质性**：同一批学生样本中存在“易判别”与“边界模糊”两类；固定 $T$ 步对所有样本一刀切，可能带来不必要计算或过拟合风险。
2. **TabNet 的 step 本质是“逐步推理/逐步选特征”**：这与动态计算（adaptive computation）思想天然兼容。
3. **效率与泛化的双重诉求**：学业预警系统常需要周期性全量预测，推理成本与延迟具有现实约束；同时希望在难样本上保留足够表达能力。

> 研究问题表述建议：如何让 TabNet 根据样本难度自适应地分配 decision steps，使“计算预算”成为可学习资源，并在不牺牲可解释性的条件下实现更优的准确率-效率权衡。

#### 3.3.2 方法（做了什么 + 关键公式）

SASC 采用“软步长加权”（soft halting）方式：对每个样本 $b$、每个 step $t$ 计算权重 $\alpha_{b,t}$，用加权和替代固定求和。

定义初始剩余权重：$r_0 = 1$

**(1) 每步决策表征**

TabNet 产生 $\mathbf{d}_{b}^{(t)}\in\mathbb{R}^{N_d}$。

**(2) 计算样本第t步停止概率p，并计算该步权重$\alpha$**
$$
\ p_{b,t}=\sigma(\mathbf{w}^{\top}\mathbf{d}_{b}^{(t)}+b),
$$
$$
\alpha_{b,t}=r_{t-1}\odot p_t
$$

**(3) 样本级聚合输出**
$$
\mathbf{d}_{b}=\sum_{t=1}^{T}\alpha_{b,t}\,\mathbf{d}_{b}^{(t)}.
$$

**(4) 计算代价项（鼓励更少的有效步数）**

定义样本的期望步数：

$$
\mathbb{E}[T_b]=\sum_{t=1}^{T}r_{b,t-1}.
$$

将其作为正则加入：

$$
L_{\text{step}}=\frac{1}{B}\sum_{b=1}^{B}\mathbb{E}[T_b].
$$

#### 3.3.3 实现（怎么做）

- 结构实现：
  - 在每个 step 的 $\mathbf{d}^{(t)}$ 后接一个轻量线性层/MLP 输出 $p_{b,t}$。
  - 训练期用 soft 权重保证可微；推理期可选择 soft 聚合或根据累计权重进行 early-exit。
- 与模块4联动（建议写成“方法亮点”）：若每步都能得到中间预测分布 $\mathbf{p}^{(t)}$，可用“期望代价下降是否足够”作为停止依据：

$$
r^{(t)}(\mathbf{x})=\min_{a} \sum_{y} C(y,a)p^{(t)}(y\mid\mathbf{x}),
$$

当 $r^{(t-1)}-r^{(t)}<\varepsilon$ 时停止继续计算。

#### 3.3.4 实验（与谁比 + 怎么验证 + 凭什么有效）

- **对比对象（与谁比）**
  1. 固定步长 TabNet（同参数量、不同 $T$）。
  2. 将 $T$ 设小的轻量 TabNet、将 $T$ 设大的高精度 TabNet（展示效率-性能曲线）。

- **消融设计（怎么验证）**
  - 去掉 $L_{\text{step}}$：观察是否出现“总是用满步数”的退化。
  - 不做逐样本权重（$\alpha_{b,t}=1/T$）：退化为平均融合。

- **有效性依据（凭什么有效）**
  - 自适应计算理论指出：对易样本减少计算可在不显著影响性能的情况下节省预算；对难样本保留更多步数可维持表达能力。该思想迁移到 TabNet 的 step 机制具有合理性。

#### 3.3.5 小结（贡献与代价）

- **贡献**：把 TabNet 的 step 从固定超参转为样本自适应资源分配，形成可控的“性能-效率”权衡机制。
- **代价/风险**：
  - 训练目标更复杂（需联合调 $\lambda_{\text{step}}$），且可能出现权重塌缩（所有质量集中到少数步或平均分散）。
  - 若需要真实推理加速，必须配合 early-exit 实现与硬件计时评测。


### 3.4 创新点4：代价和序级敏感策略（Decision-Aligned Cost & Ordinal Sensitive Learning, DACOS）

#### 3.4.1 动机（为什么需要）

1. **学业预警等级具有明确序级**：将其当作无序多分类会忽略“错得远更严重”的结构信息。
2. **错判代价不对称**：例如“高风险判为低风险（漏报）”通常比“低风险判为高风险（误报）”更不可接受。
3. **训练目标与部署决策不一致**：标准交叉熵优化 $\arg\max$ 精度，但实际系统需要最小化业务代价（期望风险）。

> 研究问题表述建议：如何把“序级结构+错判代价体系”从评价指标层面前移到训练与推理规则中，使模型输出概率分布更适合做代价最小化决策。

#### 3.4.2 方法（做了什么 + 关键公式）

**(1) 定义代价矩阵 $C$（体现序级与不对称）**

对真实等级 $y$ 与采取动作/预测等级 $a$：

$$
C(y,a)=\eta (y-a)^2 + \alpha\max(0,y-a) + \beta\max(0,a-y),
$$

其中常用设定为 $\alpha>\beta$ 以强调“低估/漏报更贵”。

**(2) 输出层选择：softmax 或 ordinal head（建议优先 ordinal）**

- softmax：

$$
\mathbf{p}=\mathrm{softmax}(\text{logits}),\quad \mathbf{p}\in\mathbb{R}^{K}.
$$

- ordinal head：建模累积概率 $q_j(\mathbf{x})=P(y>j\mid\mathbf{x})$：

$$
q_j(\mathbf{x})=\sigma(s_j(\mathbf{x})),\quad j=0,\dots,K-2.
$$

并由累积概率恢复类别概率：

$$
P(y=0)=1-q_0,
$$
$$
P(y=k)=q_{k-1}-q_{k}\quad (1\le k\le K-2),
$$
$$
P(y=K-1)=q_{K-2}.
$$

**(3) Bayes 期望风险与决策对齐损失（核心）**

对每个动作 $a$ 的期望代价（风险）：

$$
R(a\mid \mathbf{x})=\sum_{y=0}^{K-1} C(y,a)\,p(y\mid\mathbf{x}).
$$

推理时采用 Bayes 最小风险：

$$
\hat{y}(\mathbf{x})=\arg\min_{a} R(a\mid\mathbf{x}).
$$

为使训练可微，将“选择最小风险动作”软化为 softmin 分布（温度 $\tau>0$）：

$$
\pi(a\mid\mathbf{x})=\frac{\exp\big(-R(a\mid\mathbf{x})/\tau\big)}{\sum_{a'}\exp\big(-R(a'\mid\mathbf{x})/\tau\big)}.
$$

把“真实等级对应动作应当最优”转为交叉熵形式的风险对齐损失：

$$
L_{\text{risk}}(\mathbf{x},y)=-\log \pi(a=y\mid\mathbf{x}).
$$

**(4) 总损失组合（稳定训练）**

建议把基础损失（CE 或 ordinal 阈值损失）与风险损失联合：

$$
L = L_{\text{CE/ordinal}} + \lambda_{\text{risk}}L_{\text{risk}} + \lambda_{\text{sparse}}L_{\text{sparse}}.
$$

> 写作强调点：DACOS 不仅“换一个 loss”，还明确规定了部署阶段的 Bayes 决策规则，使训练目标与推理动作一致（decision-aligned）。

#### 3.4.3 实现（怎么做）

- 代价矩阵 $\mathbf{C}\in\mathbb{R}^{K\times K}$：作为常量张量参与矩阵乘法计算（对 batch：$\mathbf{R}=\mathbf{p}\mathbf{C}$）。
- 训练技巧：
  - $\lambda_{\text{risk}}$ 可采用 warm-up（前若干 epoch 较小，后期增大），以避免训练初期不稳定。
  - $\tau$ 控制 softmin 的“近似硬度”，需在验证集上调优。
- 评估一致性：论文中应同时报告（i）传统分类指标（如 Macro-F1），（ii）期望代价（Expected Cost）等决策指标，以体现“决策对齐”的必要性。

#### 3.4.4 实验（与谁比 + 怎么验证 + 凭什么有效）

- **对比对象（与谁比）**
  1. 标准 CE（argmax 推理）。
  2. 类别加权 CE / Focal Loss（作为“代价/不平衡处理”的常见基线）。
  3. 传统序级方法（如 proportional odds / SVOR 等）与树模型的 ordinal 变体（若实现）。

- **消融设计（怎么验证）**
  - 只用 $L_{\text{CE/ordinal}}$；只用 $L_{\text{risk}}$；二者联合。
  - 固定 $\mathbf{C}$ 的不同构造（仅平方项 vs 加入方向惩罚）对错判分布的影响。

- **有效性依据（凭什么有效）**
  - Bayes 决策理论指出：当代价矩阵给定时，最优决策为最小化期望风险；以 $L_{\text{risk}}$ 直接优化 softmin 风险，可促使输出分布更贴近“可用于最小风险决策”的形态。

#### 3.4.5 小结（贡献与代价）

- **贡献**：将“序级结构 + 业务代价”从评价层上升为训练与推理层的统一目标，形成可解释且可部署的决策对齐预警模型。
- **代价/风险**：
  - 需要明确且可辩护的代价矩阵定义（论文需给出领域解释）。
  - 引入额外超参（$\lambda_{\text{risk}},\tau,\alpha,\beta,\eta$），调参工作量增加。

---

## 4）实验设计总方案（基线/指标/消融/对比）

### 4.1 基线模型（Baselines）

建议从“强表格基线 + 可解释基线 + 任务结构基线”三类组织：

1. **树模型强基线**：XGBoost、LightGBM、CatBoost（表格任务常见强基线，尤其对非线性与缺失处理友好）。
2. **深度表格基线**：原始 TabNet、FT-Transformer 或其他表格神经网络（若你论文范围允许）。
3. **可解释/稀疏基线**：L1/L2 正则 Logistic Regression、Group Lasso（用于验证“组级选择/稀疏解释”的必要性）。
4. **序级/代价敏感基线**：
   - ordinal regression（如 proportional odds / SVOR）。
   - 类别加权 CE / focal loss（与 DACOS 对照）。

### 4.2 评价指标（Metrics）

为满足“分类质量 + 序级一致性 + 决策代价 + 效率”四条主线，建议至少包含：

1. **分类指标**：Accuracy、Macro-F1、Weighted-F1。
2. **序级指标**：MAE（将等级视作有序标号后的平均绝对误差）、Quadratic Weighted Kappa（若合适）。
3. **代价指标**：Expected Cost（按你的 $\mathbf{C}$ 计算的平均期望代价）。
4. **效率指标**（对应模块3）：
   - 平均使用步数 $\mathbb{E}[T_b]$；
   - 推理时延/吞吐（在同硬件与批大小下测量）。

> 重要写作建议：把 Expected Cost 作为主指标之一，用于论证 DACOS 的“意义而非仅仅准确率”。

### 4.3 消融实验（Ablation）

建议采用“逐模块 + 交互项”两层消融：

- 逐模块：
  - Base TabNet
  - + MAF
  - + DM-GFS
  - + SASC
  - + DACOS
  - Full EduRisk-TabNet

- 交互项（验证模块之间是否互补）：
  - MAF × DM-GFS（缺失表征是否影响组选择稳定性）
  - SASC × DACOS（risk 下降作为停止依据是否更合理）

### 4.4 对比实验（Comparisons）

- 与树模型对比：强调“可解释性（mask）+ 序级/代价对齐”的综合优势，而非仅比准确率。
- 与 ordinal/cost-sensitive 传统方法对比：强调“端到端学习 + 可解释选择 + 决策一致性”。

### 4.5 解释性与稳定性评估（建议作为实验子任务）

1. **解释一致性**：不同随机种子/不同训练子集下的特征重要性一致性（可用排名相关系数、Jaccard 等）。
2. **组级解释**：输出 $\mathbf{m}_g^{(t)}$ 的跨步多样性与跨样本模式（对应 $L_{\text{group}}$ 的作用）。

### 4.6 复现性与统计检验

- 固定随机种子、多次重复训练（例如 5 次）报告均值与方差。
- 如条件允许，做显著性检验（例如配对 $t$ 检验或 Wilcoxon）以避免偶然性。

---

## 5）参考文献（每个创新点 ≥10 篇；并给出“适合论证什么”）

> 注意：以下参考文献均为公开可检索的真实文献；写作时请统一引用格式（GB/T 7714 或学校模板）。


### 创新点1：缺失感知融合策略（MAF）

[1] Arik S O, Pfister T. TabNet: Attentive Interpretable Tabular Learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 2021.

[2] Rubin D B. Inference and Missing Data. *Biometrika*, 1976, 63(3): 581–592.

[3] Little R J A, Rubin D B. *Statistical Analysis with Missing Data* (2nd ed.). Wiley, 2002.

[4] Schafer J L. *Analysis of Incomplete Multivariate Data*. Chapman & Hall/CRC, 1997.

[5] van Buuren S. *Flexible Imputation of Missing Data* (2nd ed.). Chapman & Hall/CRC, 2018.

[6] van Buuren S, Groothuis-Oudshoorn K. mice: Multivariate Imputation by Chained Equations in R. *Journal of Statistical Software*, 2011, 45(3).

[7] Che Z, Purushotham S, Cho K, et al. Recurrent Neural Networks for Multivariate Time Series with Missing Values. *Scientific Reports*, 2018, 8: 6085.

[8] Yoon J, Jordon J, van der Schaar M. GAIN: Missing Data Imputation using Generative Adversarial Nets. *Proceedings of ICML (PMLR)*, 2018.

[9] Cao W, Wang D, Li J, et al. BRITS: Bidirectional Recurrent Imputation for Time Series. *Advances in Neural Information Processing Systems (NeurIPS)*, 2018.

[10] Perez E, Strub F, de Vries H, et al. FiLM: Visual Reasoning with a General Conditioning Layer. *Proceedings of AAAI Conference on Artificial Intelligence*, 2018.

[11] Bianchi F M, Livi L, Mikalsen K Ø, et al. Learning representations for multivariate time series with missing data using Temporal Kernelized Autoencoders. *arXiv preprint*, 2018.

**适用性说明：**

[1] 适合论证 TabNet 作为表格深度学习基座，其“逐步特征选择 + 稀疏可解释 mask”的基本机制与改造接口。

[2] 适合论证缺失机制可在一定条件下影响推断方式（MAR 等概念），为“缺失不是简单噪声”的论证提供统计学基础。

[3] 适合论证缺失数据分析的经典理论与方法谱系，支撑你对学业预警数据缺失特性的严谨讨论。

[4] 适合论证多变量缺失数据的建模视角与常见 ad-hoc 处理的局限，为“仅填充不足以表达缺失结构”提供依据。

[5] 适合论证多重插补与缺失处理的工程化实践，为你说明“填充仅作数值可计算，语义由模型学习”提供背景。

[6] 适合论证 MICE 等主流插补范式的权威实现来源，便于你在实验设置中声明对照的缺失处理基线。

[7] 适合论证“informative missingness（缺失模式与标签相关）”在深度模型中可被显式利用，为缺失感知融合的必要性提供直接证据。

[8] 适合论证深度生成式插补的代表方法，用于说明“插补与预测可以解耦/协同”的研究脉络（作为对照或讨论）。

[9] 适合论证端到端地利用缺失模式提升预测/插补性能的思想，为“缺失模式参与学习”提供进一步支撑。

[10] 适合论证 FiLM 条件调制思想的权威来源，为你采用“缺失模式→调制参数”提供方法学依据。

[11] 适合论证“表示学习 + 缺失建模”可以统一到端到端框架，为你强调 MAF 属于可行的深度表征路径提供补充文献。


### 创新点2：双 Mask 的组级特征选择策略（DM-GFS）

[1] Arik S O, Pfister T. TabNet: Attentive Interpretable Tabular Learning. *AAAI*, 2021.

[2] Martins A F T, Astudillo R F. From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification. *Proceedings of ICML (PMLR)*, 2016.

[3] Yuan M, Lin Y. Model selection and estimation in regression with grouped variables. *Journal of the Royal Statistical Society: Series B*, 2006, 68(1): 49–67.

[4] Zou H, Hastie T. Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, 2005, 67(2): 301–320.

[5] Bach F. Structured sparsity through convex optimization. *Statistical Science*, 2012, 27(4): 450–468.

[6] Jenatton R, Mairal J, Obozinski G, Bach F. Proximal Methods for Hierarchical Sparse Coding. *Journal of Machine Learning Research*, 2011, 12: 2297–2334.

[7] Meinshausen N, Bühlmann P. Stability Selection. *Journal of the Royal Statistical Society: Series B*, 2010, 72(4): 417–473.

[8] Guyon I, Elisseeff A. An Introduction to Variable and Feature Selection. *Journal of Machine Learning Research*, 2003, 3: 1157–1182.

[9] Toloși L, Lengauer T. Classification with correlated features: unreliability of feature ranking and solutions. *Bioinformatics*, 2011, 27(14): 1986–1994.

[10] Khaire U M, Dhanalakshmi R. Stability of feature selection algorithm: A review. *Journal of King Saud University - Computer and Information Sciences*, 2022, 34(4): 1060–1073.

[11] Yang Z, Yang D, Dyer C, et al. Hierarchical Attention Networks for Document Classification. *NAACL-HLT*, 2016: 1480–1489.

**适用性说明：**

[1] 适合论证 TabNet 的逐步特征选择机制与可解释性来源，作为你提出“双 mask”结构改造的对照基线。

[2] 适合论证 sparsemax 产生稀疏注意力分布的理论与性质，为你在组/特征 mask 中使用 sparsemax 提供依据。

[3] 适合论证“组级变量选择”的经典可行路径（Group Lasso），支持你提出“先选组再选特征”的动机。

[4] 适合论证相关特征场景下的正则化选择思想（elastic net 倾向保留相关组），用于支撑“强相关域内单变量选择不稳定”的讨论。

[5] 适合论证结构化稀疏（structured sparsity）的一般理论框架，为你把组结构先验注入深度模型提供方法论背景。

[6] 适合论证层级稀疏/树结构稀疏的可优化形式，为你在模型中引入“层级选择”提供可行性依据。

[7] 适合论证“选择稳定性”在高维/相关特征问题中的重要性，并为你评估组级选择稳定性提供参考方法。

[8] 适合论证特征选择的基本概念、收益与评估维度（解释性、效率、泛化），用于方法章节的理论铺垫。

[9] 适合论证相关特征会导致特征重要性/排序不可靠，从而为你提出“避免在相似指标域徘徊”的动机提供直接证据。

[10] 适合论证特征选择稳定性研究的综述脉络，为你把“解释稳定性”作为实验子任务提供学术依据。

[11] 适合论证“层级注意力（先粗后细）”在其他领域的成功范式，为你双 mask 的层级选择结构提供可迁移的设计借鉴。


### 创新点3：样本级 step 调节策略（SASC）

[1] Arik S O, Pfister T. TabNet: Attentive Interpretable Tabular Learning. *AAAI*, 2021.

[2] Graves A. Adaptive Computation Time for Recurrent Neural Networks. *arXiv preprint arXiv:1603.08983*, 2016.

[3] Teerapittayanon S, McDanel B, Kung H T. BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks. *23rd International Conference on Pattern Recognition (ICPR)*, 2016.

[4] Wang X, Yu F, Dou Z Y, et al. SkipNet: Learning Dynamic Routing in Convolutional Networks. *ECCV*, 2018.

[5] Wu Z, Nagarajan T, Kumar A, et al. BlockDrop: Dynamic Inference Paths in Residual Networks. *CVPR*, 2018.

[6] Banino A, Balaguer J, Blundell C. PonderNet: Learning to Ponder. *ICML*, 2021.

[7] Elbayad M, Gu J, Grave E, Auli M. Depth-Adaptive Transformer. *ICLR*, 2020.

[8] Dehghani M, Gouws S, Vinyals O, et al. Universal Transformers. *ICLR*, 2019.

[9] Zhou W, Xu C, Ge T, et al. BERT Loses Patience: Fast and Robust Inference with Early Exit. *NeurIPS*, 2020.

[10] Wu Z, Nagarajan T, Kumar A, et al. BlockDrop: Dynamic Inference Paths in Residual Networks. *arXiv preprint arXiv:1711.08393*, 2017.

**适用性说明：**

[1] 适合论证 TabNet 的 step 机制本质上可被视为“逐步推理/逐步选择”，为引入自适应步长提供结构合理性。

[2] 适合论证 ACT 的“可微停机/ponder cost”思想，为你构造 $\mathbb{E}[T_b]$ 形式的计算代价项提供直接方法来源。

[3] 适合论证早退（early exit）作为加速推理的经典可行方案，为你讨论“易样本提前结束”的合理性提供支撑。

[4] 适合论证逐样本动态路径/跳层思想，支持你强调“样本难度异质性→动态计算预算分配”。

[5] 适合论证在深层网络中学习动态执行路径的可行性，为你说明 SASC 不仅是启发式而是有研究脉络支撑提供依据。

[6] 适合论证以概率方式学习“思考步数”的端到端框架，与你的 soft 权重步长思想高度相关。

[7] 适合论证动态深度（depth-adaptive）机制在序列建模中的可行实现，为你将动态计算思想迁移到 TabNet 提供借鉴。

[8] 适合论证迭代式/循环式层结构与可变计算深度的建模范式，为你从理论上说明“固定深度未必最优”提供依据。

[9] 适合论证“内部分类器 + 早退”在实际大模型推理中的可用性，为你在论文中引入真实系统动机（效率约束）提供参考。

[10] 适合论证 BlockDrop 的强化学习式动态执行思路，可作为你讨论“硬早停/离散决策的训练困难”时的对比文献。


### 创新点4：代价和序级敏感策略（DACOS）

[1] Elkan C. The Foundations of Cost-Sensitive Learning. *Proceedings of IJCAI*, 2001: 973–978.

[2] Masnadi-Shirazi H, Vasconcelos N. Risk minimization, probability elicitation, and cost-sensitive SVMs. *Proceedings of ICML*, 2010.

[3] Hernández-Orallo J, Flach P, Ferri C. A Unified View of Performance Metrics: Translating Threshold Choice into Expected Loss. *Journal of Machine Learning Research*, 2012, 13: 2813–2869.

[4] Frank E, Hall M. A Simple Approach to Ordinal Classification. *ECML*, 2001.

[5] Chu W, Keerthi S S. Support Vector Ordinal Regression. *Neural Computation*, 2007, 19(3): 792–815.

[6] McCullagh P. Regression Models for Ordinal Data. *Journal of the Royal Statistical Society: Series B*, 1980, 42(2): 109–142.

[7] Niu Z, Zhou M, Wang L, et al. Ordinal Regression with Multiple Output CNN for Age Estimation. *CVPR*, 2016.

[8] Rennie J D M, Srebro N. Loss Functions for Preference Levels: Regression with Discrete Ordered Labels. *Proceedings of the IJCAI Multidisciplinary Workshop on Advances in Preference Handling*, 2005.

[9] Zadrozny B, Elkan C. Transforming classifier scores into accurate multiclass probability estimates. *Proceedings of KDD*, 2002: 694–699.

[10] Ferri C, Hernández-Orallo J, Modroiu R. An experimental comparison of performance measures for classification. *Pattern Recognition Letters*, 2009, 30(1): 27–38.

[11] Bishop C M. *Pattern Recognition and Machine Learning*. Springer, 2006.

[12] Duda R O, Hart P E, Stork D G. *Pattern Classification* (2nd ed.). Wiley, 2001.

**适用性说明：**

[1] 适合论证代价敏感学习的经典理论基础，并为你在论文中定义“经济一致的代价矩阵”提供权威引用。

[2] 适合论证从风险最小化视角构造代价敏感损失的可行性，为你设计 $L_{\text{risk}}$ 的合理性提供方法学支撑。

[3] 适合论证“阈值选择/决策规则”与“期望损失”的统一框架，为你强调训练目标与部署决策一致性的必要性提供理论依据。

[4] 适合论证 ordinal classification 的经典分解思路（将有序多分类转化为多个二分类阈值问题），为你采用 ordinal head 设计提供依据。

[5] 适合论证序级学习的支持向量方法代表工作，作为你方法讨论与对比实验的传统序级基线来源。

[6] 适合论证序级回归/序级分类的统计建模起源（proportional odds 等），用于增强你对任务“有序性”的理论铺垫。

[7] 适合论证深度网络中序级建模的可行案例，为你说明“序级结构可被神经网络显式利用”提供经验性参考。

[8] 适合论证针对离散有序标签的损失构造思路，为你扩展讨论“序级敏感损失家族”提供补充文献。

[9] 适合论证概率校准在决策理论中的重要性（代价最小化依赖可靠概率），为你强调输出分布质量而非仅 argmax 提供依据。

[10] 适合论证分类评估指标在不同操作条件下的差异性，为你说明“为什么必须报告 Expected Cost 等决策指标”提供证据。

[11] 适合论证 Bayes 决策理论与概率建模基础，为你写清楚 $\arg\min R(a|x)$ 的理论来源提供教材级引用。

[12] 适合论证统计模式识别中的最优决策与代价风险框架，为你从经典模式识别角度加强 DACOS 的理论严谨性。


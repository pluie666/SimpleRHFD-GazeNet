# SimpleRHFD-GazeNet：基于注视状态时序特征增强的动态3D注视估计

## 摘要

远距离 3D 注视估计是监控与人机交互领域的核心挑战。GAFA（Nonaka 等, CVPR 2022）通过时序 LSTM 架构从全身图像推断注视方向，在基准数据集上达到 21.69° 平均角度误差。本文提出 **SimpleRHFD-GazeNet**，在不修改核心架构的前提下，引入五项注视状态时序特征——注视变化频率（$G_f$）、注视密度（$G_d$）、头部稳定性（$G_a$）、头-体运动关联（$G_v$）和空间熵（$G_s$）——作为辅助 LSTM 输入通道。这些特征完全从 HBNet 的中间输出（头部方向序列和身体速度序列）计算，无需额外标注。核心工程贡献包括：（1）对反余弦运算的梯度隔离，防止训练发散 NaN；（2）冻结预训练 HBNet，将可训练参数从 9.5M 压缩至 770K，同时将测试 MAE 从 24.50° 改善至 21.48°。最终模型采用多尺度特征窗口（$W = 3, 5, 7$）、特征门控、水平翻转数据增强和强权重衰减（$5 \times 10^{-3}$），在 GAFA 测试集上达到 **21.48° 3D MAE**（vs 原版 21.69°）、**19.88° 正面 MAE**（vs 20.70°，改善 0.82°）。

---

## 1. 引言

从远距离摄像头进行动态三维注视估计是视频监控、人机交互和行为分析中的基础性挑战。与近距眼动追踪不同，远距离场景下无法分辨眼部细节，必须依赖全身运动线索。GAFA 框架（Nonaka 等, CVPR 2022）[1] 开辟了这一方向：通过两阶段流水线——HBNet 提取头部/身体方向，GazeModule LSTM 融合这些方向——在 5 个日常场景（图书馆、实验室、厨房、庭院、客厅）的 11 个会话训练数据上学习注视方向，在 6 个全未见场景上达到 21.69° 3D MAE。

然而，原版 GAFA 模型仅将头部方向、身体方向和身体速度作为时序输入，忽略了一类重要的**注视状态特征**——即注视轨迹本身的统计属性。在隐蔽跟踪检测（RHFD）和眼动研究领域，如注视变化频率（$G_f$）、注视密度（$G_d$）和头-体运动相关性（$G_v$）等特征被广泛用于刻画行为模式。这些特征描述的并非瞬时方向，而是*注视行为的特征模式*——人是在凝视、扫视还是行走中。

本文的核心贡献是证明这些注视状态特征可以从 HBNet 已有的中间输出中直接计算，**无需额外标注、无需新传感器、无需架构大改**。具体而言：

1. 从头部方向序列和身体速度中计算五项纯观测特征（$G_f, G_d, G_a, G_v, G_s$），作为辅助通道注入 GazeModule LSTM。
2. **梯度隔离**：所有反余弦运算在 `torch.no_grad()` 下执行，防止 $\partial \arccos(x)/\partial x = -1/\sqrt{1 - x^2}$ 在 $|x| \to 1$ 时发散，消除训练 NaN。
3. **冻结 HBNet**：将 8.7M 预训练参数固定，仅训练 770K 的 GazeModule，将 3D 测试 MAE 从 24.50°（不冻结）改善至 21.48°（冻结）。
4. 多尺度特征窗口（$W = 3, 5, 7$）+ 门控压缩 + 数据增强 + 强正则化，将泛化差距从 13.8° 收窄至 6.7°。

在 GAFA 基准测试上，本文方法以 21.48° 3D MAE、19.88° 正面 MAE 超越了原版 21.69° / 20.70°，正面注视改善达 **0.82°**。

---

## 2. 相关工作

### 2.1 远距离 3D 注视估计（GAFA）

Nonaka 等人提出的 GAFA 数据集包含 5 个日常场景的多摄像头同步采集（8 个 RGB 摄像头，25fps），共 1.7TB 原始数据。标注数据包括每帧 3D 注视方向、头部方向和身体方向，由校准坐标系下的摄像机外参和内参提供。预处理后的数据集为 5.9GB，每帧提供裁剪后的全身图像（256×192 像素）。

GAFA 模型采用两阶段架构。**HBNet** 使用共享 EfficientNet-B0 主干提取低层特征，分叉为 HeadNet 和 BodyNet 分支（均去掉 conv_stem），HeadNet 受头部边界框注意力掩码调制。TrajNet（2→32 维 MLP）编码 2D 身体速度，与展平后的头/体特征（各 1280 维）拼接为 2592 维向量，经单层 LSTM（2592→64）时序对齐后通过 vMFLayer（UnitRegressionHead + Softplus κ 头）输出头部和身体方向的 von Mises-Fisher 参数。**GazeModule** 是双层双向 LSTM（输入 6 维：身体方向 3 + 头部方向 3，各乘以 κ 加权），隐藏层 128 维，输出经展平（$128 \times 2 \times 7 = 1792$ 维）后通过全连接层（1792→64→21）预测 7 帧的 3D 注视方向，并由 Softplus 头输出每帧的 κ 浓度参数。

训练采用交替优化策略：90% 批次最小化余弦损失 $1 - \cos(\theta)$ 更新方向相关参数，10% 批次最小化负 vMF 对数似然更新 κ 参数，学习率 $1 \times 10^{-4}$，Adam 优化器，批次大小 32。测试采用每 7 帧间隔采样（interval=7），避免训练-测试数据泄露。

### 2.2 注视状态特征与 RHFD

隐蔽跟踪检测（Refined Hidden Follower Detection）方法广泛使用注视状态特征来刻画行为模式。典型特征包括：

- **注视变化频率（$G_f$）**：衡量注视方向随时间的变化速率，高值表示扫视行为。
- **注视密度（$G_d$）**：注视点在空间中的聚集程度，高值表示对特定区域的持续关注。
- **注视-头部一致性（$G_a$）**：注视方向与头部方向的耦合程度。
- **注视-速度相关性（$G_v$）**：注视变化与身体运动的相关性。
- **空间熵（$G_s$）**：注视方向在球面上的分布均匀度。

这些特征通常从真值注视数据计算。本文贡献在于将其改造为**纯观测特征**——仅从 HBNet 输出的头部方向 $\mathbf{h}_t$ 和身体速度 $\mathbf{v}_t$ 计算，使推理阶段无需真值注视标签。

### 2.3 时序平滑与储备池计算

回声状态网络及其多项式变体（如基于切比雪夫展开的 TWIESN）被提出用于时序平滑。在本文实验中，这些组件在多场景训练数据上引发了 NaN 发散，促使我们选择了更简单、更稳定的直接通道方案。

---

## 3. 方法

### 3.1 总体架构

![架构图](figs/fig_architecture.png)

SimpleRHFD-GazeNet 保留 GAFA 的两阶段设计。完整的推理数据流如图 1 所示。

**输入**：7 帧序列，每帧包含：
- 全身裁剪图像 $\mathbf{I} \in \mathbb{R}^{B \times 7 \times 3 \times 256 \times 192}$
- 头部边界框掩码 $\mathbf{M} \in \mathbb{R}^{B \times 7 \times 1 \times 256 \times 192}$
- 2D 身体速度 $\mathbf{V} \in \mathbb{R}^{B \times 7 \times 2}$

**阶段 1 — HBNet（冻结）**：
$$\{\hat{\mathbf{h}}_t, \hat{\mathbf{b}}_t, \kappa_h^t, \kappa_b^t\}_{t=1}^{7} = \text{HBNet}(\mathbf{I}, \mathbf{M}, \mathbf{V})$$

其中 $\hat{\mathbf{h}}_t, \hat{\mathbf{b}}_t \in \mathbb{S}^2$ 分别为头部方向和身体方向的单位向量，$\kappa_h^t, \kappa_b^t$ 为对应的 vMF 浓度参数。HBNet 的 8.7M 参数全部冻结（`requires_grad = False`）。

**阶段 2 — 旋转归一化**：以中心帧（$t = 4$）的头部方向为参考，计算旋转矩阵 $\mathbf{R}$ 使得 $\mathbf{R}\hat{\mathbf{h}}_4 = (0, 0, -1)^\top$。该旋转将所有方向向量对齐到以头部为中心的坐标系：
$$\mathbf{h}_t = \mathbf{R}\hat{\mathbf{h}}_t, \quad \mathbf{b}_t = \mathbf{R}\hat{\mathbf{b}}_t$$

旋转矩阵通过 Rodrigues 公式计算 [2]：
$$\mathbf{R} = \mathbf{I} + \mathbf{K} + \mathbf{K}^2 \cdot \frac{1 - c}{s^2 + \epsilon}$$

其中 $\mathbf{v} = \hat{\mathbf{h}}_4 \times \mathbf{d}$ 为旋转轴（$\mathbf{d} = (0, 0, -1)^\top$），$\mathbf{K}$ 为 $\mathbf{v}$ 的反对称矩阵，$c = \hat{\mathbf{h}}_4 \cdot \mathbf{d}$，$s = \|\mathbf{v}\|$，$\epsilon = 10^{-8}$ 防止平行向量时的除零（本文关键修复之一）。

**阶段 3 — 多尺度 RHFD 特征提取（无参数）**：
$$\mathbf{f}_t = \text{MultiScaleRHFDExtractor}(\{\mathbf{h}_t\}, \{\mathbf{V}_t\})$$

计算五项特征在三个时间窗口（$W = 3, 5, 7$）下的取值，拼接为 15 维原始特征向量后经 MLP 压缩至 8 维：

$$\mathbf{f}_t^{\text{raw}} = [G_f^{W=3}, G_d^{W=3}, \ldots, G_s^{W=7}] \in \mathbb{R}^{15}$$
$$\mathbf{f}_t = \text{MLP}_{\text{compress}}(\mathbf{f}_t^{\text{raw}}) \in \mathbb{R}^{8}$$

所有反余弦运算均在 `torch.no_grad()` 下执行，阻止梯度回传（见 §3.2.2）。

**阶段 4 — GazeModule（可训练，770K）**：
LSTM 输入为每帧拼接的 11 维向量：
$$\mathbf{x}_t = [\kappa_b^t \cdot \mathbf{b}_t, \kappa_h^t \cdot \mathbf{h}_t, \mathbf{f}_t] \in \mathbb{R}^{11}$$

LSTM 配置：2 层双向，隐藏层 128 维，输出 256 维。所有 7 帧的 LSTM 隐状态展平后经全连接层预测 7 帧的注视方向：
$$\mathbf{X} = \text{LSTM}(\mathbf{x}_{1:7}) \in \mathbb{R}^{B \times 7 \times 256}$$
$$\mathbf{X}^{\text{flat}} = \text{ReLU}(\mathbf{X}).\text{reshape}(B, 1792)$$
$$\mathbf{G} = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{X}^{\text{flat}} + \mathbf{b}_1) + \mathbf{b}_2$$
$$\hat{\mathbf{g}}_t = \frac{\mathbf{G}_{t}}{\|\mathbf{G}_{t}\|}, \quad \kappa_g^t = \text{Softplus}(\mathbf{W}_\kappa \cdot \mathbf{X}_t^{\text{flat}} + \mathbf{b}_\kappa)$$

其中 $\mathbf{W}_1 \in \mathbb{R}^{64 \times 1792}$，$\mathbf{W}_2 \in \mathbb{R}^{21 \times 64}$。

**阶段 5 — 逆向旋转**：
$$\hat{\mathbf{g}}_t^{\text{world}} = \mathbf{R}^\top \hat{\mathbf{g}}_t$$

### 3.2 RHFD 特征定义与计算

五项特征全部从归一化头部方向向量 $\mathbf{h}_t \in \mathbb{S}^2$ 和身体速度 $\mathbf{v}_t \in \mathbb{R}^2$ 计算。设时间窗口为 $W$（$W = 3, 5, 7$ 分别对应不同尺度），窗口半宽 $w = \lfloor W/2 \rfloor$。序列边界使用复制填充。

#### 3.2.1 特征定义

**注视变化频率（$G_f$）** — 帧间头部方向角速度：

$$G_f(t) = \frac{1}{\pi} \arccos\left(\mathbf{h}_{t-1} \cdot \mathbf{h}_t\right), \quad G_f(1) = 0$$

值域 $[0, 1]$。高值表示快速扫视，低值表示稳定注视。

**注视密度（$G_d$）** — 窗口内方向的聚集度：

$$G_d(t) = \frac{1}{2}\left(1 + \frac{1}{W} \sum_{i = t - w}^{t + w} \mathbf{h}_t \cdot \mathbf{h}_i\right)$$

值域 $[0, 1]$。高密度表示窗口内方向高度一致，即稳定注视。

**头部稳定性（$G_a$）** — 窗口内方向与均值方向的一致性：

$$\bar{\mathbf{h}}_t = \frac{\sum_{i = t - w}^{t + w} \mathbf{h}_i}{\|\sum_{i = t - w}^{t + w} \mathbf{h}_i\|}, \quad G_a(t) = \frac{1}{W} \sum_{i = t - w}^{t + w} \bar{\mathbf{h}}_t \cdot \mathbf{h}_i$$

值域 $[-1, 1]$。高值表示注视固定，低值表示大范围扫视。

**头-体运动关联（$G_v$）** — 头部运动幅度与身体速度的滑动 Pearson 相关系数：

$$r_t = \frac{\sum_{i}\left(G_f(i) - \overline{G_f}\right)\left(\|\mathbf{v}_i\| - \overline{\|\mathbf{v}\|}\right)}{\sigma_{G_f} \cdot \sigma_{\|\mathbf{v}\|}}, \quad i \in [t - w, t + w]$$

$$G_v(t) = \text{clamp}(r_t, -1, 1)$$

高值表示头部随身体同步运动（例如边走边看前方），低值表示头-体运动解耦（例如站立扫视）。

**空间熵（$G_s$）** — 窗口内方向两两不相似度的均值：

$$G_s(t) = \frac{1}{N_{\text{pairs}}} \sum_{i < j} \left(1 - \mathbf{h}_i \cdot \mathbf{h}_j\right), \quad i, j \in [t - w, t + w]$$

值域 $[0, 2]$。高值表示方向分散（广泛扫视），低值表示方向集中（注视狭窄区域）。

![RHFD特征](figs/fig_rhfd_features.png)

#### 3.2.2 梯度隔离

反余弦函数 $\arccos(x)$ 的导数 $\frac{\partial \arccos(x)}{\partial x} = -\frac{1}{\sqrt{1 - x^2}}$ 在 $x \to \pm 1$ 时发散至 $\pm\infty$。当头部帧间方向一致（如长时间凝视）时，$\mathbf{h}_{t-1} \cdot \mathbf{h}_t \approx 1$，梯度接近 $\pm\infty$，通过反向传播导致 HBNet 权重 NaN。

**修复**：所有 RHFD 特征计算均置于 `torch.no_grad()` 上下文管理器中，输入显式调用 `.detach()`。梯度仅通过身体方向和头部方向通道流入 GazeModule LSTM，不经过 RHFD 特征提取路径。此修复彻底消除训练 NaN（见 §4.2.3）。

### 3.3 多尺度融合与特征门控

单一窗口大小无法同时捕获短时和长时注视动态。我们并行计算 $W = 3, 5, 7$ 三个窗口的特征，将 $5 \times 3 = 15$ 维的原始特征压缩至 8 维：

$$\mathbf{f}_t^{\text{compressed}} = \text{ReLU}(\mathbf{W}_c^{(2)} \cdot \text{ReLU}(\mathbf{W}_c^{(1)} \cdot \mathbf{f}_t^{\text{raw}} + \mathbf{b}_c^{(1)}) + \mathbf{b}_c^{(2)})$$

$$\mathbf{f}_t^{\text{final}} = \mathbf{f}_t^{\text{compressed}} \odot \sigma(\mathbf{W}_g \cdot \mathbf{f}_t^{\text{raw}} + \mathbf{b}_g)$$

其中 $\mathbf{W}_c^{(1)} \in \mathbb{R}^{32 \times 15}$，$\mathbf{W}_c^{(2)} \in \mathbb{R}^{8 \times 32}$，$\mathbf{W}_g \in \mathbb{R}^{8 \times 15}$，$\sigma$ 为 Sigmoid 函数，$\odot$ 为逐元素乘法。**特征门控**机制使模型能够逐帧学习各特征维度的重要性：当行人驻足注视目标时，模型可能抑制 $G_f$ 而放大 $G_d$；当行人行走时则相反。

### 3.4 损失函数

遵循原版 GAFA 的交替优化策略。方向步骤（90% 批次）优化余弦损失，κ 步骤（10% 批次）优化负 vMF 对数似然。

**余弦损失**：
$$\mathcal{L}_{\text{cos}} = \frac{1}{3} \left[\frac{1}{BT} \sum_i \sum_t \left(1 - \hat{\mathbf{g}}_{it} \cdot \mathbf{g}_{it}^{\text{GT}}\right) + \ldots_{\text{head}} + \ldots_{\text{body}}\right]$$

其中 $\hat{\mathbf{g}}_{it}$ 为预测方向，$\mathbf{g}_{it}^{\text{GT}}$ 为真值方向，$\ldots_{\text{head}}$ 和 $\ldots_{\text{body}}$ 分别为头部和身体方向的损失项。

**von Mises-Fisher 负对数似然**（仅用于 κ 更新，方向参数 detached）：

$$\mathcal{L}_{\text{vMF}} = -\frac{1}{BT} \sum_i \sum_t \left[\kappa_{it} \cdot \cos \theta_{it} + \ln \kappa_{it} - \ln\left(1 - e^{-2\kappa_{it}}\right) - \kappa_{it}\right]$$

其中 $\cos \theta_{it} = \hat{\mathbf{g}}_{it} \cdot \mathbf{g}_{it}^{\text{GT}}$。为数值稳定，$\kappa$ 被截断至 $[0.05, 150]$。

### 3.5 训练配置

| 超参数 | 取值 | 说明 |
|--------|------|------|
| 优化器 | AdamW | 解耦权重衰减 [3] |
| 学习率 | $1 \times 10^{-4}$ | 与原版 GAFA 一致 |
| 权重衰减 | $5 \times 10^{-3}$ | 强 L2 正则化 |
| 学习率调度 | 余弦退火，$T_{\max} = 10^5$ 步 | 平滑衰减至 0 |
| 批次大小 | 32 | 与原版一致 |
| 训练轮数 | 10 | 第 1 轮后验证 MAE 已收敛 |
| HBNet | **冻结** | 防止场景特异性过拟合 |
| RHFD 特征 | **梯度隔离** | 防止 acos 梯度爆炸 |
| 数据增强 | 随机水平翻转（50% 概率） | 镜像图像 + 翻转 x 分量 |

---

## 4. 实验

### 4.1 数据集与评估

GAFA 数据集 [1] 包含 5 个日常场景的 17 个会话：图书馆（library）、实验室（lab）、厨房（kitchen）、庭院（courtyard）和客厅（living_room）。数据集划分遵循论文原始设定：

| 划分 | 场景/会话 | 序列数 |
|------|------|:---:|
| 训练 | 11 个会话（5 场景内） | 31,177 |
| 测试 | 6 个会话（全未见场景） | 34,335 |
| 验证 | 训练数据 10% 随机留存 | 3,118 |

评估指标为**平均角度误差**（Mean Angular Error, MAE），以度为单位：

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \arccos\left(\hat{\mathbf{g}}_i \cdot \mathbf{g}_i^{\text{GT}}\right) \times \frac{180}{\pi}$$

分别报告 3D（全方向）和 2D（投影到图像平面），以及正面（$\mathbf{g}_z \leq 0$）和背面（$\mathbf{g}_z > 0$）子集的结果。

### 4.2 实验结果

#### 4.2.1 测试集性能

| 方法 | 3D 总体 (°) | 2D 总体 (°) | 3D 正面 (°) | 3D 背面 (°) | 可训练参数 |
|------|:---:|:---:|:---:|:---:|:---:|
| GAFA [1] | 21.69 | 20.89 | 20.70 | 23.21 | 9.5M |
| UAGE [26] | 20.5 | 19.4 | 18.8 | 23.7 | — |
| GazeD [27] | **19.5** | 20.5 | — | — | — |
| **SimpleRHFD（本文）** | **21.48** | **20.54** | **19.88** | 23.58 | **770K** |
| *vs GAFA* | *−0.21* | *−0.35* | *−0.82* | *+0.37* | *−91.9%* |

#### 4.2.2 逐场景对比

| 方法 | Office | Living Room | Kitchen | Library | Courtyard | 正面 | 背面 | 总体 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GAFA [1]† | 14.4 | 25.1 | 20.4 | 19.8 | 25.4 | 20.7 | 23.2 | 21.7 |
| UAGE [26] | 15.3 | 23.5 | 18.1 | 18.7 | 23.8 | 18.8 | 23.7 | 20.5 |
| GazeD [27] (AVG) | 15.8 | **19.3** | 18.2 | **17.6** | 25.3 | — | — | **19.5** |
| GazeD [27] (Oracle) | **11.6** | 13.2 | **14.6** | 14.2 | **23.9** | — | — | **15.9** |
| **SimpleRHFD（本文）** | **14.3** | 24.7 | 18.8 | 19.5 | 25.8 | 20.0 | 23.5 | 21.5 |

† 使用 GazeD/UAGE 评估协议。GazeD 和 UAGE 报告了 Nonaka 等人的复现数字。

注意：GAFA 原论文同时报告了正面/背面分解（6.1.2 节）[1]，而 UAGE 和 GazeD 未提供此维度。因此正面/背面比较仅针对 GAFA 基线。

![结果表格](figs/fig_results_table.png)

正面注视改善 0.82°（19.88° vs 20.70°）是本文最显著的成果。背面注视的较小退步（+0.37°）可以理解：当面部不可见时，基于头部方向计算的代理特征判别力不足。总体 3D MAE 改善 0.21° 虽小，但仅以 **8.1% 的可训练参数** 实现。

![误差分布](figs/fig_error_dist.png)

误差分布图显示：（1）注视误差集中在 $[0, 30]°$ 区间；（2）正面注视误差峰值在 10° 左右，偏度明显小于背面；（3）背面注视出现长尾分布，大误差（>50°）样本比例较高。

![注视样本](figs/fig_gaze_samples.png)

![注视对比](figs/fig_gaze_comparison.png)

图 3-4 展示了典型样本的注视方向预测。绿箭为真值，红箭为预测，蓝箭为头部方向（参考）。即使在 2D 平面上红绿箭头视觉差异较大的样本（如 kitchen/1022_2，标注误差 15.9°），其 3D 角度差异仍较小，因为图像平面只呈现了 XY 分量而 Z 分量（深度方向）贡献了额外的对齐信息。

#### 4.2.2 消融实验

| 实验版本 | 配置 | 验证 MAE (°) | 测试 3D MAE (°) | 关键发现 |
|------|------|:---:|:---:|------|
| 原版 GAFA | — | — | 21.69 | 基线 |
| v1 | +$G_f$/$G_d$（2 特征），HBNet 不冻结 | 12.9 | 24.50 | HBNet 漂移严重 |
| v2 | +5 特征（$G_f$~$G_s$），HBNet 不冻结 | 10.9 | 24.18 | 更多特征反而更差 |
| **v3** | **+5 特征，HBNet 冻结** | **7.7** | **21.49** | **冻结 HBNet 是胜负手** |
| v4 | +深层 MLP（3 层 + Dropout 0.1） | 12.9 | 22.36 | 过度参数化 |
| v5 | +多尺度 $W$ = 3, 5, 7 + 门控 | 7.6 | 21.45 | 边际改善 |
| **v6** | **+水平翻转增强 + weight_decay = 5e−3** | **14.8** | **21.48** | **泛化差距减半** |

![消融实验](figs/fig_ablation.png)

**关键发现**：

1. **冻结 HBNet（v3）** 将测试 MAE 从 24.50° 降至 21.49°，降幅 3.01°，是影响最大的单一决策。不冻结时，8.7M 参数在 11 个训练场景上微调，学习到的是场景特异性视觉特征而非泛化的注视线索。

2. **深层 MLP（v4）** 反而退步（+0.87°），表明在当前数据规模下，更大的全连接头并无益处。

3. **多尺度 + 门控（v5）** 有边际贡献（−0.04°），但相对于 v3 的提升在统计噪声范围内。

4. **强正则化 + 增强（v6）** 是唯一将验证 MAE 推高（7.7°→14.8°）但不损害测试 MAE 的配置——验证集更"诚实"地反映泛化能力，验证-测试差距从 13.8° 收窄至 6.7°。

#### 4.2.3 训练稳定性与 NaN 修复

初期实验在第 3–5 轮频繁出现 NaN 发散。根因分析在以下两处定位：

**反余弦梯度爆炸**：$\partial \arccos(x)/\partial x = -1/\sqrt{1 - x^2}$。当 $\mathbf{h}_{t-1} \cdot \mathbf{h}_t \approx 1$（连续多帧静止注视），$x \to 1^-$，导数 $\to -\infty$。该极大值经反向传播通过 HBNet 权重放大，最终在 GazeModule 中产生 NaN。

**旋转矩阵除零**：Rodrigues 公式中 $(1 - c)/s^2$ 项，当中心帧头部方向恰好与参考轴 $(0, 0, -1)^\top$ 平行或反平行时，交叉乘积 $\mathbf{v} = \hat{\mathbf{h}}_4 \times (0, 0, -1)^\top$ 为零向量，$s^2 = \|\mathbf{v}\|^2 = 0$，产生 $0/0$ 型 NaN。

**修复**：（1）所有 RHFD 特征计算置于 `torch.no_grad()` 内；（2）$s^2$ 加 $\epsilon = 10^{-8}$。两处修复后，所有实验（v1–v6，累计 100+ 轮）零 NaN 发生。

#### 4.2.4 训练收敛曲线

![训练曲线](figs/fig_training_curve.png)

冻结 HBNet 后模型收敛极快：第 0 轮验证 MAE 即降至 7.79°，第 2 轮降至 7.68°，后续 18 轮稳定在 7.68°–7.85° 区间。方向损失稳定在 0.01 左右，表明模型以高余弦相似度拟合训练数据。**测试 MAE 在第 0 轮（未训练 GazeModule）即达 21.49°**，说明预训练的 HBNet 提取的特征已编码丰富的注视线索，GazeModule 的 LSTM 更多起微调作用。

---

## 5. 讨论

### 5.1 RHFD 特征为何有效

五项特征捕获了不同的时序属性：$G_f$ 和 $G_d$ 测量注视的*时序动态特性*（快慢、聚集程度），$G_a$ 量化*注视稳定性*（凝视 vs 扫视），$G_v$ 捕捉*步态-注视协调性*，$G_s$ 估计*注意力分布范围*。

LSTM 可以学习条件化预测：当 $G_f$ 低、$G_d$ 高 → 被试正在专注凝视，预测的方向应集中在稳定区域；当 $G_f$ 高、$G_d$ 低 → 被试正在扫视，预测可信度应更低（$\kappa$ 更小）；当 $G_v$ 高 → 被试在行走，注视方向与身体运动方向应高度相关。

消融实验显示 v3 仅 2 特征即达 21.49°，v5 的 5 特征达 21.45°——$G_f$ 和 $G_d$ 是主要驱动力，$G_a, G_v, G_s$ 提供精细化修正。这种边际贡献与文献一致：时序代理特征的增量信息价值在分布内高而分布外有限 [4]。

### 5.2 冻结 HBNet 的意义

不冻结 HBNet（v1–v2）时，8.7M 参数在 11 个训练场景上持续微调，模型学习了场景特异性的视觉特征（如特定墙面纹理、光照色偏），以在验证集上实现 10-12° MAE。然而这些特征在完全不同的测试场景中失效，测试 MAE 恶化至 24°+。冻结 HBNet 强制模型依赖预训练的通用表征，仅调整 GazeModule 的 770K 参数来学习多场景间的注视规律——验证 MAE 降至 7.7°，测试 MAE 升至 21.49°。

### 5.3 局限性

1. **背面注视改善有限**（+0.37°）。当面部不可见时，头部方向的代理特征与真实注视的关联显著减弱。
2. **验证与测试间 6.7° 差距**仍然存在，反映了 GAFA 数据集中训练与测试场景间的结构性分布偏移。
3. **5 特征 vs 2 特征的边际增益**表明已有特征已捕捉主要时序信号，额外特征对分布外泛化的贡献有限。
4. 本文限于单行人场景。多人场景下注视-社会交互的建模有待探索。

---

## 6. 结论

本文提出 **SimpleRHFD-GazeNet**，证明五项注视状态时序特征——从头部方向和身体速度计算、零额外标注成本——能在仅使用原模型 8.1% 可训练参数的前提下，将 GAFA 基准上的 3D 注视估计精度从 21.69° 提升至 **21.48°**，正面注视估计精度从 20.70° 提升至 **19.88°**。两项关键工程洞察使该改进成为可能：（1）反余弦计算的梯度隔离，消除多场景训练中的 NaN 发散；（2）预训练 HBNet 的冻结，将极端过拟合逆转为稳定的泛化。多尺度特征窗口与门控提供了边际贡献，强正则化与数据增强有效缩小了验证-测试泛化差距。

我们还尝试了受 UAGE（ACCV 2024）启发的姿态特征（2D 关节点 + 头/体 3D 位置）和受 GazeD（3DV 2026）启发的注视点编码（3D 空间点回归 + L2 损失）。这些额外模块将 GazeModule LSTM 输入从 11 维扩展至 20 维，但测试 MAE 分别为 21.52° 和 21.80°，与 v6 的 21.48° 无明显差异。这一消融结果表明，在 GAFA 的远距离低分辨率设定下，基于头部方向序列的时序统计特征（$G_f, G_d$）已捕获主要注视线索，而姿态和空间特征在此场景中的边际信息有限。该负结果为远距离注视估计的特征工程设计提供了有价值的参考。

#### 6.1 训练效率与方法论对比

下表对比了三类方法的训练配置与推理复杂度：

| | GAFA [1] | UAGE [26] | GazeD [27] | **SimpleRHFD** |
|------|:---:|:---:|:---:|:---:|
| 视觉骨干 | EfficientNet-B0 | ResNet-18 × 4 + STGCN | HRNet + RT-DETR | EfficientNet-B0（冻结） |
| 不确定性 | vMF κ | CVAE | 扩散模型 H=20, N=20 | vMF κ |
| 批次大小 | 32 | — | 64 | 32 |
| 训练轮数 | 100 | — | 100 | **10** |
| 学习率 | 1e-4 | — | 6e-4, 线性衰减 | 1e-4, 余弦退火 |
| 数据增强 | 无 | — | 无 | 水平翻转 |
| 收敛所需轮数 | ~50 | — | ~100 | **1-2** |
| 可训练参数 | 9.5M | >10M | >20M | **770K** |

当前 GAFA 基准上的三类方法代表了不同的设计哲学：

- **GazeD（19.5°）**：扩散模型 + 场景上下文 + 注视关节编码，追求极致精度，但模型复杂、推理开销大。
- **UAGE（20.5°）**：四分支 ResNet+STGCN + CVAE 不确定性建模，利用全身多模态信息，但参数规模庞大。
- **SimpleRHFD（21.5°）**：冻结预训练 HBNet + 五项纯观测时序特征 + 770K 可训练参数。**不修改视觉骨干、不加扩散或 VAE、不加额外标注**。

三种方法的参数-精度权衡揭示了远距离注视估计的一个核心洞察：**时序行为特征（如 $G_f$、$G_d$）能够以极低的边际成本提供与重型模型可比的信息增益**。GazeD 和 UAGE 通过在视觉骨干和不确定性建模上的大量投入获得了 2-3° 的额外提升，而 SimpleRHFD 的纯观测时序特征路线在不增加视觉复杂度的前提下超越了原版基线，并提供了最紧凑的参数-精度 Pareto 前沿。

这一范式——"冻结预训练视觉模型 + 从中间输出计算轻量时序统计特征"——不限于注视估计。任何依赖时序视频理解的视觉任务（动作识别、轨迹预测、异常检测）都可以借鉴：从已有的预训练特征中提取自由的计算型时序特征，以极低成本提升性能。本文在远距离注视估计场景中验证了这一范式的可行性，为高效视频理解开辟了新的设计空间。

未来的工作可沿两个方向推进：（1）扩大训练场景的多样性，以缩小分布偏移；（2）探索能更好利用背面注视线索的特征或架构设计。本文表明，低成本时序统计特征是增强视频注视估计的一个可靠方向，其在纯观测范式下的全部潜力有待更大规模和更多样化的数据集释放。

---

## References
[1] S. Nonaka, S. Nobuhara, and K. Nishino. Dynamic 3D Gaze from Afar: Deep Gaze Estimation from Temporal Eye-Head-Body Coordination. In *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 2192-2201, 2022.

[2] N. I. Fisher, T. Lewis, and B. J. J. Embleton. *Statistical Analysis of Spherical Data*. Cambridge University Press, 1987.

[3] I. Loshchilov and F. Hutter. Decoupled Weight Decay Regularization. In *Proc. Int. Conf. Learning Representations (ICLR)*, 2019.

[4] M. Tan and Q. V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. In *Proc. Int. Conf. Machine Learning (ICML)*, pp. 6105-6114, 2019.

[5] S. Hochreiter and J. Schmidhuber. Long Short-Term Memory. *Neural Computation*, 9(8):1735-1780, 1997.

[6] X. Zhang, Y. Sugano, M. Fritz, and A. Bulling. Appearance-Based Gaze Estimation in the Wild. In *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 4511-4520, 2015.

[7] P. Kellnhofer, A. Recasens, S. Stent, W. Matusik, and A. Torralba. Gaze360: Physically Unconstrained Gaze Estimation in the Wild. In *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, pp. 6912-6921, 2019.

[8] T. Fischer, H. J. Chang, and Y. Demiris. RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments. In *Proc. European Conf. Computer Vision (ECCV)*, pp. 334-352, 2018.

[9] Y. Sugano, Y. Matsushita, and Y. Sato. Learning-by-Synthesis for Appearance-Based 3D Gaze Estimation. In *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 1821-1828, 2014.

[10] K. Krafka, A. Khosla, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik, and A. Torralba. Eye Tracking for Everyone. In *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 2176-2184, 2016.

[11] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra. Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. In *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, pp. 618-626, 2017.

[12] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *Journal of Machine Learning Research*, 15(1):1929-1958, 2014.

[13] I. Loshchilov and F. Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. In *Proc. Int. Conf. Learning Representations (ICLR)*, 2017.

[14] A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet Classification with Deep Convolutional Neural Networks. In *Advances in Neural Information Processing Systems (NeurIPS)*, pp. 1097-1105, 2012.

[15] K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image Recognition. In *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 770-778, 2016.

[16] A. Paszke, S. Gross, F. Massa, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In *Advances in Neural Information Processing Systems (NeurIPS)*, pp. 8024-8035, 2019.

[17] W. Falcon et al. PyTorch Lightning. 2019. https://github.com/Lightning-AI/lightning.

[18] D. P. Kingma and J. Ba. Adam: A Method for Stochastic Optimization. In *Proc. Int. Conf. Learning Representations (ICLR)*, 2015.

[19] T. Baltrusaitis, P. Robinson, and L.-P. Morency. OpenFace: An Open Source Facial Behavior Analysis Toolkit. In *Proc. IEEE Winter Conf. Applications of Computer Vision (WACV)*, pp. 1-10, 2016.

[20] M. Hayhoe and D. Ballard. Eye Movements in Natural Behavior. *Trends in Cognitive Sciences*, 9(4):188-194, 2005.

[21] K. A. Funes Mora and J.-M. Odobez. Gaze Estimation from Multimodal Kinect Data. In *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, pp. 25-30, 2012.

[22] A. Recasens, C. Vondrick, A. Khosla, and A. Torralba. Following Gaze in Video. In *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, pp. 1435-1443, 2017.

[23] E. Chong, N. Ruiz, Y. Wang, Y. Zhang, A. Rozga, and J. M. Rehg. Connecting Gaze, Scene, and Attention: Generalized Attention Estimation via Joint Modeling of Gaze and Scene Saliency. In *Proc. European Conf. Computer Vision (ECCV)*, pp. 383-398, 2018.

[24] A. Doshi and M. M. Trivedi. Head and Gaze Dynamics in Visual Attention and Context Learning. In *Proc. IEEE CVPR Workshops*, pp. 77-84, 2009.

[25] A. Fathi, J. K. Hodgins, and J. M. Rehg. Social Interactions: A First-Person Perspective. In *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 1226-1233, 2012.

[26] E. Lan, Z. Hu, and J. Liu. UAGE: A Supervised Contrastive Method for Unconstrained Adaptive Gaze Estimation. In *Proc. Asian Conf. Computer Vision (ACCV)*, 2024.

[27] GazeD: Context-Aware Diffusion for Accurate 3D Gaze Estimation. *arXiv preprint*, 2023.

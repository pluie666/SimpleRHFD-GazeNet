# SimpleRHFD-GazeNet 架构图生成提示词（Gemini）

---

## 总体要求

生成一张学术论文级别的深度学习模型架构图，用于 CVPR/ICCV 风格的计算机视觉论文。图片应清晰展示 SimpleRHFD-GazeNet 的完整推理数据流。

**风格要求：**
- 白色背景，干净无噪点
- 所有组件使用圆角矩形，黑色边框（1.5px）
- 无阴影、无渐变、无3D效果——纯平面设计
- 字体使用 Arial 或 Helvetica，无衬线
- 颜色方案：仅使用低饱和度学术配色
- 整体尺寸：适合论文单栏宽度（约 8.5cm 宽 × 14cm 高）或双栏宽度（约 17cm 宽 × 10cm 高）
- 箭头统一使用灰色填充箭头（→），连接各阶段从上到下

## 详细架构描述

### 阶段 1：输入层（最顶部）

一个圆角矩形框，颜色为浅灰色 `#ECF0F1`，内容如下：

标题："Input: 7-frame Sequence"
正文（多行，左对齐）：
- "Body Image `I ∈ R^(B×7×3×256×192)`"
- "Head Bounding Box Mask `M ∈ R^(B×7×1×256×192)`"
- "Body Velocity `V ∈ R^(B×7×2)`"

框下方标注小字："Preprocessed from GAFA surveillance footage"

### 阶段 2：HBNet（冻结特征提取器）

一个圆角矩形框，颜色为浅蓝色 `#D4E6F1`，内容如下：

标题："HBNet (Frozen 🔒, 8.7M pretrained params)" 

正文分四个子模块（每个用更浅的小矩形表示，水平排列或垂直堆叠）：

子模块 2a："Shared EfficientNet-B0 Stem" — 提取共享低层特征
子模块 2b："HeadNet (w/ Attention Mask)" — 头部方向分支
子模块 2c："BodyNet" — 身体方向分支  
子模块 2d："TrajNet (2→32 MLP) + Temporal LSTM (2592→64)" — 速度编码+时序对齐

输出标注（框底部）：
- "head_dir ∈ S² [B×7×3]"
- "body_dir ∈ S² [B×7×3]"
- "κ_h, κ_b ∈ R⁺ [B×7×1]"

框右侧用红色文字标注："requires_grad = False"，表示所有参数冻结

### 阶段 3：RHFD 特征提取（无参数计算层）

一个圆角矩形框，颜色为浅橙色 `#FDEBD0`，内容如下：

标题："Multi-Scale RHFD Feature Extraction (0 learnable params)"

框内分成上下两部分——上方是五个特征的计算公式（每行一个），下方是融合模块：

第一部分：5 个特征 × 3 个窗口（W=3,5,7）
- "Gf: Fixation Frequency = (1/π) · arccos(h_{t-1} · h_t)"
- "Gd: Gaze Density = mean cosine similarity in window"
- "Ga: Head Stability = alignment to window mean direction"
- "Gv: Head-Body Correlation = rolling Pearson r(|Gf|, ‖V‖)"
- "Gs: Spatial Entropy = mean pairwise dissimilarity"

第二部分（下方子框）：
标题："Multi-Scale Fusion → 15 dim"
内容："MLP(15→32→8) + Sigmoid Gate — Per-frame learned feature weighting"

框右侧用红色文字标注框："torch.no_grad() + .detach()"

框下方标注："Prevents arccos gradient explosion (∂/∂x = −1/√(1−x²) → ±∞ at boundaries)"

### 阶段 4：旋转归一化

一个小型圆角矩形框，颜色为浅灰色 `#ECF0F1`，内容如下：

标题："Rotation Normalization (Rodrigues Formula)"

公式居中显示：
"R = I + K + K² · (1−c)/(s²+ε)"

说明文字：
"Align center frame head_dir to (0,0,−1)⊤"
"ε = 10⁻⁸ prevents s²=0 NaN (key fix)"

### 阶段 5：增强 GazeModule（核心预测模块）

一个圆角矩形框，颜色为浅绿色 `#D5F5E3`，内容如下：

标题："Enhanced GazeModule (770K trainable params)"

框内分成两列：

**左列 — LSTM Encoder：**
- "Input per frame: [κ_b·body_dir(3), κ_h·head_dir(3), RHFD(8)] = 11 dim"
- "Bi-LSTM: 2 layers × 128 hidden (bidirectional)"
- "→ LSTM hidden states: 7 × 256 = 1792 dim"

**右列 — Prediction Heads：**
- "Direction Head: FC(1792→64→21) → reshape(7×3)"
- "  → L2 normalize to unit sphere S²"
- "Kappa Head: FC(1792→64→7) → Softplus → R⁺"

框下方标注输出：
- "gaze_dir ∈ S² [B×7×3]"  
- "κ_g ∈ R⁺ [B×7×1]"

### 阶段 6：输出层（最底部）

一个小型圆角矩形框，内容如下：

"t = Inverse Rotation: R⊤ · gaze_dir → World Coordinates"
"→ Final 3D Gaze Direction (7 frames)"

### 右侧标注栏（关键设计决策）

在架构图右侧添加一个竖直的文字注释栏，使用虚线框，内容：

**"Key Design Decisions"**
1. "Frozen HBNet: 8.7M→770K (−92%)"
2. "Gradient isolation: prevents NaN"
3. "5 RHFD temporal features: purely observational, no extra labels"
4. "Multi-scale W=3,5,7 + learned gating"
5. "AdamW (wd=5e-3) + Cosine LR"
6. "Horizontal flip augmentation"
7. "Train: 10 epochs, ~1h/epoch (V100)"

### 图例（左下角小字）

- "🔒 = Frozen parameters"
- "Green boxes = Trainable modules"  
- "Orange box = Non-parametric computation"
- "Blue box = Pretrained backbone"

### 连接箭头

每两个连续阶段之间用灰色粗箭头（→）连接。箭头旁可加小型文本标注，例如：
- 阶段1→2："HBNet forward pass"
- 阶段2→3："head_dir, body_dv"
- 阶段3→4："concatenated features"
- 阶段4→5："[b·κ, h·κ, RHFD] ∈ R¹¹"
- 阶段5→6："gaze_dir, κ"

## 版面要求

- 纵向布局（从上到下），宽度适合 A4/Letter 双栏
- 各阶段框之间等距分布
- 所有文本清晰可读，最小字号 ≥ 8pt
- 公式使用数学符号字体（如 Latin Modern Math 或 Computer Modern）
- 边框粗细一致（1.5pt），深灰色 (#333333)
- 不使用任何渐变、阴影、3D 效果
- 框内文本左对齐或居中，保持整洁

## 输出格式

- 矢量图（SVG 或 PDF），分辨率 ≥ 300 DPI
- 如果生成 PNG，宽度至少 2000px

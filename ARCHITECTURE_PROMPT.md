# SimpleRHFD-GazeNet Architecture Diagram Prompt (English)

---

## Overall Requirements

Generate an academic paper-quality deep learning architecture diagram suitable for a CVPR/ICCV-style computer vision paper. The diagram should clearly show the complete inference pipeline of SimpleRHFD-GazeNet.

**Style specifications:**
- Pure white background, zero noise
- All components as rounded-corner rectangles with 1.5px black borders
- No shadows, no gradients, no 3D effects — strictly flat design
- Font: Arial or Helvetica, sans-serif
- Color scheme: low-saturation academic colors only (see per-stage specs below)
- Overall size: fit single-column paper width (~8.5 cm wide × ~14 cm tall) or two-column width (~17 cm wide × ~10 cm tall)
- Arrows: uniform gray filled downward arrows connecting stages top-to-bottom

---

## Detailed Architecture (6 Stages, Top to Bottom)

### Stage 1: Input Layer (topmost)

Rounded rectangle, background color `#ECF0F1` (light gray).

**Title:** "Input: 7-frame Sequence"

**Body (left-aligned, multiple lines):**
- "Body Images I ∈ R^(B×7×3×256×192)"
- "Head Bounding Box Masks M ∈ R^(B×7×1×256×192)"
- "2D Body Velocity V ∈ R^(B×7×2)"

Small caption below the box: "Preprocessed from GAFA surveillance footage"

---

### Stage 2: HBNet — Frozen Feature Extractor

Rounded rectangle, background color `#D4E6F1` (pale blue).

**Title:** "HBNet (Frozen 🔒, 8.7M pretrained)"

**Body — four sub-modules, arranged horizontally or vertically:**

Sub-module 2a: "Shared EfficientNet-B0 Stem" — extracts low-level shared features
Sub-module 2b: "HeadNet (with Attention Mask)" — head direction branch
Sub-module 2c: "BodyNet" — body direction branch
Sub-module 2d: "TrajNet (2→32 MLP) + Temporal Alignment LSTM (2592→64)" — velocity encoding and temporal fusion

**Output labels (at bottom of box):**
- "head_dir ∈ S² [B×7×3]"
- "body_dir ∈ S² [B×7×3]"
- "κ_h, κ_b ∈ R⁺ [B×7×1]"

**Right margin annotation in red:** "requires_grad = False"

---

### Stage 3: Multi-Scale RHFD Feature Extraction (Non-parametric)

Rounded rectangle, background color `#FDEBD0` (pale orange).

**Title:** "Multi-Scale RHFD Feature Extraction (0 learnable params)"

**Body — upper half: five feature formulas (one per line), lower half: compression module.**

**Upper half — raw features (5 types × 3 windows W=3,5,7 → 15 dim raw):**
- "Gf: Fixation Frequency = (1/π)·arccos(h_{t-1}·h_t) — angular velocity"
- "Gd: Gaze Density = mean cosine similarity in a W-frame sliding window"
- "Ga: Head Stability = alignment of window frames to their mean direction"
- "Gv: Head-Body Correlation = rolling Pearson r between |Gf| and ‖V‖"
- "Gs: Spatial Entropy = mean pairwise cosine dissimilarity"

**Lower half (inner sub-box):**
"Multi-Scale Compression: MLP(15→32→8) + Sigmoid Gating → 8 dim final features"

**Right margin annotation in red:** "torch.no_grad() + .detach()"

**Small caption below the box:**
"Prevents arccos gradient explosion: ∂arccos(x)/∂x = −1/√(1−x²) → ±∞ at |x|→1"

---

### Stage 4: Rotation Normalization

Small rounded rectangle, background color `#ECF0F1` (light gray).

**Title:** "Rotation Normalization"

**Formula (centered):**
"R = I + K + K²·(1−c)/(s²+ε)"

**Small text below the formula:**
"Align center frame (t=4) head_dir to (0,0,−1)⊤"
"ε = 10⁻⁸: prevents division-by-zero NaN when s²=0 (key fix)"

---

### Stage 5: Enhanced GazeModule — Core Predictor

Rounded rectangle, background color `#D5F5E3` (pale green).

**Title:** "Enhanced GazeModule (770K trainable params)"

**Body — two-column layout:**

**Left column — "LSTM Encoder":**
- "Per-frame input: [κ_b·body_dir(3), κ_h·head_dir(3), RHFD_features(8)] = 11 dim"
- "Bi-LSTM: 2 layers × 128 hidden (bidirectional)"
- "Hidden states: 7 × 256 = 1792 dim"

**Right column — "Prediction Heads":**
- "Direction Head: FC(1792→64→21) → reshape to 7×3"
- "  → L2 normalize to unit sphere S²"
- "Kappa Head: FC(1792→64→7) → Softplus → κ ∈ R⁺"

**Output labels (at bottom of box):**
- "gaze_dir ∈ S² [B×7×3]"
- "κ_g ∈ R⁺ [B×7×1]"

---

### Stage 6: Output Layer (bottommost)

Small rounded rectangle, background color `#ECF0F1` (light gray).

**Content (centered):**
"Inverse Rotation: R⊤ · gaze_dir → World Coordinates"
"→ Final 3D Gaze Directions for all 7 frames"
"→ von Mises-Fisher concentration κ per frame"

---

## Sidebar: Key Design Decisions (Right side, dashed border)

- 🔒 "Frozen HBNet: 8.7M → 770K trainable (−91.9%)"
- "RHFD features: purely observational, zero extra labels"
- "Gradient isolation on arccos: eliminates NaN"
- "Multi-scale windows W=3,5,7 + learned per-frame gating"
- "AdamW optimizer (weight_decay=5×10⁻³)"
- "Cosine annealing LR schedule"
- "Random horizontal flip augmentation"
- "Train: 10 epochs, ~1 hour/epoch (NVIDIA V100 32GB)"

---

## Legend (Bottom-left corner, small font)

- 🔒 "Frozen / Non-trainable"
- Green boxes (#D5F5E3) = Trainable modules
- Orange box (#FDEBD0) = Non-parametric computation
- Blue box (#D4E6F1) = Pretrained backbone (frozen)
- Gray boxes (#ECF0F1) = Input/Output/Transform

---

## Connecting Arrows

Thick gray arrows (→) between each consecutive stage. Arrow annotations:

| From → To | Arrow Label |
|-----------|-------------|
| Stage 1 → Stage 2 | "HBNet forward pass" |
| Stage 2 → Stage 3 | "head_dir sequence" |
| Stage 3 → Stage 4 | "concatenated features [B×T×(6+8)]" |
| Stage 4 → Stage 5 | "[κ_b·b_dir, κ_h·h_dir, RHFD_features] ∈ R¹¹" |
| Stage 5 → Stage 6 | "gaze_dir, κ" |

---

## Technical Specifications

- **Color palette:**
  - Light gray: `#ECF0F1`
  - Pale blue: `#D4E6F1`
  - Pale orange: `#FDEBD0`
  - Pale green: `#D5F5E3`
  - Border color: `#333333`
  - Arrow color: `#888888`
  - Red annotation text: `#C0392B`

- **Typography:**
  - Stage titles: 11pt bold
  - Body text: 9pt regular
  - Formulas: 9pt (Computer Modern or Latin Modern Math)
  - Captions/annotations: 7.5pt italic
  - Minimum font size: 8pt

- **Layout:**
  - Vertical stack, equal spacing between stages
  - Box corner radius: 6px
  - Fixed border width: 1.5pt
  - Arrow thickness: 2pt

- **Output format:**
  - Vector (SVG or PDF), resolution ≥ 300 DPI
  - If PNG: minimum width 2000 pixels

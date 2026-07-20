# SimpleRHFD-GazeNet Architecture Diagram Prompt

Generate a clean, professional neural network architecture diagram for a computer vision paper.

---

## Overall Style

- Clean white background
- Soft rounded rectangles with thin borders (1px, #999999)
- Gentle, harmonious pastel colors — nothing neon or harsh
- Sans-serif font (Arial / Helvetica)
- Left-to-right or top-to-bottom flow with simple gray arrows
- Leave space (gray dashed boxes) where sample images can be inserted later

---

## Color Palette (Soft & Academic)

| Element | Color | Hex |
|---------|-------|-----|
| Input | Soft gray | `#F5F5F5` |
| HBNet (Frozen) | Ice blue | `#E3F2FD` |
| RHFD Features | Warm cream | `#FFF3E0` |
| GazeModule | Soft green | `#E8F5E9` |
| Output | Soft gray | `#F5F5F5` |
| Borders | Medium gray | `#BDBDBD` |
| Text | Dark gray | `#333333` |
| Arrow | Gray | `#9E9E9E` |

---

## Layout: 4 Main Blocks (Left to Right)

Arrange as a horizontal flow from left to right, or a 2×2 grid if space is tight.

---

### Block 1: Input

Box title: **Input (7-frame sequence)**

Show 3 small gray placeholder boxes labeled "[Insert body image frames]" arranged horizontally as a film strip.

Below the placeholder, list:
```
Body images (B×7×3×256×192)
Head masks (B×7×1×256×192)
Body velocity (B×7×2)
```

---

### Block 2: Frozen HBNet — Head & Body Direction

Box title: **HBNet 🔒 (Frozen, 8.7M)**

Two small placeholder boxes side-by-side:
- Left: "[Insert head crop + gaze arrow]"
- Right: "[Insert body image]"

Below:
```
EfficientNet-B0 → HeadNet + BodyNet
Output: head_dir [3], body_dir [3], κ
```

Annotation below box in red: *Pretrained weights frozen — no fine-tuning*

---

### Block 3: RHFD Temporal Features — Gaze Behavior

Box title: **RHFD Temporal Features (0 params)**

A simple table or list of the 5 features with brief descriptions:

| Feature | What It Measures |
|---------|-----------------|
| Gf | How fast gaze direction changes (frame-to-frame angular speed) |
| Gd | How concentrated gaze is (cosine similarity in a local window) |
| Ga | How stable the head is (variance of direction) |
| Gv | Whether head moves with body (correlation with walking) |
| Gs | How spread out head directions are (spatial entropy) |

Small note below: *Computed from head_dir & body_dv only — no extra labels*

---

### Block 4: GazeModule + Output

Box title: **GazeModule LSTM (770K trainable)**

Inside:
```
LSTM (11→128, 2-layer bidirectional)
  ↓
FC → 3D gaze direction + confidence (κ)
```

A small placeholder: "[Insert result image: body with gaze arrow overlay]"

Output label: **Final 3D Gaze Direction (per frame) + uncertainty**

---

## Connecting Arrows

Simple gray arrows (→) between blocks, with short text labels:

- Block 1 → Block 2: "HBNet forward pass"
- Block 2 → Block 3: "head_dir"
- Block 3 → Block 4: "head_dir + body_dir + 5 RHFD features"
- Block 4 → output: "gaze direction"

---

## Additional Annotations (small text boxes below or beside the main flow)

1. **Design:** "Only 770K trainable params (8% of original GAFA)"
2. **Training:** "10 epochs, ~1h/epoch on V100"
3. **Key trick:** "All RHFD features run under torch.no_grad() to prevent NaN"

---

## Image Placeholders

In the final PNG/SVG, leave clearly marked dashed rectangles where I will insert actual images:

- **Placeholder A:** Inside Input block — 3 small filmstrip frames of cropped body images
- **Placeholder B:** Inside HBNet block — 1 head crop with gaze arrow overlay  
- **Placeholder C:** Inside Output block — 1 body image with predicted (red) vs ground truth (green) gaze arrows

Each placeholder should have:
- A dashed gray border
- Centered text: "[Insert image]"
- Light gray fill `#FAFAFA`

---

## Example of Desired Style

Think of diagrams from papers like:
- ResNet / DenseNet architecture figures (clean blocks + arrows)
- CVPR/ICCV method overview figures (not overly detailed, visually clear)
- The key is READABILITY over exhaustive detail — a reader should understand the pipeline in 10 seconds

---

## Output Format

- SVG or PDF (vector) preferred, or high-res PNG (≥2000px wide)
- Landscape orientation, ~16:9 or ~3:2 aspect ratio

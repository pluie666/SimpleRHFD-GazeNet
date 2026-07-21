# Motivation Options After Research

## Option A: "Efficiency-First Gaze Estimation" (Recommended)

**Core argument**: The GAFA benchmark has been pushed by increasingly heavy models (GazeD: diffusion + HRNet, UAGE: 4-branch ResNet + CVAE). We show that lightweight, purely observational temporal features achieve competitive accuracy at 1-8% of their parameter cost. The contribution is **methodological** — a new paradigm for efficient video-based gaze estimation — not a new SOTA number.

**Pros**: Honest, defensible, clearly differentiated from prior work.
**Cons**: Reviewer may ask "why not just use GazeD if accuracy is what matters?"

## Option B: "Closing the Gap to SOTA"

**Core argument**: We improve over GAFA baseline (21.48° vs 21.69°) and demonstrate that RHFD features provide complementary signals. While not yet beating UAGE/GazeD, we identify the efficiency-accuracy trade-off and suggest that combining our approach with theirs could push below 19°.

**Pros**: Frame as "progress toward SOTA".
**Cons**: Weaker claim; "we didn't beat them but could" is not a paper.

## Option C: "Temporal Feature Engineering for Distant Gaze"

**Core argument**: A systematic ablation of 7 feature configurations reveals what works (Gf, Gd), what doesn't (pose, gaze-point), and why (arccos gradient NaN, HBNet overfitting). The contribution is the **ablation study itself** as a guide for future feature engineering.

**Pros**: Strong ablation story; negative results add value.
**Cons**: Less exciting than a new method; may not meet novelty bar for top venues.

---

## Recommendation: Option A (Efficiency-First) with Option C (Ablation Depth)

Position the paper as: "We're not claiming to beat GazeD at accuracy. We're showing that a fundamentally different approach — lightweight observational features + frozen backbone — can match or approach heavy methods at 1-8% the cost. And we systematically documented what works and what doesn't."

This framing has been pre-validated in the existing report's Section 6.1 (Training Efficiency and Methodological Comparison).

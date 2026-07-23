"""
Architecture diagram for GazeStateNet — clean block-diagram style.
Inspired by GAFA (CVPR 2022) Fig.1 layout.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
os.makedirs("figs", exist_ok=True)

# ── Colors (soft academic) ──
C = {
    "input":  "#F0F0F0", "backbone":"#D4E6F1", "features":"#FDEBD0",
    "head":   "#D5F5E3", "output": "#F0F0F0",  "border":"#555555",
    "text":   "#333333",  "arrow": "#888888",
}

fig, ax = plt.subplots(figsize=(9, 11), facecolor="white")
ax.set_xlim(0, 10); ax.set_ylim(0, 18.5); ax.axis("off")

def box(cx, cy, w, h, color, title="", body="", fs_title=9.5, fs_body=7.5):
    r = mpatches.FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.15",
                                 facecolor=color, edgecolor=C["border"], lw=1.2)
    ax.add_patch(r)
    if title:
        ax.text(cx, cy+h/2-0.3, title, ha="center", va="top", fontsize=fs_title,
                fontweight="bold", color=C["text"])
    if body:
        ax.text(cx, cy, body, ha="center", va="center", fontsize=fs_body,
                color=C["text"], linespacing=1.4)

def arrow(y_top, y_bot, label="", x=5):
    ax.annotate("", xy=(x, y_bot+0.15), xytext=(x, y_top-0.15),
                arrowprops=dict(arrowstyle="->", lw=2, color=C["arrow"]))
    if label:
        ax.text(x+0.35, (y_top+y_bot)/2, label, fontsize=6.5, color="#888", va="center")

# ── Stage 1 — Input ──
box(5, 16.9, 8.2, 1.5, C["input"],
    "Input: 7-frame Sequence",
    "Body Images\n[Bx7x3x256x192]\n\nHead Masks\n[Bx7x1x256x192]"
    "\n\nBody Velocity\n[Bx7x2]")
arrow(16.15, 15.3, "preprocessed from GAFA")

# ── Stage 2 — HBNet (Frozen) ──
box(5, 14.6, 8.2, 2.6, C["backbone"],
    "HBNet (Frozen, 8.7M pretrained)",
    "Shared EfficientNet-B0 Stem\n\n"
    "HeadNet (w/ Attention Mask)          BodyNet\n"
    "          TrajNet (MLP: 2 -> 32)  +  Temporal LSTM (2592 -> 64)\n\n"
    "       head_dir [3]      body_dir [3]      k_h, k_b [1]")
arrow(13.3, 12.3, "head_dir, body_dv")

# Annotation: frozen
ax.text(9.5, 14.5, "requires_grad = False", fontsize=6.5, color="#C0392B", style="italic",
        rotation=90, va="center")

# ── Stage 3 — RHFD Features ──
box(5, 11.5, 8.2, 2.4, C["features"],
    "Multi-Scale RHFD Features (0 params)",
    "Gf: Fixation Frequency           Gd: Gaze Density\n"
    "Ga: Head Stability               Gv: Head-Body Correlation\n"
    "Gs: Spatial Entropy\n\n"
    "3 windows (W=3,5,7) -> 15 dim -> MLP(15->32->8) + Sigmoid Gating")
ax.text(9.3, 11.2, "torch.no_grad() + detach()", fontsize=6.5, color="#C0392B",
        style="italic", rotation=90, va="center")
arrow(10.3, 9.3, "RHFD features [8]")

# ── Stage 4 — Rotation + GazeModule ──
box(5, 8.2, 8.2, 3.2, C["head"],
    "Rotation Normalization + Enhanced GazeModule (770K)",
    "Rotation: R = I + K + K^2 (1-c)/(s^2+eps)    (align center head to [0,0,-1])\n\n"
    "LSTM: [k_b b_dir(3), k_h h_dir(3), RHFD(8)] = 11 dim\n"
    "      2-layer bidirectional LSTM (11 -> 128, 256 output per frame)\n\n"
    "Flatten: 7x256 = 1792 -> FC(1792 -> 64 -> 21) -> 3D direction [7x3]\n"
    "                                -> FC(1792 -> 64 -> 7)  -> kappa [7x1]")
arrow(6.35, 5.35, "gaze_dir, k")

# ── Stage 5 — Output ──
box(5, 4.5, 8.2, 1.3, C["output"],
    "Inverse Rotation -> Final 3D Gaze Direction",
    "gaze_dir_world = R^T gaze_dir   (all 7 frames)\n"
    "von Mises-Fisher uncertainty kappa per frame")

# ── Sidebar — Key Design ──
sx, sy = 0.5, 2.0
sidebar = mpatches.FancyBboxPatch((sx, sy), 2.2, 2.5, boxstyle="round,pad=0.1",
                                   facecolor="white", edgecolor=C["border"], lw=1, ls="--")
ax.add_patch(sidebar)
ax.text(sx+1.1, sy+2.25, "Key Design Decisions", ha="center", fontsize=7, fontweight="bold")
lines = [
    "1. Frozen HBNet\n   (8.7M -> 770K, -92%)",
    "2. Gradient isolation\n   (arccos NaN fix)",
    "3. 5 RHFD features\n   (purely observational)",
    "4. Multi-scale W=3,5,7\n   + learned gating",
    "5. AdamW (wd=5e-3)\n   + Cosine LR",
]
for i, txt in enumerate(lines):
    ax.text(sx+0.15, sy+2.0-i*0.45, txt, fontsize=6.2, va="top", linespacing=1.1)

# ── Title ──
ax.text(5, 18.0, "GazeStateNet Architecture Overview",
        ha="center", fontsize=13, fontweight="bold")

# ── Legend ──
legend_items = [
    (C["backbone"], "Frozen backbone"), (C["features"], "Non-parametric features"),
    (C["head"], "Trainable (770K)"), (C["input"], "Input / Output"),
]
for i, (col, label) in enumerate(legend_items):
    lx = 6.5 + i*0.85
    ax.add_patch(mpatches.Rectangle((lx, 0.8), 0.7, 0.3, facecolor=col, edgecolor=C["border"], lw=1))
    ax.text(lx+0.35, 0.65, label, ha="center", fontsize=6, color=C["text"])

fig.tight_layout()
# Export
for fmt, dpi in [(".svg", None), (".pdf", None), (".png", 300)]:
    fig.savefig(f"figs/fig_architecture{fmt}", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
print("[OK] figs/fig_architecture.svg/.pdf/.png")
plt.close(fig)

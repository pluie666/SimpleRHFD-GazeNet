"""
Generate clean input/output frame illustrations for the architecture diagram.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
os.makedirs("figs/frames", exist_ok=True)

# ── Colors ──
BODY_COLOR = "#D5D8DC"
HEAD_BB = "#E74C3C"       # red — head mask outline
GT_ARROW = "#27AE60"       # green — ground truth
PRED_ARROW = "#2980B9"     # blue — predicted
BG = "#FFFFFF"

def draw_person(ax, x, y, scale=1.0, head_radius=14, body_height=120):
    """Draw a simple person silhouette."""
    s = scale
    # Head
    head = plt.Circle((x, y + body_height + head_radius), head_radius*s,
                       facecolor=BODY_COLOR, edgecolor="#999", lw=1)
    ax.add_patch(head)
    # Body (trapezoid)
    top_w = 22*s
    bot_w = 35*s
    body_h = body_height*s
    body_pts = [(x-top_w/2, y+body_h), (x+top_w/2, y+body_h),
                (x+bot_w/2, y), (x-bot_w/2, y)]
    ax.add_patch(plt.Polygon(body_pts, facecolor=BODY_COLOR, edgecolor="#999", lw=1))
    # Legs
    leg_w = 8*s; leg_h = 50*s
    for lx in [x-10*s, x+10*s]:
        ax.add_patch(plt.Rectangle((lx-leg_w/2, y-leg_h), leg_w, leg_h,
                     facecolor=BODY_COLOR, edgecolor="#999", lw=1))
    # Head bounding box (red dashed)
    cx, cy = x, y + body_height + head_radius*s
    r = head_radius*s + 4
    ax.add_patch(plt.Rectangle((cx-r, cy-r), 2*r, 2*r, fill=False,
                 edgecolor=HEAD_BB, lw=2, ls="--"))

def draw_arrow(ax, x, y, dx, dy, color, lw=2, head_w=5, head_l=7):
    """Draw an arrow."""
    ax.arrow(x, y, dx, dy, color=color, lw=lw, head_width=head_w,
             head_length=head_l, length_includes_head=True)

# ══════════════════════════════════════════════════════════
# INPUT FRAMES — 7-frame sequence with head masks
# ══════════════════════════════════════════════════════════
for mode, arrow_color, suffix in [
    ("input", None, "input_frames"),
    ("output", GT_ARROW, "output_frames"),
]:
    fig, axes = plt.subplots(1, 7, figsize=(14, 3.2), facecolor=BG)
    np.random.seed(42)
    for i, ax in enumerate(axes):
        ax.set_xlim(0, 100); ax.set_ylim(0, 190); ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("#F8F9FA")
        # subtle floor line
        ax.axhline(y=5, xmin=0.1, xmax=0.9, color="#CCC", lw=1)

        # Person in slightly varying poses
        lean = 3*np.sin(i*0.8)
        draw_person(ax, 50+lean, 10, scale=0.7, head_radius=16, body_height=130)

        # Head mask label on frame 1
        if mode == "input" and i == 0:
            ax.text(50, 170, "Head Mask (M)", fontsize=7, color=HEAD_BB,
                    fontweight="bold", ha="center")
            ax.annotate("", xy=(50, 158), xytext=(50+18, 168),
                        arrowprops=dict(arrowstyle="->", color=HEAD_BB, lw=1.2))

        # Gaze arrow for output
        if mode == "output":
            angles = [-30, -20, -15, -10, -5, 0, +10]  # varying gaze angle
            ang = np.radians(angles[i])
            dx, dy = 28*np.sin(ang), 28*np.cos(ang)
            draw_arrow(ax, 50, 148, dx, dy, arrow_color, lw=2.5)
            # ground truth overlay
            gt_ang = np.radians(angles[i]+3)  # slight offset
            gtx, gty = 30*np.sin(gt_ang), 30*np.cos(gt_ang)
            draw_arrow(ax, 50, 148, gtx, gty, GT_ARROW, lw=1.8, head_w=4, head_l=6)
            if i == 3:
                ax.text(20, 180, "— Ground Truth\n— Predicted",
                        fontsize=5.5, color="#555", va="top",
                        bbox=dict(facecolor="white", edgecolor="#DDD", pad=2))
                draw_arrow(ax, 10, 170, 8, 0, GT_ARROW, lw=1.5, head_w=3, head_l=4)
                draw_arrow(ax, 10, 162, 8, 0, PRED_ARROW, lw=1.5, head_w=3, head_l=4)

        # Frame number
        ax.text(50, 8, f"t = {i+1}", fontsize=7, ha="center", color="#888")
        ax.text(3, 183, f"F{i+1}", fontsize=6, color="#AAA")

    fig.suptitle(f"{'Input Sequence' if mode=='input' else 'Output Sequence'} — 7-frame Gaze Estimation",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"figs/frames/{suffix}.png", dpi=200, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    fig.savefig(f"figs/frames/{suffix}.pdf", bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"[OK] figs/frames/{suffix}.png/.pdf")

print("Done — input and output frame sequences ready.")

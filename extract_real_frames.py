"""
Extract REAL GAFA frames for architecture diagram input/output panels.
"""
import sys, os, cv2, torch, numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, ".")

from dataloader.gafa import create_gafa_dataset

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def denorm(img):
    """img: [C,H,W] -> [H,W,C] in [0,1]"""
    img = img.transpose(1,2,0) * STD + MEAN
    return np.clip(img, 0, 1)

# Pick across ALL training scenes for diversity
all_scenes = [
    "library/1026_3", "library/1028_2", "lab/1013_1", "lab/1014_1",
    "kitchen/1022_4", "kitchen/1015_4", "living_room/004", "living_room/005",
    "courtyard/004", "courtyard/005",
]
dset = create_gafa_dataset(7, all_scenes, root_dir="./data/preprocessed", interval=30)

os.makedirs("figs/frames", exist_ok=True)

# Try a few samples until we get a clean one
for sample_idx in [0, 5, 10, 20, 50]:
    batch = dset[sample_idx]
    img = batch["image"].numpy()          # [7, 3, 256, 192]
    mask = batch["head_mask"].numpy()      # [7, 1, 256, 192]
    gaze = batch["gaze_dir"].numpy()       # [7, 3]

    # ── INPUT FRAMES ──
    fig, axes = plt.subplots(7, 1, figsize=(2.8, 18),
                              facecolor="white", gridspec_kw={"hspace": 0.05})
    for i in range(7):
        ax = axes[i]
        frame = denorm(img[i])
        ax.imshow(frame)
        # Overlay head mask contour
        m = mask[i, 0]
        if m.max() > 0:
            contours, _ = cv2.findContours((m*255).astype(np.uint8),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                ax.plot(c[:,0,0], c[:,0,1], color="#E74C3C", lw=1.5, linestyle="--")
        ax.set_title(f"t={i+1}", fontsize=9, pad=2)
        ax.axis("off")
    fig.suptitle("Input Sequence (Head Mask M in red dashed)", fontsize=12, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.02)
    fig.savefig(f"figs/frames/input_real_{sample_idx}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"figs/frames/input_real_{sample_idx}.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── OUTPUT FRAMES ──
    fig, axes = plt.subplots(7, 1, figsize=(2.8, 18),
                              facecolor="white", gridspec_kw={"hspace": 0.05})
    for i in range(7):
        ax = axes[i]
        frame = denorm(img[i])
        ax.imshow(frame)
        # Gaze arrow (project 3D -> 2D)
        g = gaze[i]
        d2 = g[:2] / (np.linalg.norm(g[:2]) + 1e-8)
        cx, cy = 96, 70  # approx head center
        ax.arrow(cx, cy, d2[0]*35, d2[1]*35, color="#27AE60", lw=3,
                 head_width=8, head_length=8, length_includes_head=True)
        ax.set_title(f"t={i+1}", fontsize=9, pad=2)
        ax.axis("off")
    fig.suptitle("Output Sequence (Ground Truth Gaze in green)", fontsize=12, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.02)
    fig.savefig(f"figs/frames/output_real_{sample_idx}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"figs/frames/output_real_{sample_idx}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Sample {sample_idx} -> input_real_{sample_idx}.png, output_real_{sample_idx}.png")
    break  # Just one clean sample for the architecture diagram

print("Done. Check figs/frames/ for the best sample.")

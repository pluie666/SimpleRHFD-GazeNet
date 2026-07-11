"""
Generate per-sample gaze error visualization figures for the paper.

Shows individual frames with:
  - Green arrow = Ground Truth gaze
  - Red arrow   = Predicted gaze
  - Blue arrow  = Head direction (reference)
  - Angular error label in degrees
  - Sorted by error magnitude (best to worst)

Output: figs/fig_gaze_samples.png (2 rows × 4 columns = 8 examples)
"""
import os, sys, argparse
sys.path.insert(0, '.')
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.gazenet import SimpleRHFDGazeNet
from dataloader.gafa import create_gafa_dataset

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def denorm(img):
    """Denormalize a single image [C,H,W] -> [H,W,C] in [0,1]."""
    img = img.transpose(1, 2, 0) * STD + MEAN
    return np.clip(img, 0, 1)

def draw_arrow(ax, x, y, dx, dy, color, lw=2.5, label=''):
    ax.arrow(x, y, dx*45, dy*45, color=color, width=lw, head_width=7, head_length=7,
             alpha=0.85, zorder=5, label=label)

def angular_error(pred_3d, gt_3d):
    cos = np.clip(np.dot(pred_3d, gt_3d), -1, 1)
    return np.degrees(np.arccos(cos))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--weights', type=str, default='./models/weights/gazenet_GAFA.pth')
    parser.add_argument('--n_examples', type=int, default=8)
    opt = parser.parse_args()

    os.makedirs('figs', exist_ok=True)

    # Load model
    model = SimpleRHFDGazeNet(n_frames=7).cuda()
    if opt.checkpoint:
        ckpt = torch.load(opt.checkpoint, map_location='cuda')
        model.load_state_dict(ckpt['state_dict'], strict=False)
    model.load_pretrained_hbnet(opt.weights)
    model.eval()

    # Use test scenes
    test_scenes = ['living_room/006', 'library/1029_2', 'lab/1013_2', 'kitchen/1022_2']
    all_samples = []

    for scene in test_scenes:
        dset = create_gafa_dataset(7, [scene], root_dir='./data/preprocessed', interval=1)
        loader = DataLoader(dset, batch_size=1, shuffle=True, num_workers=0)
        it = iter(loader)

        for _ in range(min(5, len(dset))):  # 5 samples per scene
            batch = next(it)
            n_frames = batch['image'].shape[1]
            center = n_frames // 2

            img = batch['image'].cuda()
            hm = batch['head_mask'].cuda()
            dv = batch['body_dv'].cuda()
            gt = batch['gaze_dir'].numpy()[0, center]

            with torch.no_grad():
                res, head_r, _ = model(img, hm, dv)
            pred = res['direction'].cpu().numpy()[0, center]
            head = head_r['direction'].cpu().numpy()[0, center]

            err = angular_error(pred, gt)
            frame_img = denorm(batch['image'].numpy()[0, center])
            all_samples.append({
                'img': frame_img, 'gt': gt, 'pred': pred, 'head': head,
                'error': err, 'scene': scene,
            })

    # Sort by error magnitude for diverse visualization
    all_samples.sort(key=lambda x: x['error'])

    # Pick evenly spaced samples from best to worst
    n = len(all_samples)
    indices = [0, max(1, n//8), n//4, n//3, n//2, 2*n//3, 3*n//4, n-1]
    indices = sorted(set(min(i, n-1) for i in indices))[:opt.n_examples]
    selected = [all_samples[i] for i in indices]

    # Plot 2 rows × 4 cols (or adapt for n_examples)
    n_cols = min(4, opt.n_examples)
    n_rows = (opt.n_examples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4.5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, (ax, s) in enumerate(zip(axes.flat, selected)):
        ax.imshow(s['img'])

        # Head center approximate
        cx, cy = 96, 70

        # Draw arrows
        gt_2d = s['gt'][:2]; gt_2d /= np.linalg.norm(gt_2d) + 1e-8
        pred_2d = s['pred'][:2]; pred_2d /= np.linalg.norm(pred_2d) + 1e-8
        head_2d = s['head'][:2]; head_2d /= np.linalg.norm(head_2d) + 1e-8

        draw_arrow(ax, cx, cy, gt_2d[0], gt_2d[1], color='#2ecc71', lw=3)
        draw_arrow(ax, cx, cy, pred_2d[0], pred_2d[1], color='#e74c3c', lw=3)
        draw_arrow(ax, cx, cy+20, head_2d[0], head_2d[1], color='#3498db', lw=1.5)

        ax.set_title(f'Error: {s["error"]:.1f}°', fontsize=13, fontweight='bold',
                     color='#c0392b' if s['error'] > 25 else '#27ae60')
        ax.text(5, 245, f'Scene: {s["scene"]}', fontsize=8, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
        ax.axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Ground Truth'),
        Patch(facecolor='#e74c3c', label='Predicted (Ours)'),
        Patch(facecolor='#3498db', label='Head Direction'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12, frameon=False)

    # Hide unused subplots
    for i in range(len(selected), n_rows * n_cols):
        axes.flat[i].axis('off')

    fig.suptitle('Per-Sample Gaze Estimation: Ground Truth vs SimpleRHFD-GazeNet',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('figs/fig_gaze_samples.png')
    plt.close()
    print(f'[OK] fig_gaze_samples.png ({len(selected)} examples, error range: '
          f'{selected[0]["error"]:.1f}° – {selected[-1]["error"]:.1f}°)')

    # Also print stats for paper
    errors = [s['error'] for s in all_samples]
    print(f'\nStats over {len(errors)} samples:')
    print(f'  Mean:  {np.mean(errors):.1f}° ± {np.std(errors):.1f}°')
    print(f'  Median: {np.median(errors):.1f}°')
    print(f'  Best:  {np.min(errors):.1f}°')
    print(f'  Worst: {np.max(errors):.1f}°')
    print(f'  <10°:  {sum(1 for e in errors if e < 10)}/{len(errors)} ({100*sum(1 for e in errors if e < 10)/len(errors):.1f}%)')
    print(f'  <20°:  {sum(1 for e in errors if e < 20)}/{len(errors)} ({100*sum(1 for e in errors if e < 20)/len(errors):.1f}%)')
    print(f'  <30°:  {sum(1 for e in errors if e < 30)}/{len(errors)} ({100*sum(1 for e in errors if e < 30)/len(errors):.1f}%)')


if __name__ == '__main__':
    main()

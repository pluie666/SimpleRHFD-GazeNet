"""Per-scene evaluation matching GazeD/UAGE table format."""
import sys, torch, numpy as np
sys.path.insert(0, '.')
from dataloader.gafa import create_gafa_dataset
from torch.utils.data import DataLoader
from models.utils import compute_mae

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--which', type=str, default='v6', choices=['v6','v7'],
                    help='Model version to evaluate')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='Path to .ckpt checkpoint')
parser.add_argument('--batch_size', type=int, default=32)
opt = parser.parse_args()

# Load model
if opt.which == 'v7':
    from models.gazenet_v7 import GazeStateNetV7
    model = GazeStateNetV7(n_frames=7).cuda()
else:
    from models.gazenet import GazeStateNet
    model = GazeStateNet(n_frames=7).cuda()

ckpt = torch.load(opt.checkpoint, map_location='cuda', weights_only=False)
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()

# Test scenes (matching GazeD table: Office=lab, LivingRoom=lr, Kitchen, Library, Courtyard)
scenes = {
    'Office': ['lab/1013_2'],
    'Living Room': ['living_room/006'],
    'Kitchen': ['kitchen/1022_2'],
    'Library': ['library/1029_2'],
    'Courtyard': ['courtyard/002', 'courtyard/003'],
}

print(f"{'Scene':<14} {'#Samples':>8} {'3D MAE':>8} {'2D MAE':>8}")
print("-" * 42)

all_3d, all_2d, all_f, all_b = [], [], [], []

for scene_name, scene_list in scenes.items():
    dset = create_gafa_dataset(7, scene_list, root_dir='./data/preprocessed', interval=7)
    loader = DataLoader(dset, batch_size=opt.batch_size, num_workers=4)

    mae_3d, mae_2d = [], []
    with torch.no_grad():
        for batch in loader:
            img = batch['image'].cuda()
            hm = batch['head_mask'].cuda()
            dv = batch['body_dv'].cuda()
            gl = batch['gaze_dir'].cuda()

            if opt.which == 'v7':
                res, _, _ = model(img, hm, dv,
                                  batch.get('keypoints').cuda() if batch.get('keypoints') is not None else None,
                                  batch.get('head_pos').cuda() if batch.get('head_pos') is not None else None,
                                  batch.get('body_pos').cuda() if batch.get('body_pos') is not None else None,
                                  batch.get('body_dv_3d').cuda() if batch.get('body_dv_3d') is not None else None,
                                  batch.get('height').cuda() if batch.get('height') is not None else None)
            else:
                res, _, _ = model(img, hm, dv)
            pred = res['direction']

            if gl.shape[-1] == 3:
                m = compute_mae(pred, gl)
                if not torch.isnan(m): mae_3d.append(m.item())
                gl2 = gl[..., :2] / gl[..., :2].norm(dim=-1, keepdim=True)
                p2 = pred[..., :2] / pred[..., :2].norm(dim=-1, keepdim=True)
                m2 = compute_mae(p2, gl2)
                if not torch.isnan(m2): mae_2d.append(m2.item())

                # Front/back
                fi, bi = gl[:, 0, -1] <= 0, gl[:, 0, -1] > 0
                all_f.append(compute_mae(pred[fi], gl[fi]).item() if fi.any() else float('nan'))
                all_b.append(compute_mae(pred[bi], gl[bi]).item() if bi.any() else float('nan'))

    s3d = np.mean(mae_3d)
    s2d = np.mean(mae_2d)
    all_3d.extend(mae_3d)
    all_2d.extend(mae_2d)
    print(f"{scene_name:<14} {len(dset):>8} {s3d:>7.1f}° {s2d:>7.1f}°")

print("-" * 42)
print(f"{'Overall':<14} {sum(len(scenes[s]) for s in scenes):>8} {np.mean(all_3d):>7.1f}° {np.mean(all_2d):>7.1f}°")
print()
print(f"Front 180°: {np.mean([x for x in all_f if not np.isnan(x)]):.1f}°")
print(f"Back 180°:  {np.mean([x for x in all_b if not np.isnan(x)]):.1f}°")
print()
print("Compare with:")
print("  GAFA:   14.4 | 25.1 | 20.4 | 19.8 | 25.4 = 21.7 (Front:20.7 Back:23.2)")
print("  UAGE:   15.3 | 23.5 | 18.1 | 18.7 | 23.8 = 20.5 (Front:18.8 Back:23.7)")
print("  GazeD:  15.8 | 19.3 | 18.2 | 17.6 | 25.3 = 19.5")

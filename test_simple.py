"""Test SimpleRHFDGazeNet on GAFA test set with TTA (horizontal flip)."""
import sys, torch, numpy as np
sys.path.insert(0, '.')
from models.gazenet import SimpleRHFDGazeNet
from dataloader.gafa import create_gafa_dataset
from torch.utils.data import DataLoader
from models.utils import compute_mae

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--no_tta', action='store_true', help='Disable TTA')
opt = parser.parse_args()

model = SimpleRHFDGazeNet(n_frames=7).cuda()
ckpt = torch.load(opt.checkpoint, map_location='cuda', weights_only=False)
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()

test_exp = ['library/1029_2','lab/1013_2','kitchen/1022_2',
            'living_room/006','courtyard/002','courtyard/003']
dset = create_gafa_dataset(n_frames=7, exp_names=test_exp,
                           root_dir='./data/preprocessed', interval=7)
loader = DataLoader(dset, batch_size=opt.batch_size, num_workers=4)
print(f'Test samples: {len(dset)}  TTA: {not opt.no_tta}')

def predict(model, img, hm, dv):
    gaze_res, _, _ = model(img, hm, dv)
    return gaze_res['direction']

mae_3d, mae_2d, mae_f, mae_b = [], [], [], []
with torch.no_grad():
    for i, batch in enumerate(loader):
        img = batch['image'].cuda()
        hm = batch['head_mask'].cuda()
        dv = batch['body_dv'].cuda()
        gl = batch['gaze_dir'].cuda()

        pred = predict(model, img, hm, dv)

        # TTA: horizontal flip → average predictions
        if not opt.no_tta:
            img_flip = img.flip(-1)
            hm_flip = hm.flip(-1)
            dv_flip = dv.clone(); dv_flip[:, :, 0] *= -1
            pred_flip = predict(model, img_flip, hm_flip, dv_flip)
            pred_flip[:, :, 0] *= -1  # un-flip x component
            pred = (pred + pred_flip) / 2.0
            pred = pred / pred.norm(dim=-1, keepdim=True)

        if gl.shape[-1] == 3:
            fi, bi = gl[:, 0, -1] <= 0, gl[:, 0, -1] > 0
            for lst, idx in [(mae_f, fi), (mae_b, bi), (mae_3d, slice(None))]:
                m = compute_mae(pred[idx], gl[idx])
                if not torch.isnan(m): lst.append(m.item())
            g2 = gl[..., :2] / gl[..., :2].norm(dim=-1, keepdim=True)
            p2 = pred[..., :2] / pred[..., :2].norm(dim=-1, keepdim=True)
            m2 = compute_mae(p2, g2)
            if not torch.isnan(m2): mae_2d.append(m2.item())

print(f'MAE 3D all:   {np.mean(mae_3d):.2f}')
print(f'MAE 2D all:   {np.mean(mae_2d):.2f}')
print(f'MAE 3D front: {np.mean(mae_f):.2f}')
print(f'MAE 3D back:  {np.mean(mae_b):.2f}')

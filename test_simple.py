"""Test SimpleRHFDGazeNet on GAFA test set."""
import sys, torch, numpy as np
sys.path.insert(0, '.')
from models.gazenet import SimpleRHFDGazeNet
from dataloader.gafa import create_gafa_dataset
from torch.utils.data import DataLoader
from models.utils import compute_mae

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True,
                    help='Path to .ckpt checkpoint')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--plot', action='store_true', help='Plot angular distribution')
opt = parser.parse_args()

model = SimpleRHFDGazeNet(n_frames=7).cuda()
ckpt = torch.load(opt.checkpoint, map_location='cuda')
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()

test_exp = ['library/1029_2','lab/1013_2','kitchen/1022_2',
            'living_room/006','courtyard/002','courtyard/003']
dset = create_gafa_dataset(n_frames=7, exp_names=test_exp,
                           root_dir='./data/preprocessed', interval=7)
loader = DataLoader(dset, batch_size=opt.batch_size, num_workers=4)
print(f'Test samples: {len(dset)}')

mae_3d, mae_2d, mae_f, mae_b = [], [], [], []
with torch.no_grad():
    for i,batch in enumerate(loader):
        img,hm,dv=batch['image'].cuda(),batch['head_mask'].cuda(),batch['body_dv'].cuda()
        gl=batch['gaze_dir'].cuda()
        gaze_res,_,_=model(img,hm,dv)
        pred=gaze_res['direction']
        if gl.shape[-1]==3:
            fi,bi=gl[:,0,-1]<=0,gl[:,0,-1]>0
            for lst,idx in [(mae_f,fi),(mae_b,bi),(mae_3d,slice(None))]:
                m=compute_mae(pred[idx],gl[idx])
                if not torch.isnan(m): lst.append(m.item())
            g2=gl[...,:2]/gl[...,:2].norm(dim=-1,keepdim=True)
            p2=pred[...,:2]/pred[...,:2].norm(dim=-1,keepdim=True)
            m2=compute_mae(p2,g2)
            if not torch.isnan(m2): mae_2d.append(m2.item())

print(f'MAE 3D all:   {np.mean(mae_3d):.2f}')
print(f'MAE 2D all:   {np.mean(mae_2d):.2f}')
print(f'MAE 3D front: {np.mean(mae_f):.2f}')
print(f'MAE 3D back:  {np.mean(mae_b):.2f}')
print()
print('v3 best:      21.49 | 20.44 | 19.96 | 23.45')

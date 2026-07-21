"""SimpleRHFDGazeNet V7: GazeD gaze-point + UAGE pose features + RHFD temporal features."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, pytorch_lightning as pl
from models.hbnet import HBNet
from models.utils import get_rotation, compute_mae
from models.loss import compute_basic_cos_loss, compute_kappa_vMF3_loss
from models.rhfd_features import MultiScaleRHFDExtractor
from models.pose_features import PoseFeatureExtractor


class SimpleRHFDGazeNetV7(pl.LightningModule):
    """V7: GazeD gaze-point head + UAGE pose features + RHFD temporal features.
    LSTM input: body_dir(3)+head_dir(3)+rhfd(8)+pose(6)=20 dim."""

    def __init__(self, n_frames=7, rhfd_dim=8, pose_dim=6):
        super().__init__()
        self.n_frames = n_frames
        self.hbnet = HBNet()
        self.rhfd_extractor = MultiScaleRHFDExtractor(
            window_sizes=(3, 5, 7), hidden_dim=32, output_dim=rhfd_dim)
        self.pose_extractor = PoseFeatureExtractor(output_dim=pose_dim)

        lstm_in = 6 + rhfd_dim + pose_dim  # 20
        lstm_out = 128 * 2  # bidirectional
        self.lstm = nn.LSTM(lstm_in, 128, bidirectional=True, num_layers=2)
        fd = lstm_out * n_frames  # 1792

        self.dir_layer = nn.Sequential(nn.Linear(fd, 64), nn.ReLU(), nn.Linear(64, 3*n_frames))
        self.pt_layer = nn.Sequential(nn.Linear(fd, 64), nn.ReLU(), nn.Linear(64, 3*n_frames))
        self.kappa_layer = nn.Sequential(nn.Linear(fd, 64), nn.ReLU(), nn.Linear(64, n_frames), nn.Softplus())
        self.automatic_optimization = False

    def load_pretrained_hbnet(self, path, map_location='cpu', freeze=True):
        s = torch.load(path, map_location=map_location, weights_only=False)
        if 'state_dict' in s: s = s['state_dict']
        hb = {k[6:]: v for k, v in s.items() if k.startswith('hbnet.')}
        self.hbnet.load_state_dict(hb, strict=True)
        if freeze:
            for p in self.hbnet.parameters(): p.requires_grad = False
        print(f'Loaded {len(hb)} HBNet params' + (' (FROZEN)' if freeze else ''))
        return self

    def forward(self, img, head_mask, body_dv, kp=None, hp=None, bp=None, dv3d=None, ht=None):
        B, T = img.shape[0], self.n_frames
        ho, bo = self.hbnet(img, head_mask, body_dv)

        with torch.no_grad():
            rf = self.rhfd_extractor(ho['direction'].detach(), body_dv.detach())['concat'].detach()
            if kp is not None and hp is not None:
                pf = self.pose_extractor(
                    kp.float(), hp.float() if hp is not None else None,
                    bp.float() if bp is not None else None,
                    dv3d.float() if dv3d is not None else None,
                    ht.float() if ht is not None else None,
                )['concat'].detach()
            else:
                pf = torch.zeros(B, T, 6, device=img.device)

        ref = ho['direction'][:, T // 2]
        dst = torch.zeros_like(ref); dst[:, 2] = -1
        R = get_rotation(ref, dst)
        hd = torch.einsum('bij,bfj->bfi', R, ho['direction']) * ho['kappa']
        bd = torch.einsum('bij,bfj->bfi', R, bo['direction']) * bo['kappa']

        x = torch.cat([bd, hd, rf, pf], dim=2).transpose(0, 1)
        lo, _ = self.lstm(x)
        lo = F.relu(lo.transpose(0, 1)).reshape(B, -1)

        d = self.dir_layer(lo).reshape(B, T, 3)
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
        p = self.pt_layer(lo).reshape(B, T, 3)
        dp = p / (p.norm(dim=-1, keepdim=True) + 1e-8)
        k = self.kappa_layer(lo).reshape(B, T, 1)

        Rt = R.transpose(1, 2)
        return {
            'direction': torch.einsum('bij,bfj->bfi', Rt, d),
            'direction_from_point': torch.einsum('bij,bfj->bfi', Rt, dp),
            'gaze_point': torch.einsum('bij,bfj->bfi', Rt, p),
            'kappa': k, 'probs': None, 'entropy': None,
        }, ho, bo

    def configure_optimizers(self):
        od = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                               lr=1e-4, weight_decay=5e-3)
        ok = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                               lr=1e-4, weight_decay=5e-3)
        sd = torch.optim.lr_scheduler.CosineAnnealingLR(od, T_max=100000)
        sk = torch.optim.lr_scheduler.CosineAnnealingLR(ok, T_max=100000)
        return [od, ok], [sd, sk]

    def training_step(self, batch, batch_idx):
        img, hm, dv = batch['image'], batch['head_mask'], batch['body_dv']
        kp, hp, bp = batch.get('keypoints'), batch.get('head_pos'), batch.get('body_pos')
        res, hr, br = self.forward(img, hm, dv, kp, hp, bp,
                                   batch.get('body_dv_3d'), batch.get('height'))
        if torch.isnan(res['direction']).any():
            return None

        od, ok = self.optimizers()

        if batch_idx % 10 != 0:
            lc = (compute_basic_cos_loss(hr, batch['head_dir']) +
                  compute_basic_cos_loss(br, batch['body_dir']) +
                  compute_basic_cos_loss(res, batch['gaze_dir'])) / 3.0
            ll = F.mse_loss(res['gaze_point'], batch['gaze_dir'].cuda())
            loss = lc + 0.3 * ll
            od.zero_grad(); self.manual_backward(loss)
            self.clip_gradients(od, gradient_clip_val=1.0, gradient_clip_algorithm='norm')
            od.step()
            self.log_dict({'direction_loss': lc, 'l2_loss': ll}, prog_bar=True)
        else:
            loss = (compute_kappa_vMF3_loss(hr, batch['head_dir']) +
                    compute_kappa_vMF3_loss(br, batch['body_dir']) +
                    compute_kappa_vMF3_loss(res, batch['gaze_dir'])) / 3.0
            if torch.isnan(loss): return None
            ok.zero_grad(); self.manual_backward(loss)
            self.clip_gradients(ok, gradient_clip_val=1.0, gradient_clip_algorithm='norm')
            ok.step()
            self.log_dict({'kappa_loss': loss}, prog_bar=True)

        self.log('train_mae', compute_mae(res['direction'], batch['gaze_dir']))
        return loss

    def validation_step(self, batch, batch_idx):
        img, hm, dv = batch['image'], batch['head_mask'], batch['body_dv']
        res, _, _ = self.forward(img, hm, dv, batch.get('keypoints'),
                                 batch.get('head_pos'), batch.get('body_pos'),
                                 batch.get('body_dv_3d'), batch.get('height'))
        m = compute_mae(res['direction'], batch['gaze_dir'])
        self.log('val_mae', m)
        self.log('val_loss', compute_kappa_vMF3_loss(res, batch['gaze_dir']))
        return m

    def validation_epoch_end(self, outputs):
        v = torch.stack([x for x in outputs]).flatten()
        m = v[~torch.isnan(v)].mean()
        print('MAE (validation): ', m)
        self.log('val_mae', m)

    def test_step(self, batch, batch_idx):
        img, hm, dv = batch['image'], batch['head_mask'], batch['body_dv']
        res, _, _ = self.forward(img, hm, dv, batch.get('keypoints'),
                                 batch.get('head_pos'), batch.get('body_pos'),
                                 batch.get('body_dv_3d'), batch.get('height'))
        pred, gl = res['direction'], batch['gaze_dir']
        if gl.shape[-1] == 3:
            fi, bi = gl[:, 0, -1] <= 0, gl[:, 0, -1] > 0
            m = compute_mae(pred, gl)
            fm = compute_mae(pred[fi], gl[fi])
            bm = compute_mae(pred[bi], gl[bi])
            g2 = gl[..., :2] / gl[..., :2].norm(dim=-1, keepdim=True)
            p2 = pred[..., :2] / pred[..., :2].norm(dim=-1, keepdim=True)
            m2 = compute_mae(p2, g2)
            fm2 = compute_mae(p2[fi], g2[fi])
            bm2 = compute_mae(p2[bi], g2[bi])
        else:
            g2 = gl[..., :2] / gl[..., :2].norm(dim=-1, keepdim=True)
            p2 = pred[..., :2] / pred[..., :2].norm(dim=-1, keepdim=True)
            m = fm = bm = 0
            m2 = compute_mae(p2, g2); fm2 = bm2 = 0
        return m, m2, fm, fm2, bm, bm2

    def test_epoch_end(self, outputs):
        m = np.nanmean([x[0] for x in outputs])
        m2 = np.nanmean([x[1] for x in outputs])
        fm = np.nanmean([x[2] for x in outputs])
        fm2 = np.nanmean([x[3] for x in outputs])
        bm = np.nanmean([x[4] for x in outputs])
        bm2 = np.nanmean([x[5] for x in outputs])
        print(f'MAE (3D front): {fm:.4f}')
        print(f'MAE (2D front): {fm2:.4f}')
        print(f'MAE (3D back):  {bm:.4f}')
        print(f'MAE (2D back):  {bm2:.4f}')
        print(f'MAE (3D all):   {m:.4f}')
        print(f'MAE (2D all):   {m2:.4f}')

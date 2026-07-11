# SimpleRHFD-GazeNet: Enhancing 3D Gaze Estimation via Gaze-State Temporal Features

This repository provides an enhanced implementation of dynamic 3D gaze estimation, built upon the [GAFA framework](https://openaccess.thecvf.com/content/CVPR2022/html/Nonaka_Dynamic_3D_Gaze_From_Afar_Deep_Gaze_Estimation_From_Temporal_CVPR_2022_paper.html) (Nonaka et al., CVPR 2022). **SimpleRHFD-GazeNet** integrates five gaze-state temporal features — fixation frequency (Gf), gaze density (Gd), head stability (Ga), head-body correlation (Gv), and spatial entropy (Gs) — to improve gaze direction accuracy.

### Key Results

| Metric | Original GAFA | SimpleRHFD (Ours) | Improvement |
|--------|:---:|:---:|:---:|
| 3D MAE (all) | 21.69° | **21.49°** | -0.20° |
| 3D MAE (front) | 20.70° | **19.96°** | -0.74° |
| 2D MAE (all) | 20.89° | **20.44°** | -0.45° |
| Trainable Params | 9.5M | **770K** | -91.9% |

## Citation

If you use this code, please cite both the original GAFA paper and our work:

```bibtex
@InProceedings{Nonaka_2022_CVPR,
    author    = {Nonaka, Soma and Nobuhara, Shohei and Nishino, Ko},
    title     = {Dynamic 3D Gaze From Afar: Deep Gaze Estimation From Temporal Eye-Head-Body Coordination},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {2192-2201}
}
```

## Architecture

```
Body Image [B,T,3,256,192] + Head Mask + Body Velocity
    │
    ▼
┌─────────────────────────────┐
│      HBNet (frozen)         │  ← Pretrained EfficientNet-B0 branches
│  head_dir, body_dir, kappa  │
└─────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│   RHFD Feature Extraction    │  ← Gf, Gd, Ga, Gv, Gs (no learnable params)
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  Rotation Normalization       │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│    Enhanced GazeModule (770K)     │
│  LSTM(11→128, bidir×2)           │
│  → Flatten → FC → direction + κ  │
└──────────────────────────────────┘
    │
    ▼
  Inverse Rotation → Final 3D Gaze
```

## Prerequisites

Tested with Python 3.8 on Ubuntu 20.04. Dependencies:

```
pip install pytorch-lightning efficientnet-pytorch albumentations opencv-python-headless matplotlib tqdm
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/pluie666/SimpleRHFD-GazeNet.git
cd SimpleRHFD-GazeNet
pip install pytorch-lightning efficientnet-pytorch albumentations opencv-python-headless matplotlib tqdm
```

### 2. Download data and weights

- Preprocessed GAFA data: [Google Drive (5.9GB)](https://drive.google.com/file/d/1ef8uKVlq4jLKGZ2gVLDwHx6u0HayPTZf/view?usp=sharing)
- Pretrained GAFA weights: [Google Drive](https://drive.google.com/file/d/1oJVaaNoMo9_qoo7q1z1ek1y-gjXwy55O/view?usp=sharing)

```bash
# Extract data
tar -xzf preprocessed.tar.gz -C data/preprocessed/

# Place weights
mkdir -p models/weights
mv gazenet_GAFA.pth models/weights/
```

### 3. Demo inference

```bash
python3 << 'EOF'
import torch, cv2, numpy as np, matplotlib.pyplot as plt
from models.gazenet import SimpleRHFDGazeNet
from dataloader.gafa import create_gafa_dataset

model = SimpleRHFDGazeNet(n_frames=7).cuda()
model.load_pretrained_hbnet('models/weights/gazenet_GAFA.pth')
model.eval()

dset = create_gafa_dataset(7, ['living_room/006'], root_dir='./data/preprocessed/')
batch = dset[0]
img = batch['image'].cuda().unsqueeze(0)
hm = batch['head_mask'].cuda().unsqueeze(0)
dv = batch['body_dv'].cuda().unsqueeze(0)

with torch.no_grad():
    gaze_res, _, _ = model(img, hm, dv)

print(f"Gaze direction: {gaze_res['direction'][0,0].cpu().numpy()}")
print(f"Confidence (kappa): {gaze_res['kappa'][0,0].item():.2f}")
EOF
```

### 4. Training

```bash
# Train SimpleRHFD-GazeNet with frozen HBNet (recommended)
python train.py --simple --epoch 20 --n_frames 7 --gpus 1 \
  --batch_size 32 --freeze_hbnet \
  --weights ./models/weights/gazenet_GAFA.pth \
  --checkpoint output/
```

**Training flags:**

| Flag | Description |
|------|-------------|
| `--simple` | Use SimpleRHFD-GazeNet (minimal, stable) |
| `--freeze_hbnet` | Freeze pretrained HBNet (recommended) |
| `--no_freeze_hbnet` | Allow HBNet fine-tuning |
| `--no_probability_head` | Use direct regression head |
| `--no_twiesn` | Disable TWIESN |
| `--no_ema` | Disable EMA smoothing |
| `--quick` | Single-scene debug mode |
| `--batch_size N` | Batch size (default 32) |

Hardware: V100 32GB, ~1 hour/epoch with frozen HBNet.

### 5. Evaluation

```bash
# Test on GAFA test set
python eval.py --n_frames 7 --gpus 1
```

## GAFA Dataset

The model is trained and evaluated on the [Gaze from Afar (GAFA) dataset](https://openaccess.thecvf.com/content/CVPR2022/html/Nonaka_Dynamic_3D_Gaze_From_Afar_Deep_Gaze_Estimation_From_Temporal_CVPR_2022_paper.html) (Nonaka et al., CVPR 2022), provided under CC BY 4.0 license. The dataset contains 5 daily scenes (lab, library, kitchen, courtyard, living room) with multi-camera surveillance footage and 3D gaze/head/body annotations.

**Data download links:**

- Preprocessed data (5.9GB): [Google Drive](https://drive.google.com/file/d/1ef8uKVlq4jLKGZ2gVLDwHx6u0HayPTZf/view?usp=sharing)
- Raw data (~1.7TB, multi-part): See original [GAFA repository](https://github.com/kyotovision-public/dynamic-3d-gaze-from-afar)

## Project Structure

```
├── models/
│   ├── gazenet.py          # SimpleRHFDGazeNet + original GazeNet
│   ├── hbnet.py            # HBNet (head/body direction estimation)
│   ├── rhfd_features.py    # 5 RHFD temporal features (Gf,Gd,Ga,Gv,Gs)
│   ├── rhfd_mapping.py     # Probability-first gaze mapping (experimental)
│   ├── twiesn.py           # TWIESN reservoir (experimental)
│   ├── smoothing.py        # EMA smoothing (experimental)
│   ├── loss.py             # Loss functions (cosine, vMF, probability)
│   └── utils.py            # Rotation, MAE, sphere anchors
├── dataloader/
│   └── gafa.py             # GAFA dataset loader
├── train.py                # Training script
├── eval.py                 # Evaluation script
├── demo.ipynb              # Demo notebook
├── report.md               # Technical report (English)
└── report_cn.md            # Technical report (Chinese)
```

## Reports

- [Technical Report (English)](report.md)
- [技术报告（中文）](report_cn.md)

## License

This project builds upon the GAFA framework. The GAFA dataset is provided under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/). Please refer to the original repository for dataset licensing and ethical approval details.

"""
Evaluate SimpleRHFD-GazeNet on the GAFA test set.

Usage:
    python eval.py --checkpoint output_v6/.../checkpoint.ckpt --gpus 1
    python eval.py --checkpoint output_v6/.../checkpoint.ckpt --gpus 1 --no_tta
"""
import argparse

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from dataloader.gafa import create_gafa_dataset
from models.gazenet import SimpleRHFDGazeNet

# GAFA test set (6 held-out scenes)
test_exp_names = [
    'library/1029_2',
    'lab/1013_2',
    'kitchen/1022_2',
    'living_room/006',
    'courtyard/002',
    'courtyard/003',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_frames", type=int, default=7)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained .ckpt checkpoint")
    parser.add_argument("--weights", type=str,
                        default='./models/weights/gazenet_GAFA.pth',
                        help="Path to pretrained GAFA weights (for HBNet)")
    parser.add_argument("--no_tta", action="store_true",
                        help="Disable test-time augmentation")
    opt = parser.parse_args()

    # Load model
    model = SimpleRHFDGazeNet(n_frames=opt.n_frames)
    model.load_pretrained_hbnet(opt.weights)
    ckpt = torch.load(opt.checkpoint, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"Model loaded. Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Make dataloader
    test_dset = create_gafa_dataset(
        n_frames=opt.n_frames,
        exp_names=test_exp_names,
        interval=7,
    )
    test_loader = DataLoader(test_dset, batch_size=32,
                             num_workers=4, shuffle=False)
    print(f"Test samples: {len(test_dset)}  TTA: {not opt.no_tta}")

    trainer = Trainer(
        benchmark=True,
        gpus=opt.gpus,
        precision=16,
        accelerator='ddp',
    )

    trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    main()

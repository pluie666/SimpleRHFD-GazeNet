# Evidence Bank — SimpleRHFD-GazeNet

| Claim | Evidence Type | Source | Strength |
|-------|-------------|--------|:---:|
| 21.48° 3D MAE | Quantitative | test_simple.py output | Strong |
| 0.82° frontal improvement | Quantitative | eval_per_scene.py | Strong |
| 770K trainable params | Quantitative | Model summary (train.py output) | Strong |
| NaN fix works | Qualitative + Quantitative | Zero NaN in 100+ epochs | Strong |
| Frozen HBNet improves by 3.01° | Quantitative | v1→v3 ablation | Strong |
| Gap narrows 13.8°→6.7° | Quantitative | v3 vs v6 val/test | Strong |
| Gf/Gd drive most gain | Quantitative | v3 (2-feat) vs v5 (5-feat) | Strong |
| Pose features don't help (V7) | Quantitative (negative) | v7 experiment | Moderate |
| Gaze-point doesn't help (V7b) | Quantitative (negative) | v7b experiment | Moderate |
| Office scene best (14.3°) | Quantitative | eval_per_scene.py | Strong |
| Converges in 1-2 epochs | Quantitative | Training curves | Strong |
| UAGE is 20.5° | Citation | UAGE paper Table | Strong |
| GazeD is 19.5° | Citation | GazeD paper Table | Strong |
| RHFD features are observational | Qualitative (definitional) | Code: rhfd_features.py | Strong |

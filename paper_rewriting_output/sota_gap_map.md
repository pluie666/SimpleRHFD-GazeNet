# SOTA Gap Map — SimpleRHFD-GazeNet

## Current State of the Art on GAFA Benchmark

| Rank | Method | 3D MAE | Architecture | Params | Year |
|------|--------|:---:|------|:---:|------|
| 1 | GazeD (AVG) | 19.5° | Diffusion + HRNet + DETR | >20M | 2026 |
| 2 | GazeD (Oracle) | 15.9° | Same + oracle context | >20M | 2026 |
| 3 | UAGE | 20.5° | 4-branch ResNet+STGCN+CVAE | >10M | 2024 |
| 4 | IEEE Gaze-Pose | 22.2° | Body pose + Transformer | ~5M | 2024 |
| **5** | **SimpleRHFD (ours)** | **21.48°** | **Frozen EffNet + 5 RHFD features** | **770K** | **2026** |
| 6 | GAFA (original) | 21.69° | EfficientNet + LSTM | 9.5M | 2022 |
| 7 | Frontal gaze baseline | 28.8° | — | 0 | — |

## Gap Analysis

### Accuracy Gap vs SOTA

| Comparison | Our MAE | Their MAE | Gap | Meaning |
|------------|:---:|:---:|:---:|------|
| vs GazeD (AVG) | 21.48° | 19.5° | +1.98° | GazeD is substantially more accurate |
| vs UAGE | 21.48° | 20.5° | +0.98° | UAGE is moderately more accurate |
| vs GAFA | 21.48° | 21.69° | −0.21° | We beat the original baseline |

### Efficiency Gap (Our Strength)

| Comparison | Our Params | Their Params | Ratio | Meaning |
|------------|:---:|:---:|:---:|------|
| vs GazeD | 770K | >20M | 26× | We are 26× smaller |
| vs UAGE | 770K | >10M | 13× | We are 13× smaller |
| vs GAFA | 770K | 9.5M | 12× | We are 12× smaller |

### Convergence Speed Gap

| Comparison | Our Epochs | Their Epochs | Ratio | Meaning |
|------------|:---:|:---:|:---:|------|
| vs GazeD | 1-2 | 100 | 50-100× | We converge 50-100× faster |
| vs GAFA | 1-2 | 50-100 | 25-50× | We converge 25-50× faster |

## Unique Position Claims

1. **Only method on GAFA that uses purely observational temporal features** (Gf, Gd, Ga, Gv, Gs) — UAGE uses learned pose features, GazeD uses learned scene context
2. **Only method that freezes the pretrained backbone** — all others fine-tune their visual backbones
3. **Only method reporting a systematic NaN fix** — the arccos gradient isolation problem is undocumented in prior GAFA work
4. **Honest negative results** — we report that UAGE-inspired and GazeD-inspired features don't help at this resolution, saving future researchers time

## Positioning Strategy

```
Accuracy:  GazeD (19.5°) > UAGE (20.5°) > SimpleRHFD (21.5°) > GAFA (21.7°)
Efficiency: SimpleRHFD (770K) ≫ GAFA (9.5M) ≫ UAGE (>10M) ≫ GazeD (>20M)
Speed:     SimpleRHFD (1-2ep) ≫ GAFA (50ep) ≫ GazeD (100ep)
```

**Recommended framing**: "SimpleRHFD-GazeNet occupies the efficiency-accuracy Pareto frontier for GAFA gaze estimation — competitive accuracy at a fraction of the parameter and training cost of heavier SOTA methods."

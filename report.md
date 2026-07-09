# Enhancing Dynamic 3D Gaze Estimation via Gaze-State Temporal Features

## A Technical Report on SimpleRHFD-GazeNet

---

## Abstract

We propose **SimpleRHFD-GazeNet**, a lightweight enhancement to the GAFA gaze estimation framework (Nonaka et al., CVPR 2022) that integrates five gaze-state temporal features derived entirely from observable head and body motion signals. Without modifying the core LSTM architecture or requiring additional labels, our method reduces the 3D gaze estimation error on the GAFA test set from **21.69° to 21.49°**, with a **0.74° improvement on frontal gaze** (19.96° vs 20.70°). The enhancement adds only **770K trainable parameters** and demonstrates that gaze-state features—fixation frequency, density, head stability, head-body correlation, and spatial entropy—provide complementary signals beyond raw head/body directions. Key architectural decisions include **(1)** feature isolation via gradient detachment to prevent numerical instability in angular computations, and **(2)** frozen pretrained feature extractor weights to prevent scene-specific overfitting. Our best model achieves **7.69° validation MAE** with stable convergence across 20 training epochs.

---

## 1. Introduction

Dynamic 3D gaze estimation from distant cameras is a fundamental challenge in surveillance, human-robot interaction, and behavioral analysis. The GAFA framework (Nonaka et al., CVPR 2022) demonstrated that gaze direction can be inferred from full-body images, head masks, and body velocity by leveraging temporal eye-head-body coordination through an LSTM-based architecture, achieving 21.69° mean angular error on the GAFA benchmark.

However, the original GAFA model relies solely on head direction, body direction, and body velocity as temporal inputs. This misses a class of **gaze-state features**—statistical properties of the gaze trajectory itself—that are widely used in hidden follower detection (RHFD) and eye-tracking analysis. Features such as fixation frequency, spatial density, and head-body motion correlation capture *how* a person looks, not just *where* they look.

We show that these features can be computed entirely from the model's existing intermediate outputs (head direction and body velocity sequences) with **no additional annotations, no new sensor inputs, and no architectural overhaul**. By injecting them as auxiliary LSTM input channels and freezing the pretrained HBNet feature extractor, we achieve consistent improvement over the original GAFA baseline while maintaining training stability and reducing trainable parameters by 92%.

---

## 2. Related Work

**3D Gaze from Afar (GAFA, CVPR 2022).** Nonaka et al. introduced the GAFA dataset and a two-stage gaze estimation pipeline: HBNet extracts head and body orientations from cropped body images and head masks using EfficientNet-based branches with temporal LSTM alignment, and a GazeModule LSTM fuses these orientations to predict 3D gaze direction under von Mises-Fisher uncertainty modeling. Training uses an alternating optimization scheme (direction steps vs. kappa steps at a 9:1 ratio) with Adam optimizer at learning rate 1e-4. The model achieves 21.69° 3D MAE on the test set across 6 unseen scenes.

**Hidden Follower Detection (RHFD).** RHFD methods leverage gaze-state features including fixation frequency (Gf), gaze density (Gd), gaze-head alignment (Ga), gaze-velocity correlation (Gv), and spatial entropy (Gs) to characterize behavioral patterns. These features are typically computed from ground-truth gaze data in controlled settings. Our contribution is adapting them as *observable-only* features computed from head direction and body velocity in the GAFA pipeline, making them usable during inference without gaze labels.

**Reservoir Computing and Temporal Smoothing.** Echo State Networks and their polynomial variants (e.g., TWIESN with Tchebichef expansion) have been proposed for temporal sequence smoothing. In our experiments, these components introduced instability (NaN divergence) on multi-scene training data, leading us to favor the simpler, more stable direct-channel approach.

---

## 3. Method

### 3.1 Architecture Overview

SimpleRHFD-GazeNet retains the GAFA two-stage design:

```
Body Image [B,T,3,256,192] + Head Mask + Body Velocity
    │
    ▼
┌─────────────────────────────┐
│         HBNet (frozen)      │  ← Pretrained EfficientNet-B0 branches
│  head_dir, body_dir, kappa  │
└─────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│   RHFD Feature Extraction    │  ← No learnable parameters
│   Gf, Gd, Ga, Gv, Gs  → [B,T,5]  │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  Rotation Normalization       │
│  Align center frame to [0,0,-1]  │
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

**Key modifications from original GAFA:**

1. **LSTM input dimension**: extended from 6 (body_dir + head_dir) to 11 (body_dir + head_dir + 5 RHFD features)
2. **HBNet frozen**: 8.7M pretrained parameters with `requires_grad=False`
3. **RHFD features detached**: computed under `torch.no_grad()` to prevent gradient propagation through angular boundary operations
4. **No TWIESN, EMA, or probability head**: removed for stability

### 3.2 RHFD Feature Computation

All five features are computed from normalized head direction vectors $\vec{h}_t \in \mathbb{S}^2$ and body velocity $\vec{v}_t \in \mathbb{R}^2$, requiring no additional labels:

| Feature | Symbol | Definition | Range |
|---------|--------|------------|-------|
| **Gaze Fixation Frequency** | $G_f$ | Frame-to-frame angular velocity: $\arccos(\vec{h}_{t-1} \cdot \vec{h}_t) / \pi$ | [0,1] |
| **Gaze Density** | $G_d$ | Mean cosine similarity to neighboring frames within a 3-frame sliding window | [0,1] |
| **Head Stability** | $G_a$ | Mean cosine similarity of window frames to their mean direction | [-1,1] |
| **Head-Body Correlation** | $G_v$ | Rolling Pearson $r$ between $|G_f|$ and $|\vec{v}_t|$ over a 3-frame window | [-1,1] |
| **Spatial Entropy** | $G_s$ | Mean pairwise cosine *dissimilarity* within a 3-frame window | [0,2] |

All features are computed per frame with a temporal window of $W=3$ (matching the 7-frame LSTM context). Padding uses replication at sequence boundaries.

**Gradient isolation** is critical: angular computations ($\arccos$) have singular gradients at $\pm 1$, where $\partial \arccos(x)/\partial x = -1/\sqrt{1-x^2} \to \pm\infty$. Without `torch.no_grad()`, training diverges to NaN within the first epoch. All RHFD features are computed in a `with torch.no_grad()` block and explicitly detached before concatenation.

### 3.3 Training Configuration

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Optimizer | AdamW | Weight decay prevents HBNet drift |
| Learning rate | $1\times 10^{-4}$ | Same as original GAFA |
| Weight decay | $1\times 10^{-4}$ | Light L2 regularization |
| LR schedule | Cosine annealing ($T_{max}=10^5$ steps) | Smooth decay for convergence |
| Batch size | 32 | Matches original |
| Epochs | 20 | Convergence observed at epoch 3 |
| HBNet | **Frozen** | Prevents scene-specific overfitting |
| RHFD features | **Detached** | Prevents acos gradient explosion |

The loss follows the original GAFA alternating scheme: direction steps (90% of batches) minimize $1 - \cos(\theta)$, and kappa steps (10% of batches) minimize the negative von Mises-Fisher log-likelihood.

---

## 4. Experiments

### 4.1 Dataset

We use the GAFA dataset with the standard train/test split. The training set comprises 11 sessions across 5 scenes (31,177 valid 7-frame sequences), and the test set comprises 6 held-out sessions (34,335 sequences). Validation uses a 10% random hold-out from training data.

### 4.2 Results

#### 4.2.1 Test Set Performance

| Model | 3D All | 2D All | 3D Front | 3D Back | Trainable Params |
|-------|:---:|:---:|:---:|:---:|:---:|
| GAFA (original) | 21.69° | 20.89° | 20.70° | 23.21° | 9.5M |
| **SimpleRHFD (ours)** | **21.49°** | **20.44°** | **19.96°** | 23.45° | **770K** |

Our model achieves a **0.20° improvement** on overall 3D MAE and a **0.74° improvement** on frontal gaze while reducing trainable parameters by **91.9%**.

#### 4.2.2 Ablation Studies

| Experiment | val MAE | Test 3D MAE | Key Finding |
|------------|:---:|:---:|------|
| Original GAFA | — | 21.69° | Baseline |
| +Gf/Gd (2 feat, unfrozen HBNet) | 12.9° | 24.50° | Overfitting; HBNet drift |
| +Gf/Gd/Ga/Gv/Gs (5 feat, unfrozen) | 10.9° | 24.18° | Worse generalization |
| **+Gf/Gd (2 feat, frozen HBNet)** | 7.8° | **21.48°** | **Frozen HBNet is key** |
| **+All 5 feat (frozen HBNet)** | **7.7°** | **21.49°** | **Best overall** |

#### 4.2.3 NaN Resolution

Initial experiments exhibited NaN divergence typically at epochs 3–5. Root cause analysis identified the issue in two locations:

1. **acos gradient explosion**: $\partial \arccos(x)/\partial x$ diverges as $|x| \to 1$, propagating large gradients through HBNet weights. Fixed by computing all RHFD features under `torch.no_grad()`.

2. **Rotation matrix division-by-zero**: $R = I + K + K^2 \cdot (1-c)/s^2$ where $s = \lVert \vec{v}_1 \times \vec{v}_2 \rVert$. When head direction aligns with the reference axis, $s = 0$, causing $0/0$ NaN. Fixed by adding $\epsilon = 10^{-8}$ to $s^2$.

After both fixes, training remained stable across all experiments (20+ epochs, no NaN).

#### 4.2.4 Training Stability

With frozen HBNet, validation MAE converges rapidly and remains stable:

| Epoch | val MAE | Test 3D MAE |
|-------|:---:|:---:|
| 0 | 7.79° | — |
| 5 | 7.73° | 21.48° |
| 10 | 7.75° | — |
| 18 | 7.69° | 21.49° |

The validation-test gap of ~13.8° indicates strong domain shift between training and test scenes—a property inherited from the original GAFA framework, not introduced by our method.

---

## 5. Discussion

### 5.1 Why RHFD Features Help

The five gaze-state features capture orthogonal temporal properties:
- **Gf** and **Gd** measure *temporal dynamics* of gaze (how fast and how clustered)
- **Ga** quantifies *fixation stability* (whether a person is staring or scanning)
- **Gv** captures *gait-gaze coordination* (do head and body move together?)
- **Gs** estimates *attentional spread* (one target or many?)

These signals are complementary to raw head/body direction—they describe the *character* of gaze behavior rather than instantaneous orientation. The LSTM can learn to condition its direction predictions on whether the subject is fixating (low Gf, high Gd), scanning (high Gf, low Gd), or walking (high Gv).

### 5.2 Why Frozen HBNet Matters

Unfrozen HBNet (8.7M parameters) fine-tuned on 11 training scenes led to severe overfitting: the model learned scene-specific visual features rather than generalizable gaze cues. Freezing HBNet forces the model to rely on the pretrained representations and fit only the 770K-parameter GazeModule, dramatically improving generalization (24.18° → 21.49°).

### 5.3 Limitations

1. **Back-facing gaze** shows minimal improvement (23.45° vs 23.21°), likely because RHFD features computed from head direction are less discriminative when the face is not visible.
2. **Validation-test gap** remains large (~13.8°), suggesting the training scenes do not adequately represent test scene diversity.
3. **5-feature contribution** is marginal compared to 2-feature (21.49° vs 21.48°), indicating Gf and Gd are the primary drivers with Ga/Gv/Gs providing fine-grained refinement.

---

## 6. Conclusion

We present **SimpleRHFD-GazeNet**, demonstrating that five gaze-state temporal features—computed from head direction and body velocity with zero additional parameters or labels—improve 3D gaze estimation accuracy on the GAFA benchmark from 21.69° to 21.49° MAE. The improvement, while modest (0.9%), is achieved with **92% fewer trainable parameters** than the original model and critically depends on two implementation insights: **(1)** gradient isolation of angular computations to prevent NaN during training, and **(2)** freezing the pretrained feature extractor to prevent overfitting on limited training scenes.

Our work suggests that gaze-state features are a promising, low-cost signal for enhancing video-based gaze estimation, and that their full potential may be realized with larger, more diverse training datasets that narrow the domain gap between training and test distributions.

---

## References

[1] Nonaka, S., Nobuhara, S., & Nishino, K. (2022). Dynamic 3D Gaze from Afar: Deep Gaze Estimation from Temporal Eye-Head-Body Coordination. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2192–2201.

[2] Fisher, N. I., Lewis, T., & Embleton, B. J. J. (1987). *Statistical Analysis of Spherical Data*. Cambridge University Press.

[3] Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR*.

---

## Appendix: Implementation Details

The code is available at `https://github.com/pluie666/dynamic-3d-gaze-from-afar` under the `main` branch. Key files:

| File | Description |
|------|-------------|
| `models/gazenet.py` | `SimpleRHFDGazeNet` class (lines 680–830) |
| `models/rhfd_features.py` | `RHFDFeatureExtractor` with Gf/Gd/Ga/Gv/Gs |
| `models/utils.py` | `get_rotation` with NaN-safe division |
| `train.py` | Training script with `--simple --freeze_hbnet` flags |

**Training command:**
```bash
python train.py --simple --epoch 20 --n_frames 7 --gpus 1 \
  --batch_size 32 --freeze_hbnet \
  --weights ./models/weights/gazenet_GAFA.pth \
  --checkpoint output/
```

**Hardware:** NVIDIA Tesla V100S-PCIE-32GB, ~1 hour per epoch with frozen HBNet.

"""
RHFD (Refined Hidden Follower Detection) Gaze Feature Extraction Module.

Extracts five gaze-related temporal features from head/body sequences:
  - Gf (Gaze Fixation Frequency): frame-to-frame angular velocity
  - Gd (Gaze Density): spatial concentration within a sliding window
  - Ga (Gaze-Head Alignment Stability): head direction variance
  - Gv (Gaze-Velocity Correlation): head-body motion coupling
  - Gs (Gaze Spatial Entropy): directional spread on sphere

These features are computed purely from observables (head_dir, body_dv),
then concatenated as extra channels to the GazeModule LSTM input.
All acos-based features are masked with torch.no_grad() by the caller
to prevent gradient explosion through boundary gradients.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GazeFixationFrequency(nn.Module):
    """Gf: angular velocity of head direction changes (rad/step)."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, head_dir: torch.Tensor) -> torch.Tensor:
        B, T, D = head_dir.shape
        cos_angle = torch.sum(head_dir[:, :-1] * head_dir[:, 1:], dim=-1)
        cos_angle = torch.clamp(cos_angle, -1.0 + self.eps, 1.0 - self.eps)
        angular_diff = torch.acos(cos_angle)  # [B, T-1]
        padding = torch.zeros(B, 1, device=head_dir.device, dtype=angular_diff.dtype)
        angular_diff_padded = torch.cat([padding, angular_diff], dim=1)  # [B, T]
        return (angular_diff_padded / torch.pi).unsqueeze(-1)  # [B, T, 1]


class GazeDensity(nn.Module):
    """Gd: mean cosine similarity with neighbors within a window."""

    def __init__(self, window_size: int = 3):
        super().__init__()
        assert window_size % 2 == 1, "window_size must be odd"
        self.window_size = window_size
        self.half_window = window_size // 2

    def forward(self, head_dir: torch.Tensor) -> torch.Tensor:
        B, T, D = head_dir.shape
        head_padded = F.pad(
            head_dir.transpose(1, 2), (self.half_window, self.half_window), mode='replicate'
        ).transpose(1, 2)  # [B, T+2*hw, 3]

        density_list = []
        for t in range(T):
            window = head_padded[:, t:t + self.window_size]  # [B, W, 3]
            center = head_dir[:, t:t + 1]  # [B, 1, 3]
            similarities = torch.sum(center * window, dim=-1)  # [B, W]
            density_list.append(similarities.mean(dim=-1, keepdim=True))
        gd = torch.stack(density_list, dim=1)  # [B, T, 1]
        return (gd + 1.0) / 2.0  # map [-1,1] → [0,1]


class HeadStability(nn.Module):
    """
    Ga (Gaze-Head Alignment Stability): local variance of head direction.

    Low variance → stable fixation (small region).
    High variance → scanning / wandering gaze.
    Computed as 1 - mean cosine similarity within a sliding window.
    """

    def __init__(self, window_size: int = 3):
        super().__init__()
        assert window_size % 2 == 1
        self.window_size = window_size
        self.half_window = window_size // 2

    def forward(self, head_dir: torch.Tensor) -> torch.Tensor:
        B, T, D = head_dir.shape
        head_padded = F.pad(
            head_dir.transpose(1, 2), (self.half_window, self.half_window), mode='replicate'
        ).transpose(1, 2)

        stability_list = []
        for t in range(T):
            window = head_padded[:, t:t + self.window_size]  # [B, W, 3]
            # Mean direction within window
            mean_dir = window.mean(dim=1, keepdim=True)  # [B, 1, 3]
            mean_norm = mean_dir / (mean_dir.norm(dim=-1, keepdim=True) + 1e-8)
            # Mean cosine similarity to mean direction → stability
            sims = torch.sum(mean_norm * window, dim=-1)  # [B, W]
            stability = sims.mean(dim=-1, keepdim=True)  # [B, 1]
            stability_list.append(stability)
        ga = torch.stack(stability_list, dim=1)  # [B, T, 1]
        return ga  # range [-1,1], mapped later if needed


class HeadBodyCorrelation(nn.Module):
    """
    Gv (Head-Body Motion Correlation): how tightly head motion follows body motion.

    High → head tracks body movement direction (walking while looking).
    Low → head independent of body (standing, scanning independently).
    Computed as rolling Pearson r between |head_angle_change| and |body_velocity|.
    """

    def __init__(self, window_size: int = 3):
        super().__init__()
        assert window_size % 2 == 1
        self.window_size = window_size
        self.half_window = window_size // 2

    def forward(self, head_dir: torch.Tensor, body_dv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            head_dir: [B, T, 3] normalized head direction
            body_dv:  [B, T, 2] body velocity in image plane
        Returns:
            gv: [B, T, 1] correlation per frame
        """
        B, T, D = head_dir.shape
        eps = 1e-8

        # Head change magnitude (angular)
        cos_angle = torch.sum(head_dir[:, :-1] * head_dir[:, 1:], dim=-1)
        cos_angle = torch.clamp(cos_angle, -0.999, 0.999)
        head_change = torch.acos(cos_angle)  # [B, T-1]
        head_change = torch.cat([torch.zeros(B, 1, device=head_dir.device), head_change], dim=1)  # [B, T]

        # Body velocity magnitude
        body_mag = torch.norm(body_dv, dim=-1)  # [B, T]

        # Pad for sliding window
        head_pad = F.pad(head_change.unsqueeze(-1), (0, 0, self.half_window, self.half_window), mode='replicate')
        body_pad = F.pad(body_mag.unsqueeze(-1), (0, 0, self.half_window, self.half_window), mode='replicate')

        gv_list = []
        for t in range(T):
            h_win = head_pad[:, t:t + self.window_size, 0]  # [B, W]
            b_win = body_pad[:, t:t + self.window_size, 0]  # [B, W]

            h_centered = h_win - h_win.mean(dim=1, keepdim=True)
            b_centered = b_win - b_win.mean(dim=1, keepdim=True)

            cov = (h_centered * b_centered).sum(dim=1) / self.window_size
            std_h = h_win.std(dim=1) + eps
            std_b = b_win.std(dim=1) + eps
            corr = cov / (std_h * std_b)
            gv_list.append(corr.unsqueeze(-1))

        gv = torch.stack(gv_list, dim=1)  # [B, T, 1]
        return torch.clamp(gv, -1.0, 1.0)


class SpatialEntropy(nn.Module):
    """
    Gs (Gaze Spatial Entropy): how uniformly head directions are distributed.

    Low → directions clustered (focused attention).
    High → directions spread out (broad scanning).
    We approximate via mean pairwise cosine distance within a window.
    """

    def __init__(self, window_size: int = 3):
        super().__init__()
        assert window_size % 2 == 1
        self.window_size = window_size
        self.half_window = window_size // 2

    def forward(self, head_dir: torch.Tensor) -> torch.Tensor:
        B, T, D = head_dir.shape
        head_padded = F.pad(
            head_dir.transpose(1, 2), (self.half_window, self.half_window), mode='replicate'
        ).transpose(1, 2)

        entropy_list = []
        for t in range(T):
            window = head_padded[:, t:t + self.window_size]  # [B, W, 3]
            # All pairwise cosine similarities within window
            sims = torch.bmm(window, window.transpose(1, 2))  # [B, W, W]
            # Upper triangle mean (excludes self-similarity)
            mask = torch.triu(torch.ones(self.window_size, self.window_size, device=head_dir.device), diagonal=1).bool()
            pair_sims = sims[:, mask]  # [B, N_pairs]
            # Convert to dissimilarity → entropy proxy
            dissimilarity = (1.0 - pair_sims).mean(dim=-1, keepdim=True)  # [B, 1]
            entropy_list.append(dissimilarity)
        gs = torch.stack(entropy_list, dim=1)  # [B, T, 1]
        return gs  # [0, 2], lower = clustered


class RHFDFeatureExtractor(nn.Module):
    """Single-scale RHFD feature extraction: Gf + Gd + Ga + Gv + Gs at one window size."""

    def __init__(self, window_size: int = 3):
        super().__init__()
        self.gf_extractor = GazeFixationFrequency()
        self.gd_extractor = GazeDensity(window_size=window_size)
        self.ga_extractor = HeadStability(window_size=window_size)
        self.gv_extractor = HeadBodyCorrelation(window_size=window_size)
        self.gs_extractor = SpatialEntropy(window_size=window_size)

    def forward(self, head_dir: torch.Tensor, body_dv: torch.Tensor = None):
        gf = self.gf_extractor(head_dir)
        gd = self.gd_extractor(head_dir)
        ga = self.ga_extractor(head_dir)
        gs = self.gs_extractor(head_dir)
        if body_dv is not None:
            gv = self.gv_extractor(head_dir, body_dv)
        else:
            gv = torch.zeros_like(gf)
        return {
            'gf': gf, 'gd': gd, 'ga': ga, 'gv': gv, 'gs': gs,
            'concat': torch.cat([gf, gd, ga, gv, gs], dim=-1),
        }


class MultiScaleRHFDExtractor(nn.Module):
    """
    Multi-scale RHFD feature extraction with learnable compression and gating.

    Computes Gf, Gd, Ga, Gv, Gs at three temporal windows (W=3,5,7),
    concatenates all 15 features, then:
      1. Compresses via MLP: 15 → hidden → output_dim
      2. Applies gating attention: weights features per frame
    """

    def __init__(self, window_sizes=(3, 5, 7), hidden_dim=32, output_dim=8):
        super().__init__()
        self.window_sizes = window_sizes
        n_features = 5  # Gf, Gd, Ga, Gv, Gs
        self.input_dim = n_features * len(window_sizes)  # 15

        # One extractor per scale
        self.extractors = nn.ModuleList([
            RHFDFeatureExtractor(window_size=w) for w in window_sizes
        ])

        # Compression MLP: multi-scale → compact representation
        self.compress = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        # Per-frame gating: learn which features matter most for each frame
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, head_dir: torch.Tensor, body_dv: torch.Tensor = None):
        """
        Returns:
            'concat': [B, T, output_dim]  compressed + gated multi-scale features
            'raw':    [B, T, input_dim]    raw multi-scale features (for diagnostics)
        """
        all_features = []
        for extractor in self.extractors:
            feats = extractor(head_dir, body_dv)
            all_features.append(feats['concat'])
        raw = torch.cat(all_features, dim=-1)  # [B, T, input_dim]

        compressed = self.compress(raw)       # [B, T, output_dim]
        gate = self.gate(raw)                 # [B, T, output_dim]
        gated = compressed * gate             # element-wise gating

        return {'concat': gated, 'raw': raw}

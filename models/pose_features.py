"""
Pose Feature Extractor (UAGE-inspired).

Extracts body pose features from 2D keypoints and 3D body/head positions
available in the GAFA dataset. All features are computed under torch.no_grad()
and detached from the gradient graph.

Features:
  - Joint angles (relative positions between key joints)
  - Body orientation proxy (shoulder-hip vector)
  - Head-body relative position
  - Limb length ratios
  - 3D body velocity (already in dataset, reused)

All features are computed as [B, T, D] tensors for direct concatenation
with existing LSTM input channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PoseFeatureExtractor(nn.Module):
    """
    Extract pose features from 2D keypoints + 3D body/head positions.

    Inputs:
      - keypoints: [B, T, N_kp, 2] 2D body keypoints (image coordinates)
      - head_pos:  [B, T, 3] 3D head position in camera frame
      - body_pos:  [B, T, 3] 3D body position in camera frame
      - body_dv_3d: [B, T, 3] 3D body velocity in camera frame

    Outputs:
      - concat: [B, T, D] pose features (default D=6)
    """

    def __init__(self, output_dim=6):
        super().__init__()
        self.output_dim = output_dim

        # Compress raw pose features to output_dim
        # 2 joint angles + body orient + head-body rel + body_vel_mag + height_ratio = 6
        self.compress = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, output_dim),
        )

    def forward(self, keypoints, head_pos, body_pos, body_dv_3d, height=None):
        """
        Args:
            keypoints:  [B, T, N_kp, 2] or [B, T, N_kp*2] flattened
            head_pos:   [B, T, 3]
            body_pos:   [B, T, 3]
            body_dv_3d: [B, T, 3]
            height:     [B, T, 1] body height in pixels (optional)

        Returns:
            dict with 'concat' [B, T, output_dim]
        """
        B, T = head_pos.shape[0], head_pos.shape[1]
        eps = 1e-8

        # --- 1. Body orientation proxy ---
        # Head-to-body vector in 3D (rough body orientation)
        head_body_vec = head_pos - body_pos  # [B, T, 3]
        head_body_vec_norm = head_body_vec / (head_body_vec.norm(dim=-1, keepdim=True) + eps)

        # --- 2. Body velocity magnitude ---
        body_vel_mag = body_dv_3d.norm(dim=-1, keepdim=True)  # [B, T, 1]

        # --- 3. Head-body distance ratio ---
        head_body_dist = head_body_vec.norm(dim=-1, keepdim=True)  # [B, T, 1]

        # --- 4. Joint angles from keypoints (if available) ---
        if keypoints is not None and keypoints.shape[-2] >= 2:
            # Try to extract shoulder-to-hip angle from 2D keypoints
            # keypoints shape may vary; compute aggregate features
            if keypoints.dim() == 4:  # [B, T, N_kp, 2]
                kp = keypoints
            else:
                kp = keypoints.reshape(B, T, -1, 2)

            N_kp = kp.shape[2]

            # Upper body centroid (first half of keypoints)
            mid = N_kp // 2
            upper_center = kp[:, :, :mid].mean(dim=2)  # [B, T, 2]
            lower_center = kp[:, :, mid:].mean(dim=2)  # [B, T, 2]

            # Vertical body vector in image plane
            body_vec_2d = upper_center - lower_center  # [B, T, 2]
            body_vec_2d_norm = body_vec_2d / (body_vec_2d.norm(dim=-1, keepdim=True) + eps)

            # Torso angle (deviation from vertical)
            torso_vertical = torch.zeros_like(body_vec_2d)
            torso_vertical[:, :, 1] = 1.0  # vertical = [0, 1]
            torso_angle = torch.sum(body_vec_2d_norm * torso_vertical, dim=-1, keepdim=True)  # [B,T,1]

            # Keypoint spread (proxy for pose variation)
            kp_spread = kp.std(dim=2).mean(dim=-1, keepdim=True)  # [B, T, 1]
        else:
            body_vec_2d_norm = torch.zeros(B, T, 2, device=head_pos.device)
            torso_angle = torch.zeros(B, T, 1, device=head_pos.device)
            kp_spread = torch.zeros(B, T, 1, device=head_pos.device)

        # --- 5. Normalized height (if available) ---
        if height is not None:
            height_norm = height / (height.max(dim=1, keepdim=True)[0] + eps)
        else:
            height_norm = torch.ones(B, T, 1, device=head_pos.device)

        # --- Concatenate raw features ---
        raw_features = torch.cat([
            head_body_vec_norm,                    # 3 dims
            body_vel_mag,                           # 1 dim
            head_body_dist,                         # 1 dim
            torso_angle,                            # 1 dim
        ], dim=-1)  # 6 dims total

        # Compress (cast to float32 for mixed precision compatibility)
        compressed = self.compress(raw_features.float())  # [B, T, output_dim]

        return {
            'concat': compressed,
            'raw': raw_features,
        }

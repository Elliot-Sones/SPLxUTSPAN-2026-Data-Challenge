"""
End-to-end differentiable projectile model for SPLxUTSPAN.

Neural net predicts a soft release moment over frames, then uses a differentiable
ballistics layer to compute (angle, depth, left_right) at the rim plane z=10ft.

This is intentionally small/regularized to avoid overfitting on ~345 shots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TargetRanges:
    # Raw target ranges used by provided MinMax scalers.
    angle_min: float = 30.0
    angle_max: float = 60.0
    depth_min: float = -12.0
    depth_max: float = 30.0
    lr_min: float = -16.0
    lr_max: float = 16.0


def _scale_minmax(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return (x - lo) / (hi - lo)


def _safe_unit(v: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))


class PhysicsEndToEndModel(nn.Module):
    """
    Input:
      - x: (B, T, C) where C=207 (69 keypoints * 3 coords) in the dataset's column order
      - participant_id: (B,) int64 in [1..5]
    Output:
      - raw_preds: (B, 3) in raw units (deg, inches, inches)
      - scaled_preds: (B, 3) in [0,1] using the official scaler ranges
      - debug: dict (optional)
    """

    def __init__(
        self,
        *,
        num_keypoints: int,
        num_coords: int,
        n_participants: int = 5,
        conv_hidden: int = 32,
        dropout: float = 0.10,
        target_ranges: Optional[TargetRanges] = None,
        hoop_x_ft: float = 5.25,
        hoop_y_ft: float = -25.0,
        rim_z_ft: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_keypoints = int(num_keypoints)
        self.num_coords = int(num_coords)
        self.C = int(num_keypoints * num_coords)
        self.T = 240  # fixed in this competition

        self.ranges = target_ranges or TargetRanges()

        # Hoop geometry (feet).
        self.register_buffer("hoop_pos_ft", torch.tensor([hoop_x_ft, hoop_y_ft, rim_z_ft], dtype=torch.float32))
        self.rim_z_ft = float(rim_z_ft)

        # Participant embedding (small).
        self.part_emb = nn.Embedding(n_participants + 1, 8)  # ids 1..5

        # Tiny temporal encoder over the full 207-D coordinate channels.
        self.conv1 = nn.Conv1d(self.C, conv_hidden, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(conv_hidden, conv_hidden, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(conv_hidden, conv_hidden, kernel_size=3, padding=1)
        self.drop = nn.Dropout(dropout)

        # Per-frame attention logits; conditioned on participant embedding.
        self.att_fc = nn.Linear(conv_hidden + 8, 1)

        # Learn an effective "ball radius" for the fingertip->ball-center offset (feet).
        # Start near 0.39ft (basketball radius) but let the model adjust.
        self.log_r_eff = nn.Parameter(torch.tensor(-0.94))  # softplus(~0.39) ≈ 0.39

        # Learn an effective gravity in ft/frame^2 (time is likely normalized).
        # Start small; model will calibrate.
        self.log_g_frame = nn.Parameter(torch.tensor(-6.5))  # softplus ~ 0.0015

        # Small linear corrections to the proxy release state, from pooled features.
        # This lets the NN compensate for systematic proxy bias.
        self.state_fc = nn.Sequential(
            nn.Linear(conv_hidden + 8, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 6),
        )

    def _extract_keypoints(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,C) -> (B,T,K,3)
        B, T, C = x.shape
        assert C == self.C
        return x.view(B, T, self.num_keypoints, self.num_coords)

    def forward(
        self,
        x: torch.Tensor,
        participant_id: torch.Tensor,
        *,
        keypoint_index: Dict[str, int],
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        keypoint_index maps base name -> keypoint slot index (0..K-1) for the dataset order.
        """
        B, T, C = x.shape
        if T != self.T or C != self.C:
            raise ValueError(f"Expected x shape (B,240,{self.C}), got {tuple(x.shape)}")

        # Replace NaNs with 0 (assumes upstream standardization).
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Encode sequence.
        h = x.transpose(1, 2)  # (B,C,T)
        h = F.relu(self.conv1(h))
        h = self.drop(h)
        h = F.relu(self.conv2(h))
        h = self.drop(h)
        h = F.relu(self.conv3(h))
        h = self.drop(h)
        h = h.transpose(1, 2)  # (B,T,H)

        pe = self.part_emb(participant_id.clamp_min(0).clamp_max(self.part_emb.num_embeddings - 1))  # (B,8)
        pe_t = pe[:, None, :].expand(B, T, pe.shape[1])
        ht = torch.cat([h, pe_t], dim=-1)

        # Attention over frames (soft release time).
        logits = self.att_fc(ht).squeeze(-1)  # (B,T)
        # Encourage later frames very slightly (release is later); keep it weak.
        time_bias = torch.linspace(-0.2, 0.2, T, device=x.device, dtype=logits.dtype)
        logits = logits + time_bias[None, :]
        w = torch.softmax(logits, dim=1)  # (B,T)

        # Build a differentiable ball center proxy from hand geometry.
        xk = self._extract_keypoints(x)  # (B,T,K,3)

        def kp(name: str) -> torch.Tensor:
            return xk[:, :, keypoint_index[name], :]  # (B,T,3)

        wrist = kp("right_wrist")
        idx_tip = kp("right_second_finger_distal")
        mid_tip = kp("right_third_finger_distal")
        ring_tip = kp("right_fourth_finger_distal")
        pinky_tip = kp("right_fifth_finger_distal")

        centroid = (idx_tip + mid_tip + ring_tip) / 3.0  # surface-ish point
        normal = torch.cross(idx_tip - wrist, pinky_tip - wrist, dim=-1)
        normal = _safe_unit(normal)

        # Ensure normal points toward wrist (ball center between fingers and wrist).
        to_wrist = wrist - centroid
        sign = torch.where((normal * to_wrist).sum(dim=-1, keepdim=True) < 0, -1.0, 1.0)
        normal = normal * sign

        r_eff = F.softplus(self.log_r_eff)  # feet
        center = centroid + normal * r_eff  # (B,T,3)

        # Velocity proxy in ft/frame (central differences).
        v = torch.zeros_like(center)
        v[:, 1:-1] = 0.5 * (center[:, 2:] - center[:, :-2])
        v[:, 0] = center[:, 1] - center[:, 0]
        v[:, -1] = center[:, -1] - center[:, -2]

        # Soft release state from attention.
        w3 = w[:, :, None]
        p0 = (w3 * center).sum(dim=1)  # (B,3) ft
        v0 = (w3 * v).sum(dim=1)  # (B,3) ft/frame

        # Small learned correction to (p0,v0) from pooled features.
        pooled = (w[:, :, None] * h).sum(dim=1)  # (B,H)
        corr_in = torch.cat([pooled, pe], dim=-1)
        delta = self.state_fc(corr_in)  # (B,6)
        dp = delta[:, :3]
        dv = delta[:, 3:]
        p0 = p0 + 0.25 * dp
        v0 = v0 + 0.25 * dv

        g_frame = F.softplus(self.log_g_frame) + 1e-6  # ft/frame^2

        # Solve for t (in frames) when z(t)=rim_z (quadratic).
        z0 = p0[:, 2]
        vz0 = v0[:, 2]
        a = -0.5 * g_frame
        b = vz0
        c = z0 - self.rim_z_ft
        disc = (b * b - 4.0 * a * c).clamp_min(0.0)
        sd = torch.sqrt(disc)
        t1 = (-b + sd) / (2.0 * a)
        t2 = (-b - sd) / (2.0 * a)
        # Choose the larger positive root; if both invalid, fall back to positive one; else clamp.
        t = torch.maximum(t1, t2)
        t_alt = torch.minimum(t1, t2)
        t = torch.where(t > 0, t, t_alt)
        t = torch.where(t > 0, t, torch.full_like(t, 30.0))  # fallback
        t = t.clamp(1.0, 300.0)

        # Position/velocity at rim plane.
        x_rim = p0[:, 0] + v0[:, 0] * t
        y_rim = p0[:, 1] + v0[:, 1] * t
        vz_rim = vz0 - g_frame * t
        vh = torch.sqrt(v0[:, 0] ** 2 + v0[:, 1] ** 2).clamp_min(1e-6)

        angle_deg = torch.rad2deg(torch.atan2(vz_rim.abs(), vh))
        depth_in = (y_rim - self.hoop_pos_ft[1]) * 12.0
        lr_in = (x_rim - self.hoop_pos_ft[0]) * 12.0

        raw = torch.stack([angle_deg, depth_in, lr_in], dim=1)

        # Scale to [0,1] using official ranges.
        scaled = torch.stack(
            [
                _scale_minmax(angle_deg, self.ranges.angle_min, self.ranges.angle_max),
                _scale_minmax(depth_in, self.ranges.depth_min, self.ranges.depth_max),
                _scale_minmax(lr_in, self.ranges.lr_min, self.ranges.lr_max),
            ],
            dim=1,
        ).clamp(0.0, 1.0)

        out: Dict[str, torch.Tensor] = {"raw": raw, "scaled": scaled}
        if return_debug:
            out["release_weights"] = w
            out["p0"] = p0
            out["v0"] = v0
            out["g_frame"] = g_frame[None].expand(B)
            out["t_frames"] = t
            out["r_eff_ft"] = r_eff[None].expand(B)
        return out


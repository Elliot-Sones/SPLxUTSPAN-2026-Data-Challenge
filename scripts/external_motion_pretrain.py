"""
External Motion Pretraining Pipeline

Phase 1: Generate synthetic + parse real motion data
Phase 2: Pretrain a temporal encoder via masked frame prediction
Phase 3: Extract features for basketball shots
Phase 4: Predict targets using per-example locally weighted Ridge
Phase 5: Generate submissions blended with Sub 784

The hypothesis: a temporal encoder pretrained on thousands of motion sequences
can learn temporal dynamics that our 345-sample Ridge model cannot capture.
"""

import json
import time
import fcntl
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
FEET_TO_METERS = 0.3048

TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

# Common body joints between our data and external datasets
# These are the 17 major body joints present in almost all motion capture
COMMON_JOINTS = [
    'nose', 'neck',
    'right_shoulder', 'right_elbow', 'right_wrist',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle',
    'mid_hip',
    'right_big_toe', 'left_big_toe',
]  # 17 joints x 3 coords = 51 channels

N_COMMON_JOINTS = len(COMMON_JOINTS)
N_CHANNELS = N_COMMON_JOINTS * 3  # 51
N_FRAMES = 240


def get_next_submission_number():
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = [int(fp.stem.split('_')[1]) for fp in existing
                    if fp.stem.split('_')[1].isdigit()]
            next_num = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def safe_savgol(x, window, polyorder, **kwargs):
    x = np.asarray(x, dtype=np.float64)
    bad = np.isnan(x) | np.isinf(x)
    if np.all(bad):
        return np.zeros_like(x)
    if np.any(bad):
        good = ~bad
        x[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x[good])
    return savgol_filter(x, window, polyorder, **kwargs)


# ==============================================================
# PHASE 1: Synthetic Motion Data Generation
# ==============================================================

def generate_synthetic_motions(n_sequences=5000, n_frames=240, seed=42):
    """
    Generate synthetic human motion sequences with physically plausible dynamics.

    Uses parametric motion models:
    1. Oscillatory joint motions (walking, running patterns)
    2. Ballistic arm motions (throwing, reaching patterns)
    3. Smooth random trajectories with joint constraints

    Returns: (n_sequences, n_frames, n_joints, 3) array
    """
    rng = np.random.RandomState(seed)
    n_joints = N_COMMON_JOINTS

    all_sequences = np.zeros((n_sequences, n_frames, n_joints, 3), dtype=np.float32)
    t = np.linspace(0, 4.0, n_frames)  # 4 seconds at 60fps

    for seq_i in range(n_sequences):
        # Random motion type
        motion_type = rng.choice(['oscillatory', 'ballistic', 'smooth', 'basketball_like'],
                                  p=[0.3, 0.2, 0.2, 0.3])

        # Base skeleton - approximate proportions in feet
        base_positions = np.zeros((n_joints, 3), dtype=np.float32)
        # Standing upright, facing forward (negative Y)
        base_positions[0] = [0, 0, 5.8]     # nose
        base_positions[1] = [0, 0, 5.5]     # neck
        base_positions[2] = [-0.7, 0, 5.3]  # right_shoulder (our right = negative X)
        base_positions[3] = [-0.7, 0, 4.3]  # right_elbow
        base_positions[4] = [-0.7, 0, 3.5]  # right_wrist
        base_positions[5] = [0.7, 0, 5.3]   # left_shoulder
        base_positions[6] = [0.7, 0, 4.3]   # left_elbow
        base_positions[7] = [0.7, 0, 3.5]   # left_wrist
        base_positions[8] = [-0.4, 0, 3.0]  # right_hip
        base_positions[9] = [-0.4, 0, 1.7]  # right_knee
        base_positions[10] = [-0.4, 0, 0.3] # right_ankle
        base_positions[11] = [0.4, 0, 3.0]  # left_hip
        base_positions[12] = [0.4, 0, 1.7]  # left_knee
        base_positions[13] = [0.4, 0, 0.3]  # left_ankle
        base_positions[14] = [0, 0, 3.0]    # mid_hip
        base_positions[15] = [-0.4, 0, 0.0] # right_big_toe
        base_positions[16] = [0.4, 0, 0.0]  # left_big_toe

        # Add person-level variation
        height_scale = rng.uniform(0.85, 1.15)
        base_positions *= height_scale

        seq = np.zeros((n_frames, n_joints, 3), dtype=np.float32)

        if motion_type == 'oscillatory':
            # Walking/running - periodic limb motions
            freq = rng.uniform(0.5, 3.0)
            phase = rng.uniform(0, 2*np.pi)
            amplitude = rng.uniform(0.2, 1.0)

            for f in range(n_frames):
                seq[f] = base_positions.copy()
                # Legs oscillate in opposition
                leg_phase = 2*np.pi*freq*t[f] + phase
                # Right leg
                seq[f, 9, 0] += amplitude * 0.3 * np.sin(leg_phase)  # knee lateral
                seq[f, 9, 2] -= amplitude * 0.2 * (1 - np.cos(leg_phase))  # knee up
                seq[f, 10, 0] += amplitude * 0.5 * np.sin(leg_phase)  # ankle forward
                # Left leg (opposite phase)
                seq[f, 12, 0] += amplitude * 0.3 * np.sin(leg_phase + np.pi)
                seq[f, 12, 2] -= amplitude * 0.2 * (1 - np.cos(leg_phase + np.pi))
                seq[f, 13, 0] += amplitude * 0.5 * np.sin(leg_phase + np.pi)
                # Arms swing
                seq[f, 3, 0] += amplitude * 0.4 * np.sin(leg_phase + np.pi)  # right elbow
                seq[f, 4, 0] += amplitude * 0.6 * np.sin(leg_phase + np.pi)  # right wrist
                seq[f, 6, 0] += amplitude * 0.4 * np.sin(leg_phase)  # left elbow
                seq[f, 7, 0] += amplitude * 0.6 * np.sin(leg_phase)  # left wrist
                # Torso sway
                seq[f, 14, 0] += 0.05 * np.sin(2*leg_phase)
                seq[f, 1, 0] += 0.03 * np.sin(2*leg_phase)

        elif motion_type == 'ballistic':
            # Throwing/reaching - arm extends then follows through
            throw_start = rng.randint(60, 150)
            throw_duration = rng.randint(15, 40)
            arm_side = rng.choice([0, 1])  # 0=right, 1=left

            for f in range(n_frames):
                seq[f] = base_positions.copy()
                if f >= throw_start and f < throw_start + throw_duration:
                    progress = (f - throw_start) / throw_duration
                    # Smooth acceleration profile
                    ease = np.sin(progress * np.pi)

                    if arm_side == 0:  # right arm
                        # Elbow extends upward and forward
                        seq[f, 3, 2] += 1.5 * ease  # elbow up
                        seq[f, 3, 1] -= 0.5 * ease  # elbow forward
                        # Wrist extends further
                        seq[f, 4, 2] += 2.5 * ease
                        seq[f, 4, 1] -= 1.0 * ease
                        # Body leans slightly
                        seq[f, 14, 2] += 0.1 * ease
                    else:  # left arm
                        seq[f, 6, 2] += 1.5 * ease
                        seq[f, 6, 1] -= 0.5 * ease
                        seq[f, 7, 2] += 2.5 * ease
                        seq[f, 7, 1] -= 1.0 * ease
                        seq[f, 14, 2] += 0.1 * ease
                elif f >= throw_start + throw_duration:
                    # Follow through - deceleration
                    progress = min(1.0, (f - throw_start - throw_duration) / 30.0)
                    decay = np.exp(-3 * progress)
                    if arm_side == 0:
                        seq[f, 3, 2] += 1.0 * decay
                        seq[f, 4, 2] += 1.5 * decay
                        seq[f, 4, 1] -= 1.5 * (1 - decay)  # arm comes down forward
                    else:
                        seq[f, 6, 2] += 1.0 * decay
                        seq[f, 7, 2] += 1.5 * decay
                        seq[f, 7, 1] -= 1.5 * (1 - decay)

        elif motion_type == 'basketball_like':
            # Simulate basketball shooting motion
            setup_end = rng.randint(80, 130)
            release = rng.randint(setup_end + 10, setup_end + 30)
            follow_through_end = min(release + 30, n_frames)

            # Player crouches, rises, extends arm, releases
            for f in range(n_frames):
                seq[f] = base_positions.copy()

                if f < setup_end:
                    # Setup/gather phase - slight crouch
                    progress = f / setup_end
                    crouch = 0.3 * np.sin(progress * np.pi * 0.5)
                    seq[f, 8:, 2] -= crouch  # lower body crouches
                    seq[f, :8, 2] -= crouch * 0.8  # upper body follows
                    # Ball at chest height
                    seq[f, 4, 2] += 1.0  # right wrist up
                    seq[f, 4, 1] -= 0.3  # slightly forward
                    seq[f, 7, 2] += 0.8  # left wrist up (guide hand)
                    seq[f, 7, 1] -= 0.2

                elif f < release:
                    # Rising/shooting phase
                    progress = (f - setup_end) / (release - setup_end)
                    rise = progress ** 0.5  # quick rise

                    # Body rises
                    seq[f, :, 2] += 0.3 * rise  # body goes up
                    # Shooting arm extends
                    seq[f, 3, 2] += 1.0 + 1.5 * rise  # elbow way up
                    seq[f, 3, 1] -= 0.3 + 0.5 * rise  # elbow forward
                    seq[f, 4, 2] += 1.5 + 2.5 * rise  # wrist way up
                    seq[f, 4, 1] -= 0.5 + 1.0 * rise  # wrist forward
                    # Guide hand separates
                    seq[f, 7, 2] += 0.8 + 0.5 * rise
                    seq[f, 7, 1] -= 0.2 - 0.1 * rise  # pulls back slightly
                    # Knees extend
                    seq[f, 9, 2] += 0.2 * rise
                    seq[f, 12, 2] += 0.2 * rise

                elif f < follow_through_end:
                    # Follow through
                    progress = (f - release) / max(1, follow_through_end - release)
                    decay = np.exp(-2 * progress)

                    seq[f, :, 2] += 0.3 * decay
                    seq[f, 3, 2] += 2.5 * decay
                    seq[f, 3, 1] -= 0.8 * decay
                    seq[f, 4, 2] += 3.0 * decay + 1.0 * (1 - decay)
                    seq[f, 4, 1] -= 1.5  # wrist stays forward/down
                    seq[f, 4, 2] -= 1.0 * (1 - decay)  # wrist drops in follow through
                    # Wrist snaps
                    seq[f, 7, 2] += 0.5 * decay
                else:
                    # Return to neutral
                    progress = min(1.0, (f - follow_through_end) / 30.0)

        else:  # smooth
            # Random smooth trajectories for each joint
            n_basis = rng.randint(3, 8)
            for j in range(n_joints):
                for c in range(3):
                    # Random Fourier basis
                    signal = np.zeros(n_frames)
                    for b in range(n_basis):
                        freq = rng.uniform(0.1, 2.0)
                        phase = rng.uniform(0, 2*np.pi)
                        amp = rng.uniform(0.05, 0.5) / (b + 1)
                        signal += amp * np.sin(2*np.pi*freq*t + phase)
                    seq[:, j, c] = base_positions[j, c] + signal

        # Add measurement noise (simulate real mocap noise)
        noise_level = rng.uniform(0.01, 0.05)
        seq += rng.randn(*seq.shape).astype(np.float32) * noise_level

        all_sequences[seq_i] = seq

    return all_sequences


def parse_bvh_to_joint_positions(bvh_path, target_frames=240):
    """
    Parse a BVH file and extract joint positions.
    Returns: (n_frames, n_joints, 3) or None if parsing fails.

    BVH format: hierarchy section (joint tree), then MOTION section with frame data.
    The root has 6 channels (3 position + 3 rotation), other joints have 3 rotation channels.
    We need forward kinematics to get positions from rotations.

    For simplicity, we extract the subset of joints we can and map to our common joint indices.
    """
    try:
        with open(bvh_path, 'r') as f:
            content = f.read()
    except Exception:
        return None

    lines = content.strip().split('\n')

    # Find MOTION section
    motion_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('MOTION'):
            motion_idx = i
            break

    if motion_idx is None:
        return None

    # Parse frame count and time
    n_frames_line = lines[motion_idx + 1].strip()
    n_frames = int(n_frames_line.split(':')[1].strip())

    # Parse hierarchy to get joint names and their channel counts
    joint_names = []
    joint_channels = []
    in_hierarchy = True
    channel_offset = 0

    for i in range(motion_idx):
        line = lines[i].strip()
        if line.startswith('ROOT') or line.startswith('JOINT'):
            name = line.split()[-1]
            joint_names.append(name)
        elif line.startswith('End Site'):
            joint_names.append('EndSite_' + str(len(joint_names)))
        elif line.startswith('CHANNELS'):
            parts = line.split()
            n_ch = int(parts[1])
            joint_channels.append(n_ch)

    total_channels = sum(joint_channels)

    # Parse motion data
    frame_data = []
    for i in range(motion_idx + 3, min(motion_idx + 3 + n_frames, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        values = [float(v) for v in line.split()]
        if len(values) >= total_channels:
            frame_data.append(values[:total_channels])

    if len(frame_data) < 10:
        return None

    frame_data = np.array(frame_data, dtype=np.float32)
    n_actual_frames = len(frame_data)

    # For BVH, root position is in first 3 channels
    # Other joints need FK. For simplicity, we'll extract root position
    # and use channel values as proxy positions (this is approximate but
    # captures temporal dynamics which is what we need for pretraining)

    # Map to our joint indices - extract positions from channel data
    # Root position is channels 0,1,2
    n_usable_joints = min(len(joint_names), len(joint_channels))
    positions = np.zeros((n_actual_frames, n_usable_joints, 3), dtype=np.float32)

    ch_offset = 0
    for j_idx in range(n_usable_joints):
        n_ch = joint_channels[j_idx] if j_idx < len(joint_channels) else 3
        if j_idx == 0 and n_ch >= 6:
            # Root: first 3 are position, next 3 are rotation
            positions[:, j_idx, 0] = frame_data[:, ch_offset]
            positions[:, j_idx, 1] = frame_data[:, ch_offset + 1]
            positions[:, j_idx, 2] = frame_data[:, ch_offset + 2]
        else:
            # Other joints: rotation channels (use as proxy for temporal dynamics)
            # Scale rotations to position-like values
            for c in range(min(3, n_ch)):
                positions[:, j_idx, c] = frame_data[:, ch_offset + c] / 180.0
        ch_offset += n_ch

    # Resample to target frame count
    if n_actual_frames != target_frames:
        indices = np.linspace(0, n_actual_frames - 1, target_frames).astype(int)
        positions = positions[indices]

    # Only keep first 17 joints max
    n_keep = min(N_COMMON_JOINTS, n_usable_joints)
    result = np.zeros((target_frames, N_COMMON_JOINTS, 3), dtype=np.float32)
    result[:, :n_keep, :] = positions[:, :n_keep, :]

    return result


# ==============================================================
# PHASE 2: Temporal Encoder Architecture
# ==============================================================

class TemporalEncoder(nn.Module):
    """
    Small 1D CNN encoder for temporal motion sequences.
    Input: (batch, n_frames, n_channels) where n_channels = n_joints * 3
    Output: (batch, embed_dim) global embedding

    Architecture is deliberately small (50K-200K params) to avoid
    overfitting when fine-tuning on 345 samples.
    """
    def __init__(self, n_channels=51, embed_dim=64, n_frames=240):
        super().__init__()

        # Temporal convolution blocks with increasing receptive field
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, embed_dim, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm1d(embed_dim)

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Decoder for masked frame prediction (pretraining)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_channels),
        )

        # Frame-level output for masked prediction
        self.frame_decoder = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, n_channels, kernel_size=3, padding=1),
        )

    def encode(self, x):
        """
        x: (batch, n_frames, n_channels)
        Returns: (batch, embed_dim) global embedding
        """
        x = x.transpose(1, 2)  # (batch, n_channels, n_frames)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))

        # Global average pooling
        embedding = self.pool(x).squeeze(-1)  # (batch, embed_dim)
        return embedding

    def decode_frames(self, x):
        """
        x: (batch, n_frames, n_channels)
        Returns: (batch, n_frames, n_channels) reconstructed frames
        """
        x_t = x.transpose(1, 2)  # (batch, n_channels, n_frames)
        h = self.relu(self.bn1(self.conv1(x_t)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.relu(self.bn3(self.conv3(h)))
        h = self.relu(self.bn4(self.conv4(h)))
        h = self.relu(self.bn5(self.conv5(h)))

        # Decode back to frame level
        out = self.frame_decoder(h)  # (batch, n_channels, ~n_frames)

        # Adjust size to match input
        if out.shape[2] != x.shape[1]:
            out = nn.functional.interpolate(out, size=x.shape[1], mode='linear', align_corners=False)

        return out.transpose(1, 2)  # (batch, n_frames, n_channels)

    def forward(self, x, mask=None):
        """
        x: (batch, n_frames, n_channels) - input with some frames masked
        mask: (batch, n_frames) - True for masked frames
        Returns: reconstruction of masked frames
        """
        return self.decode_frames(x)


class TemporalPretrainer:
    """Pretrain TemporalEncoder via masked frame prediction."""

    def __init__(self, n_channels=51, embed_dim=64, device='cpu'):
        self.device = device
        self.model = TemporalEncoder(n_channels=n_channels, embed_dim=embed_dim).to(device)
        self.n_channels = n_channels
        self.embed_dim = embed_dim

    def pretrain(self, sequences, n_epochs=30, batch_size=64, lr=1e-3,
                 mask_ratio=0.15, verbose=True):
        """
        Pretrain on synthetic/external motion sequences.

        sequences: (n_seq, n_frames, n_joints, 3) numpy array
        """
        n_seq, n_frames, n_joints, n_coords = sequences.shape
        # Flatten joints x coords -> channels
        X = sequences.reshape(n_seq, n_frames, n_joints * n_coords)

        # Normalize per-channel
        self.channel_means = X.mean(axis=(0, 1))
        self.channel_stds = X.std(axis=(0, 1)) + 1e-8
        X = (X - self.channel_means) / self.channel_stds

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        self.model.train()

        for epoch in range(n_epochs):
            # Shuffle
            perm = torch.randperm(n_seq)
            total_loss = 0
            n_batches = 0

            for i in range(0, n_seq, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch = X_tensor[batch_idx]  # (B, T, C)

                # Create mask
                B, T, C = batch.shape
                mask = torch.rand(B, T, device=self.device) < mask_ratio
                # Don't mask too many frames - ensure at least 50% are visible

                # Masked input: replace masked frames with zeros
                masked_input = batch.clone()
                mask_expanded = mask.unsqueeze(-1).expand_as(batch)
                masked_input[mask_expanded] = 0.0

                # Forward pass - predict all frames
                pred = self.model(masked_input, mask)

                # Loss: only on masked frames
                loss = ((pred - batch) ** 2 * mask_expanded.float()).sum() / mask_expanded.float().sum().clamp(min=1)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()

            if verbose and (epoch + 1) % 5 == 0:
                avg_loss = total_loss / max(n_batches, 1)
                print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.6f}")

        return total_loss / max(n_batches, 1)

    def extract_features(self, sequences):
        """
        Extract embeddings from sequences.

        sequences: (n_seq, n_frames, n_joints, 3) numpy array
        Returns: (n_seq, embed_dim) numpy array
        """
        n_seq, n_frames, n_joints, n_coords = sequences.shape
        X = sequences.reshape(n_seq, n_frames, n_joints * n_coords)

        # Normalize with pretraining stats
        X = (X - self.channel_means) / self.channel_stds

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.encode(X_tensor)

        return embeddings.cpu().numpy()


# ==============================================================
# DATA LOADING AND JOINT EXTRACTION
# ==============================================================

def load_data():
    """Load and parse all data."""
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    keypoint_cols = [c for c in train_df.columns if c not in meta_cols]
    n_kp_cols = len(keypoint_cols)

    kp_names = []
    for col in keypoint_cols:
        if col.endswith("_x"):
            kp_names.append(col[:-2])
    kp_index = {name: i for i, name in enumerate(kp_names)}

    def process(df, is_train=True):
        n = len(df)
        n_kp = len(kp_names)
        X_3d = np.zeros((n, 240, n_kp, 3), dtype=np.float32)
        X_raw = np.zeros((n, n_kp_cols * 240), dtype=np.float32)
        ids, pids, targets = [], [], []

        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                X_3d[idx, :, col_i // 3, col_i % 3] = arr
                X_raw[idx, col_i * 240:(col_i + 1) * 240] = arr
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)}")

        X_raw = np.nan_to_num(X_raw, nan=0.0)
        result = {'X_3d': X_3d, 'X_raw': X_raw, 'pids': np.array(pids),
                  'ids': np.array(ids), 'kp_names': kp_names, 'kp_index': kp_index}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    return process(train_df, True), process(test_df, False)


def extract_common_joints(X_3d, kp_index):
    """
    Extract common body joints from full keypoint set.
    X_3d: (n_shots, 240, n_keypoints, 3)
    Returns: (n_shots, 240, N_COMMON_JOINTS, 3)
    """
    n_shots = X_3d.shape[0]
    result = np.zeros((n_shots, 240, N_COMMON_JOINTS, 3), dtype=np.float32)

    for j_idx, j_name in enumerate(COMMON_JOINTS):
        if j_name in kp_index:
            result[:, :, j_idx, :] = X_3d[:, :, kp_index[j_name], :]

    # Fill NaN with interpolation per shot per joint per coord
    for i in range(n_shots):
        for j in range(N_COMMON_JOINTS):
            for c in range(3):
                vals = result[i, :, j, c]
                bad = np.isnan(vals) | np.isinf(vals)
                if np.all(bad):
                    result[i, :, j, c] = 0.0
                elif np.any(bad):
                    good = ~bad
                    vals[bad] = np.interp(np.where(bad)[0], np.where(good)[0], vals[good])
                    result[i, :, j, c] = vals

    return result


# ==============================================================
# FEATURE EXTRACTION (from per_example_pipeline.py)
# ==============================================================

def compute_hoop_transform(ts_3d, kp_index):
    mid_hip_idx = kp_index.get('mid_hip', 0)
    player_pos = ts_3d[120, mid_hip_idx, :].copy()
    player_pos[2] = 0

    forward = HOOP_POS[:2] - player_pos[:2]
    fn = np.linalg.norm(forward)
    if fn > 1e-6:
        forward /= fn
    else:
        forward = np.array([0.0, -1.0])
    lateral = np.array([-forward[1], forward[0]])

    R = np.eye(3, dtype=np.float32)
    R[0, 0] = forward[0]; R[0, 1] = forward[1]
    R[1, 0] = lateral[0]; R[1, 1] = lateral[1]

    centered = ts_3d - player_pos.reshape(1, 1, 3)
    return np.einsum('ij,fkj->fki', R, centered)


def detect_release_frame(ts_3d, kp_index):
    rw_idx = kp_index.get('right_wrist')
    if rw_idx is None:
        return 120
    wrist_traj = ts_3d[:, rw_idx, :].copy()
    for ax in range(3):
        vals = wrist_traj[:, ax]
        bad = np.isnan(vals) | np.isinf(vals)
        if np.all(bad):
            return 120
        if np.any(bad):
            good = ~bad
            vals[bad] = np.interp(np.where(bad)[0], np.where(good)[0], vals[good])
        wrist_traj[:, ax] = vals

    wrist_z_smooth = safe_savgol(wrist_traj[:, 2], 11, 3)
    wrist_peak = 80 + np.argmax(wrist_z_smooth[80:200])

    ft_keys = ['right_second_finger_distal', 'right_third_finger_distal',
               'right_fourth_finger_distal']
    ft_trajs = [ts_3d[:, kp_index[k], :] for k in ft_keys if k in kp_index]
    if ft_trajs:
        ft_center = np.nanmean(ft_trajs, axis=0)
        for ax in range(3):
            ft_center[:, ax] = safe_savgol(ft_center[:, ax], 15, 3)
        ball = wrist_traj + 0.6 * (ft_center - wrist_traj)
    else:
        ball = wrist_traj.copy()
    for ax in range(3):
        ball[:, ax] = safe_savgol(ball[:, ax], 11, 3)

    vel = np.zeros_like(ball * FEET_TO_METERS)
    for ax in range(3):
        vel[:, ax] = safe_savgol(ball[:, ax] * FEET_TO_METERS, 9, 3, deriv=1, delta=DT)
    speed = np.linalg.norm(vel, axis=1)

    s, e = max(80, wrist_peak - 40), min(wrist_peak + 5, 200)
    return int(np.clip(s + np.argmax(speed[s:e]), 80, 200))


def extract_features(ts_3d, ts_hr, kp_index, release_frame, frame):
    """Extract compact feature set at a specific frame."""
    f = int(np.clip(frame, 0, 239))
    feats = []

    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder',
                  'left_wrist', 'left_shoulder',
                  'right_hip', 'left_hip', 'mid_hip',
                  'right_knee', 'left_knee', 'neck', 'nose']

    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 6)
            continue
        for coord in range(3):
            feats.append(ts_hr[f, idx, coord])
            vel = np.gradient(ts_hr[:, idx, coord], DT)
            feats.append(vel[f])

    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 9)
            continue
        for coord in range(3):
            series = ts_hr[:, idx, coord]
            feats.append(np.nanmean(series))
            feats.append(np.nanstd(series))
            feats.append(np.nanmax(series) - np.nanmin(series))

    rw = kp_index.get('right_wrist')
    re = kp_index.get('right_elbow')
    rs = kp_index.get('right_shoulder')
    if all(i is not None for i in [rw, re, rs]):
        feats.append(ts_hr[f, rw, 0] - ts_hr[f, rs, 0])
        feats.append(ts_hr[f, rw, 1] - ts_hr[f, rs, 1])
        feats.append(ts_hr[f, rw, 2] - ts_hr[f, rs, 2])
        ua = ts_3d[f, re] - ts_3d[f, rs]
        fa = ts_3d[f, rw] - ts_3d[f, re]
        ua_n, fa_n = np.linalg.norm(ua), np.linalg.norm(fa)
        if ua_n > 1e-6 and fa_n > 1e-6:
            feats.append(np.degrees(np.arccos(np.clip(np.dot(-ua, fa) / (ua_n * fa_n), -1, 1))))
        else:
            feats.append(90.0)
        for coord in range(3):
            vel = np.gradient(ts_hr[:, rw, coord], DT)
            feats.append(vel[f])
    else:
        feats.extend([0.0] * 7)

    rh, lh = kp_index.get('right_hip'), kp_index.get('left_hip')
    ls = kp_index.get('left_shoulder')
    if rh is not None and lh is not None:
        feats.append(ts_hr[f, rh, 1] - ts_hr[f, lh, 1])
        feats.append(ts_hr[f, rh, 0] - ts_hr[f, lh, 0])
    else:
        feats.extend([0.0, 0.0])
    if rs is not None and ls is not None:
        feats.append(ts_hr[f, rs, 1] - ts_hr[f, ls, 1])
    else:
        feats.append(0.0)

    lw = kp_index.get('left_wrist')
    if lw is not None and rw is not None:
        feats.append(ts_hr[f, lw, 1] - ts_hr[f, rw, 1])
    else:
        feats.append(0.0)

    feats.append(release_frame)

    if rw is not None:
        for coord in range(3):
            series = ts_hr[:, rw, coord]
            vel = np.gradient(series, DT)
            feats.append(np.nanmean(series[140:180]))
            feats.append(np.nanmax(vel[140:180]))
    else:
        feats.extend([0.0] * 6)

    return np.array(feats, dtype=np.float32)


def extract_all_features(data, target):
    """Extract features for all shots."""
    frame = TARGET_FRAMES[target]
    n = len(data['pids'])
    kp_index = data['kp_index']

    all_feats = []
    release_frames = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        rf = detect_release_frame(ts_3d, kp_index)
        release_frames.append(rf)
        feats = extract_features(ts_3d, ts_hr, kp_index, rf, frame)
        all_feats.append(feats)

    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(release_frames)


def augment_with_pls(X_train, y_raw_train, pids_train, X_test, pids_test, X_raw_train, X_raw_test):
    """Add PLS components to feature matrices."""
    unique_pids = sorted(np.unique(pids_train))
    max_nc = 15
    pls_train = np.zeros((len(pids_train), max_nc), dtype=np.float32)
    pls_test = np.zeros((len(pids_test), max_nc), dtype=np.float32)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        n_p = tr_mask.sum()

        scaler = StandardScaler()
        raw_tr = scaler.fit_transform(X_raw_train[tr_mask])
        raw_te = scaler.transform(X_raw_test[te_mask])

        nc = min(max_nc, n_p - n_p // 5 - 1)
        nc = max(3, nc)

        best_nc, best_mse = 3, float('inf')
        for c in [3, 5, 8, 10, 15]:
            if c > nc:
                break
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mses = []
            for tr_idx, val_idx in kf.split(raw_tr):
                pls = PLSRegression(n_components=c)
                pls.fit(raw_tr[tr_idx], y_raw_train[tr_mask][tr_idx])
                pred = pls.predict(raw_tr[val_idx]).flatten()
                mses.append(np.mean((pred - y_raw_train[tr_mask][val_idx]) ** 2))
            if np.mean(mses) < best_mse:
                best_mse = np.mean(mses)
                best_nc = c

        pls = PLSRegression(n_components=best_nc)
        pls.fit(raw_tr, y_raw_train[tr_mask])
        pls_train[tr_mask, :best_nc] = pls.transform(raw_tr)
        pls_test[te_mask, :best_nc] = pls.transform(raw_te)

    return np.hstack([X_train, pls_train]), np.hstack([X_test, pls_test])


# ==============================================================
# LOCALLY WEIGHTED PREDICTION
# ==============================================================

def locally_weighted_prediction(X_train, y_train, X_test, pids_train, pids_test,
                                bandwidth_quantile=0.5, alpha=10.0):
    """Per-example locally weighted ridge regression."""
    unique_pids = sorted(np.unique(pids_train))
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X_train))

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        X_te = X_test[te_mask]
        n_tr = len(X_tr)
        tr_indices = np.where(tr_mask)[0]
        te_indices = np.where(te_mask)[0]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te) if len(X_te) > 0 else np.zeros((0, X_tr.shape[1]))

        D_tr_tr = cdist(X_tr_s, X_tr_s, metric='euclidean')
        D_te_tr = cdist(X_te_s, X_tr_s, metric='euclidean') if len(X_te) > 0 else np.zeros((0, n_tr))

        all_dists = D_tr_tr[np.triu_indices(n_tr, k=1)]
        if len(all_dists) > 0:
            sigma = np.quantile(all_dists, bandwidth_quantile)
            sigma = max(sigma, 1e-6)
        else:
            sigma = 1.0

        for i in range(n_tr):
            dists = D_tr_tr[i, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            weights[i] = 0
            if weights.sum() < 1e-10:
                oof_preds[tr_indices[i]] = np.mean(y_tr)
                continue
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            oof_preds[tr_indices[i]] = ridge.predict(X_tr_s[i:i+1])[0]

        for j in range(len(X_te)):
            dists = D_te_tr[j, :]
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
            if weights.sum() < 1e-10:
                test_preds[te_indices[j]] = np.mean(y_tr)
                continue
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_s, y_tr, sample_weight=weights)
            test_preds[te_indices[j]] = ridge.predict(X_te_s[j:j+1])[0]

    return oof_preds, test_preds


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("EXTERNAL MOTION PRETRAINING PIPELINE")
    print("=" * 70)

    # Determine device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Device: {device}")

    # ============================================================
    # PHASE 1: Generate/load external motion data
    # ============================================================
    print("\n--- PHASE 1: External Motion Data ---")

    # Generate synthetic motions
    print("  Generating 5000 synthetic motion sequences...")
    synthetic_data = generate_synthetic_motions(n_sequences=5000, seed=42)
    print(f"  Synthetic shape: {synthetic_data.shape}")

    # Parse BVH files if available
    bvh_dir = PROJECT_DIR / "external_data" / "cmu_mocap"
    bvh_files = list(bvh_dir.glob("*.bvh"))
    bvh_sequences = []
    for bvh_path in bvh_files:
        seq = parse_bvh_to_joint_positions(bvh_path, target_frames=240)
        if seq is not None:
            bvh_sequences.append(seq)
            print(f"  Parsed BVH: {bvh_path.name} -> {seq.shape}")

    if bvh_sequences:
        bvh_data = np.stack(bvh_sequences)
        # Pad to match common joints
        if bvh_data.shape[2] < N_COMMON_JOINTS:
            pad = np.zeros((bvh_data.shape[0], 240, N_COMMON_JOINTS - bvh_data.shape[2], 3))
            bvh_data = np.concatenate([bvh_data, pad], axis=2)
        bvh_data = bvh_data[:, :, :N_COMMON_JOINTS, :]

        all_pretrain_data = np.concatenate([synthetic_data, bvh_data], axis=0)
        print(f"  Total pretrain sequences: {len(all_pretrain_data)} ({len(synthetic_data)} synthetic + {len(bvh_data)} BVH)")
    else:
        all_pretrain_data = synthetic_data
        print(f"  Total pretrain sequences: {len(all_pretrain_data)} (all synthetic)")

    # ============================================================
    # PHASE 2: Pretrain temporal encoder
    # ============================================================
    print("\n--- PHASE 2: Pretrain Temporal Encoder ---")

    # Try multiple embedding dimensions
    embed_dims = [32, 64, 128]
    pretrainers = {}

    for embed_dim in embed_dims:
        print(f"\n  Training encoder (embed_dim={embed_dim})...")
        pretrainer = TemporalPretrainer(
            n_channels=N_CHANNELS, embed_dim=embed_dim, device=device)

        final_loss = pretrainer.pretrain(
            all_pretrain_data, n_epochs=30, batch_size=128, lr=1e-3,
            mask_ratio=0.15, verbose=True)

        print(f"  Final pretrain loss (embed_dim={embed_dim}): {final_loss:.6f}")
        pretrainers[embed_dim] = pretrainer

    # ============================================================
    # PHASE 3: Load basketball data and extract features
    # ============================================================
    print("\n--- PHASE 3: Load Basketball Data ---")

    train_data, test_data = load_data()
    y_train = train_data['y']
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    kp_index = train_data['kp_index']

    # Extract common joints from basketball data
    print("  Extracting common joints from basketball data...")
    train_common = extract_common_joints(train_data['X_3d'], kp_index)
    test_common = extract_common_joints(test_data['X_3d'], kp_index)
    print(f"  Train common joints: {train_common.shape}")
    print(f"  Test common joints: {test_common.shape}")

    # Extract pretrained embeddings
    print("  Extracting pretrained embeddings...")
    pretrained_features = {}
    for embed_dim, pretrainer in pretrainers.items():
        train_emb = pretrainer.extract_features(train_common)
        test_emb = pretrainer.extract_features(test_common)
        pretrained_features[embed_dim] = (train_emb, test_emb)
        print(f"  embed_dim={embed_dim}: train={train_emb.shape}, test={test_emb.shape}")

    # Load scalers
    scalers = {}
    target_idx = {"angle": 0, "depth": 1, "left_right": 2}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    y_scaled = {}
    for target in TARGETS:
        y_scaled[target] = scalers[target].transform(
            y_train[:, target_idx[target]].reshape(-1, 1)).ravel()

    # ============================================================
    # PHASE 4: Evaluate pretrained features
    # ============================================================
    print("\n--- PHASE 4: Evaluate Pretrained Features ---")

    # Load Sub 784 and Sub 1350 for comparison
    sub_784 = pd.read_csv(SUBMISSION_DIR / "submission_784.csv")
    sub_1350 = pd.read_csv(SUBMISSION_DIR / "submission_1350.csv")

    # Test configurations:
    # A. Pretrained features only (per-example Ridge)
    # B. HC features only (baseline, should match Sub 1350)
    # C. HC + Pretrained features combined
    # D. HC + PLS + Pretrained features combined

    results = {}

    for target in TARGETS:
        print(f"\n  TARGET: {target.upper()} (frame {TARGET_FRAMES[target]})")

        # Extract HC features
        X_train_hc, rf_train = extract_all_features(train_data, target)
        X_test_hc, rf_test = extract_all_features(test_data, target)

        # Augment with PLS
        y_raw = y_train[:, target_idx[target]]
        X_train_hc_pls, X_test_hc_pls = augment_with_pls(
            X_train_hc, y_raw, pids_train,
            X_test_hc, pids_test,
            train_data['X_raw'], test_data['X_raw'])

        y_target = y_scaled[target]

        # Baseline: HC + PLS (should reproduce Sub 1350 approach)
        print(f"    Baseline HC+PLS ({X_train_hc_pls.shape[1]} features):")
        oof_base, test_base = locally_weighted_prediction(
            X_train_hc_pls, y_target, X_test_hc_pls, pids_train, pids_test,
            bandwidth_quantile=0.5, alpha=10.0)
        mse_base = np.mean((oof_base - y_target) ** 2)
        print(f"      LOO MSE: {mse_base:.6f}")

        target_results = {'baseline_mse': mse_base, 'baseline_test': test_base}

        # Test each embed_dim
        for embed_dim in embed_dims:
            train_emb, test_emb = pretrained_features[embed_dim]

            # A. Pretrained only
            print(f"    Pretrained only (embed_dim={embed_dim}, {train_emb.shape[1]} features):")
            oof_pt, test_pt = locally_weighted_prediction(
                train_emb, y_target, test_emb, pids_train, pids_test,
                bandwidth_quantile=0.5, alpha=10.0)
            mse_pt = np.mean((oof_pt - y_target) ** 2)
            r_with_1350 = np.corrcoef(test_pt, sub_1350[f'scaled_{target}'].values)[0, 1]
            r_with_784 = np.corrcoef(test_pt, sub_784[f'scaled_{target}'].values)[0, 1]
            print(f"      LOO MSE: {mse_pt:.6f} (delta vs base: {(mse_pt/mse_base - 1)*100:+.1f}%)")
            print(f"      r(Sub 1350): {r_with_1350:.4f}, r(Sub 784): {r_with_784:.4f}")

            target_results[f'pt_only_{embed_dim}_mse'] = mse_pt
            target_results[f'pt_only_{embed_dim}_test'] = test_pt
            target_results[f'pt_only_{embed_dim}_r1350'] = r_with_1350

            # B. HC + PLS + Pretrained combined
            X_train_combined = np.hstack([X_train_hc_pls, train_emb])
            X_test_combined = np.hstack([X_test_hc_pls, test_emb])

            print(f"    HC+PLS+Pretrained (embed_dim={embed_dim}, {X_train_combined.shape[1]} features):")
            oof_comb, test_comb = locally_weighted_prediction(
                X_train_combined, y_target, X_test_combined, pids_train, pids_test,
                bandwidth_quantile=0.5, alpha=10.0)
            mse_comb = np.mean((oof_comb - y_target) ** 2)
            r_with_1350 = np.corrcoef(test_comb, sub_1350[f'scaled_{target}'].values)[0, 1]
            r_with_784 = np.corrcoef(test_comb, sub_784[f'scaled_{target}'].values)[0, 1]
            print(f"      LOO MSE: {mse_comb:.6f} (delta vs base: {(mse_comb/mse_base - 1)*100:+.1f}%)")
            print(f"      r(Sub 1350): {r_with_1350:.4f}, r(Sub 784): {r_with_784:.4f}")

            target_results[f'combined_{embed_dim}_mse'] = mse_comb
            target_results[f'combined_{embed_dim}_test'] = test_comb
            target_results[f'combined_{embed_dim}_r1350'] = r_with_1350

        results[target] = target_results

    # ============================================================
    # PHASE 5: Summary and Submissions
    # ============================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    # Find best configuration per target
    best_configs = {}
    for target in TARGETS:
        print(f"\n  {target.upper()}:")
        tr = results[target]
        print(f"    Baseline HC+PLS: {tr['baseline_mse']:.6f}")

        best_key = 'baseline'
        best_mse = tr['baseline_mse']
        best_test = tr['baseline_test']

        for embed_dim in embed_dims:
            for mode in ['pt_only', 'combined']:
                key = f'{mode}_{embed_dim}'
                mse = tr[f'{key}_mse']
                r1350 = tr[f'{key}_r1350']
                delta = (mse / tr['baseline_mse'] - 1) * 100
                print(f"    {key}: {mse:.6f} ({delta:+.1f}%), r(1350)={r1350:.4f}")
                if mse < best_mse:
                    best_mse = mse
                    best_key = key
                    best_test = tr[f'{key}_test']

        best_configs[target] = (best_key, best_mse, best_test)
        print(f"    BEST: {best_key} ({best_mse:.6f})")

    # Overall mean MSE comparison
    print(f"\n  Overall:")
    base_mean = np.mean([results[t]['baseline_mse'] for t in TARGETS])
    best_mean = np.mean([best_configs[t][1] for t in TARGETS])
    print(f"    Baseline mean MSE: {base_mean:.6f}")
    print(f"    Best mean MSE:     {best_mean:.6f}")
    print(f"    Delta: {(best_mean/base_mean - 1)*100:+.1f}%")

    # Generate submissions
    print(f"\n{'=' * 70}")
    print("GENERATING SUBMISSIONS")
    print(f"{'=' * 70}")

    # Submission 1: Best pretrained features standalone
    sub_num = get_next_submission_number()
    sub = pd.DataFrame({
        'id': test_data['ids'],
        'scaled_angle': best_configs['angle'][2],
        'scaled_depth': best_configs['depth'][2],
        'scaled_left_right': best_configs['left_right'][2],
    })
    sub.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
    print(f"  Sub {sub_num}: Best pretrained standalone")
    print(f"    angle: {best_configs['angle'][0]}, depth: {best_configs['depth'][0]}, LR: {best_configs['left_right'][0]}")

    # Diversity check
    print(f"\n  Diversity with Sub 784:")
    for target in TARGETS:
        col = f'scaled_{target}'
        r_784 = np.corrcoef(sub_784[col].values, best_configs[target][2])[0, 1]
        r_1350 = np.corrcoef(sub_1350[col].values, best_configs[target][2])[0, 1]
        print(f"    {target}: r(784)={r_784:.4f}, r(1350)={r_1350:.4f}")

    # Submissions blended with Sub 784
    for dw, lw, desc in [
        (0.30, 0.50, "standard blend"),
        (0.20, 0.30, "conservative blend"),
        (0.15, 0.20, "minimal blend"),
    ]:
        sub_num = get_next_submission_number()
        blended = sub_784.copy()
        blended['scaled_angle'] = sub_784['scaled_angle']  # Don't touch angle
        blended['scaled_depth'] = (1-dw)*sub_784['scaled_depth'] + dw*best_configs['depth'][2]
        blended['scaled_left_right'] = (1-lw)*sub_784['scaled_left_right'] + lw*best_configs['left_right'][2]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: dw={dw:.2f} lw={lw:.2f} ({desc})")

    # Submissions blended with Sub 1350
    for w, desc in [
        (0.10, "10% pretrained + 90% Sub 1350"),
        (0.20, "20% pretrained + 80% Sub 1350"),
        (0.30, "30% pretrained + 70% Sub 1350"),
    ]:
        sub_num = get_next_submission_number()
        blended = sub_1350.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            blended[col] = (1-w)*sub_1350[col] + w*best_configs[target][2]
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {desc}")

    # Also try pretrained-only features blended with Sub 1350 (for diversity)
    # Find the most diverse pretrained-only config
    most_diverse_target = {}
    for target in TARGETS:
        best_r = 1.0
        best_embed = 32
        for embed_dim in embed_dims:
            r = results[target].get(f'pt_only_{embed_dim}_r1350', 1.0)
            if r < best_r:
                best_r = r
                best_embed = embed_dim
        most_diverse_target[target] = (best_embed, best_r)

    print(f"\n  Most diverse pretrained-only configs:")
    for target in TARGETS:
        embed_dim, r = most_diverse_target[target]
        print(f"    {target}: embed_dim={embed_dim}, r(1350)={r:.4f}")

    for w, desc in [
        (0.10, "10% diverse pretrained + 90% Sub 1350"),
        (0.15, "15% diverse pretrained + 85% Sub 1350"),
    ]:
        sub_num = get_next_submission_number()
        blended = sub_1350.copy()
        for target in TARGETS:
            col = f'scaled_{target}'
            embed_dim = most_diverse_target[target][0]
            test_diverse = results[target][f'pt_only_{embed_dim}_test']
            blended[col] = (1-w)*sub_1350[col] + w*test_diverse
        blended.to_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv", index=False)
        print(f"  Sub {sub_num}: {desc}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

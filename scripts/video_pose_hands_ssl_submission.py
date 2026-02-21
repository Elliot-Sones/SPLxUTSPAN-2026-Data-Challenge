#!/usr/bin/env python
"""Pose+hands-domain SSL pretraining and submission generation.

This script uses external free-throw videos, extracts full challenge-style keypoints
with Mediapipe pose + hand landmarker tasks, pretrains an encoder with MAE,
fine-tunes on challenge labels, and creates submissions.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as iio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from video_ssl_transfer import (
    ChallengeRegressor,
    TemporalEncoder,
    load_challenge_dataset,
    pretrain_external_mae,
    run_groupkfold_regression,
    sample_frames_indices,
    set_seed,
)

import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import get_keypoint_columns, iterate_shots  # noqa: E402


SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]


def build_keypoint_order() -> List[str]:
    cols = get_keypoint_columns()
    names: List[str] = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in names:
            names.append(name)
    return names


KEYPOINT_NAMES = build_keypoint_order()
KEYPOINT_TO_IDX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# Mediapipe pose landmark indices.
POSE_MAP = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_big_toe": 31,
    "left_small_toe": 31,
    "right_big_toe": 32,
    "right_small_toe": 32,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_thumb": 21,
    "right_thumb": 22,
}

# Mediapipe hand landmark indices.
HAND_FINGER_MAP = {
    "first_finger": (1, 2, 3, 4),
    "second_finger": (5, 6, 7, 8),
    "third_finger": (9, 10, 11, 12),
    "fourth_finger": (13, 14, 15, 16),
    "fifth_finger": (17, 18, 19, 20),
}
HAND_KEYPOINTS = {
    name
    for name in KEYPOINT_NAMES
    if "_finger_" in name or name.endswith("_thumb") or name.endswith("_pinky")
}


def get_next_submission_number() -> int:
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = []
            for fp in existing:
                parts = fp.stem.split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    nums.append(int(parts[1]))
            next_num = max(nums) + 1 if nums else 1
            placeholder = SUBMISSION_DIR / f"submission_{next_num}.csv"
            placeholder.touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def normalize_pose_sequence(coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalize sequence by root center and shoulder distance."""
    t, j, _ = coords.shape
    out = coords.copy()

    lhip_idx = KEYPOINT_TO_IDX["left_hip"]
    rhip_idx = KEYPOINT_TO_IDX["right_hip"]
    lsho_idx = KEYPOINT_TO_IDX["left_shoulder"]
    rsho_idx = KEYPOINT_TO_IDX["right_shoulder"]

    centers = np.zeros((t, 2), dtype=np.float32)
    valid_center = np.zeros(t, dtype=bool)
    for i in range(t):
        lh_ok = mask[i, lhip_idx] > 0
        rh_ok = mask[i, rhip_idx] > 0
        ls_ok = mask[i, lsho_idx] > 0
        rs_ok = mask[i, rsho_idx] > 0
        if lh_ok and rh_ok:
            centers[i] = 0.5 * (coords[i, lhip_idx] + coords[i, rhip_idx])
            valid_center[i] = True
        elif ls_ok and rs_ok:
            centers[i] = 0.5 * (coords[i, lsho_idx] + coords[i, rsho_idx])
            valid_center[i] = True

    if valid_center.any():
        first = np.where(valid_center)[0][0]
        last_center = centers[first].copy()
        for i in range(t):
            if valid_center[i]:
                last_center = centers[i].copy()
            else:
                centers[i] = last_center
    else:
        centers[:] = 0.0

    out = out - centers[:, None, :]

    shoulder_scales = []
    for i in range(t):
        if mask[i, lsho_idx] > 0 and mask[i, rsho_idx] > 0:
            d = np.linalg.norm(out[i, lsho_idx] - out[i, rsho_idx])
            if d > 1e-6:
                shoulder_scales.append(float(d))
    if shoulder_scales:
        scale = float(np.median(shoulder_scales))
    else:
        valid = mask > 0
        if valid.any():
            vals = out[valid]
            scale = float(np.nanstd(vals))
            if scale < 1e-6:
                scale = 1.0
        else:
            scale = 1.0

    out = out / scale
    out[mask <= 0, :] = 0.0

    flat_xy = out.reshape(t, j * 2)
    flat_mask = mask.astype(np.float32)
    return np.concatenate([flat_xy, flat_mask], axis=1).astype(np.float32)


def challenge_timeseries_to_pose_sequence(
    timeseries: np.ndarray,
    keypoint_name_to_index: Dict[str, int],
    num_frames: int,
) -> np.ndarray:
    frame_idx = sample_frames_indices(timeseries.shape[0], num_frames)
    t = len(frame_idx)
    j = len(KEYPOINT_NAMES)
    coords = np.zeros((t, j, 2), dtype=np.float32)
    mask = np.zeros((t, j), dtype=np.float32)

    for i, f in enumerate(frame_idx):
        row = timeseries[int(f)]
        for j_idx, joint in enumerate(KEYPOINT_NAMES):
            kp_idx = keypoint_name_to_index.get(joint)
            if kp_idx is None:
                continue
            x = row[kp_idx * 3 + 0]
            y = row[kp_idx * 3 + 1]
            if np.isnan(x) or np.isnan(y):
                continue
            coords[i, j_idx, 0] = float(x)
            coords[i, j_idx, 1] = float(y)
            mask[i, j_idx] = 1.0

    return normalize_pose_sequence(coords=coords, mask=mask)


def cache_path_for_video(cache_dir: Path, video_path: Path, num_frames: int, vis_thr: float, hand_thr: float) -> Path:
    key = f"{video_path.resolve()}|{video_path.stat().st_mtime_ns}|{num_frames}|{vis_thr:.5f}|{hand_thr:.5f}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npz"


def ensure_pose_landmarker_model(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return model_path
    url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    )
    urllib.request.urlretrieve(url, model_path)
    return model_path


def ensure_hand_landmarker_model(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return model_path
    url = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )
    urllib.request.urlretrieve(url, model_path)
    return model_path


class PoseHandsExtractor:
    def __init__(
        self,
        pose_model_path: Path,
        hand_model_path: Path,
        min_detection_confidence: float,
        min_presence_confidence: float,
        min_tracking_confidence: float,
        num_hands: int = 2,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(pose_model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(hand_model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        )

    def _assign_hand_sides(
        self,
        hand_candidates: List[Dict[str, object]],
        pose_left_wrist: np.ndarray | None,
        pose_right_wrist: np.ndarray | None,
    ) -> Dict[str, Dict[str, object] | None]:
        assigned = {"left": None, "right": None}
        unknown = []

        for cand in hand_candidates:
            hand_wrist = cand["coords"][0]
            side = None

            if pose_left_wrist is not None and pose_right_wrist is not None:
                dl = float(np.linalg.norm(hand_wrist - pose_left_wrist))
                dr = float(np.linalg.norm(hand_wrist - pose_right_wrist))
                side = "left" if dl <= dr else "right"
            else:
                label = str(cand.get("label", "")).lower()
                if label.startswith("left"):
                    side = "left"
                elif label.startswith("right"):
                    side = "right"

            if side is None:
                unknown.append(cand)
                continue

            cur = assigned[side]
            if cur is None or cand["score"] > cur["score"]:
                assigned[side] = cand

        if unknown:
            unknown_sorted = sorted(unknown, key=lambda c: float(c["coords"][0, 0]))
            if assigned["left"] is None and unknown_sorted:
                assigned["left"] = unknown_sorted[0]
                unknown_sorted = unknown_sorted[1:]
            if assigned["right"] is None and unknown_sorted:
                assigned["right"] = unknown_sorted[-1]

        return assigned

    def _fill_hand(
        self,
        side: str,
        hand_coords: np.ndarray,
        coords: np.ndarray,
        vis: np.ndarray,
    ) -> None:
        prefix = "left" if side == "left" else "right"

        wrist_name = f"{prefix}_wrist"
        if wrist_name in KEYPOINT_TO_IDX:
            idx = KEYPOINT_TO_IDX[wrist_name]
            coords[idx] = hand_coords[0]
            vis[idx] = 1.0

        # Finger chains.
        for finger_name, lm_ids in HAND_FINGER_MAP.items():
            if finger_name == "first_finger":
                names = [
                    f"{prefix}_{finger_name}_cmc",
                    f"{prefix}_{finger_name}_mcp",
                    f"{prefix}_{finger_name}_ip",
                    f"{prefix}_{finger_name}_distal",
                ]
            else:
                names = [
                    f"{prefix}_{finger_name}_mcp",
                    f"{prefix}_{finger_name}_pip",
                    f"{prefix}_{finger_name}_dip",
                    f"{prefix}_{finger_name}_distal",
                ]

            for name, lm_id in zip(names, lm_ids):
                if name in KEYPOINT_TO_IDX:
                    idx = KEYPOINT_TO_IDX[name]
                    coords[idx] = hand_coords[lm_id]
                    vis[idx] = 1.0

        thumb_tip_name = f"{prefix}_thumb"
        if thumb_tip_name in KEYPOINT_TO_IDX:
            idx = KEYPOINT_TO_IDX[thumb_tip_name]
            coords[idx] = hand_coords[4]
            vis[idx] = 1.0

        pinky_tip_name = f"{prefix}_pinky"
        if pinky_tip_name in KEYPOINT_TO_IDX:
            idx = KEYPOINT_TO_IDX[pinky_tip_name]
            coords[idx] = hand_coords[20]
            vis[idx] = 1.0

    def detect(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        j = len(KEYPOINT_NAMES)
        coords = np.zeros((j, 2), dtype=np.float32)
        vis = np.zeros(j, dtype=np.float32)

        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
        pose_res = self.pose_landmarker.detect(img)
        hand_res = self.hand_landmarker.detect(img)

        pose_left_wrist = None
        pose_right_wrist = None

        if pose_res.pose_landmarks:
            pose_landmarks = pose_res.pose_landmarks[0]

            for joint, pose_idx in POSE_MAP.items():
                if joint not in KEYPOINT_TO_IDX:
                    continue
                lm = pose_landmarks[pose_idx]
                idx = KEYPOINT_TO_IDX[joint]
                coords[idx, 0] = float(lm.x)
                coords[idx, 1] = float(lm.y)
                vis[idx] = float(getattr(lm, "visibility", 1.0))

            if "left_wrist" in KEYPOINT_TO_IDX and vis[KEYPOINT_TO_IDX["left_wrist"]] > 0:
                pose_left_wrist = coords[KEYPOINT_TO_IDX["left_wrist"]].copy()
            if "right_wrist" in KEYPOINT_TO_IDX and vis[KEYPOINT_TO_IDX["right_wrist"]] > 0:
                pose_right_wrist = coords[KEYPOINT_TO_IDX["right_wrist"]].copy()

            # Derived virtual joints.
            if all(k in KEYPOINT_TO_IDX for k in ["left_shoulder", "right_shoulder", "neck"]):
                ls_idx = KEYPOINT_TO_IDX["left_shoulder"]
                rs_idx = KEYPOINT_TO_IDX["right_shoulder"]
                nk_idx = KEYPOINT_TO_IDX["neck"]
                if vis[ls_idx] > 0 and vis[rs_idx] > 0:
                    coords[nk_idx] = 0.5 * (coords[ls_idx] + coords[rs_idx])
                    vis[nk_idx] = min(vis[ls_idx], vis[rs_idx])

            if all(k in KEYPOINT_TO_IDX for k in ["left_hip", "right_hip", "mid_hip"]):
                lh_idx = KEYPOINT_TO_IDX["left_hip"]
                rh_idx = KEYPOINT_TO_IDX["right_hip"]
                mh_idx = KEYPOINT_TO_IDX["mid_hip"]
                if vis[lh_idx] > 0 and vis[rh_idx] > 0:
                    coords[mh_idx] = 0.5 * (coords[lh_idx] + coords[rh_idx])
                    vis[mh_idx] = min(vis[lh_idx], vis[rh_idx])

        hand_candidates: List[Dict[str, object]] = []
        if hand_res.hand_landmarks:
            handedness = hand_res.handedness or []
            for i, lms in enumerate(hand_res.hand_landmarks):
                arr = np.zeros((21, 2), dtype=np.float32)
                for k, lm in enumerate(lms):
                    arr[k, 0] = float(lm.x)
                    arr[k, 1] = float(lm.y)

                label = ""
                score = 0.0
                if i < len(handedness) and handedness[i]:
                    cat = handedness[i][0]
                    label = str(getattr(cat, "category_name", ""))
                    score = float(getattr(cat, "score", 0.0))

                hand_candidates.append({"coords": arr, "label": label, "score": score})

        if hand_candidates:
            assigned = self._assign_hand_sides(
                hand_candidates=hand_candidates,
                pose_left_wrist=pose_left_wrist,
                pose_right_wrist=pose_right_wrist,
            )
            if assigned["left"] is not None:
                self._fill_hand("left", assigned["left"]["coords"], coords, vis)
            if assigned["right"] is not None:
                self._fill_hand("right", assigned["right"]["coords"], coords, vis)

        return coords, vis

    def close(self) -> None:
        self.pose_landmarker.close()
        self.hand_landmarker.close()


def external_video_to_pose_sequence(
    video_path: Path,
    num_frames: int,
    vis_threshold: float,
    hand_threshold: float,
    extractor: PoseHandsExtractor,
) -> np.ndarray:
    reader = iio.get_reader(str(video_path), "ffmpeg")
    frames = []
    try:
        for fr in reader:
            frames.append(fr)
    finally:
        reader.close()

    t = num_frames
    j = len(KEYPOINT_NAMES)
    coords = np.zeros((t, j, 2), dtype=np.float32)
    mask = np.zeros((t, j), dtype=np.float32)

    if not frames:
        return normalize_pose_sequence(coords=coords, mask=mask)

    frame_idx = sample_frames_indices(len(frames), num_frames)
    for i, fi in enumerate(frame_idx):
        frame = frames[int(fi)]
        frame_coords, frame_vis = extractor.detect(frame_rgb=frame)
        for j_idx in range(j):
            joint_name = KEYPOINT_NAMES[j_idx]
            thr = hand_threshold if joint_name in HAND_KEYPOINTS else vis_threshold
            if frame_vis[j_idx] < thr:
                continue
            coords[i, j_idx] = frame_coords[j_idx]
            mask[i, j_idx] = 1.0

    return normalize_pose_sequence(coords=coords, mask=mask)


def maybe_add_diffs(x: np.ndarray, use_diff: bool) -> np.ndarray:
    if not use_diff:
        return x
    diffs = np.diff(x, axis=1, prepend=x[:, :1, :])
    return np.concatenate([x, diffs], axis=2)


def load_external_pose_dataset(
    external_root: Path,
    cache_dir: Path,
    max_external: int,
    num_frames: int,
    seed: int,
    vis_threshold: float,
    hand_threshold: float,
    pose_model_path: Path,
    hand_model_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ft0 = sorted((external_root / "ft0").glob("*.mp4"))
    ft1 = sorted((external_root / "ft1").glob("*.mp4"))
    if not ft0 or not ft1:
        raise FileNotFoundError(f"Expected mp4 files in {external_root / 'ft0'} and {external_root / 'ft1'}")

    rng = np.random.default_rng(seed)
    max_per_class = max_external // 2
    n0 = min(len(ft0), max_per_class)
    n1 = min(len(ft1), max_per_class)

    sel0 = rng.choice(ft0, size=n0, replace=False).tolist()
    sel1 = rng.choice(ft1, size=n1, replace=False).tolist()
    paths = sel0 + sel1
    labels = np.array([0.0] * len(sel0) + [1.0] * len(sel1), dtype=np.float32)

    order = rng.permutation(len(paths))
    paths = [paths[i] for i in order]
    labels = labels[order]

    cache_dir.mkdir(parents=True, exist_ok=True)
    seqs = []
    used_paths = []
    used_labels = []

    extractor = PoseHandsExtractor(
        pose_model_path=pose_model_path,
        hand_model_path=hand_model_path,
        min_detection_confidence=0.2,
        min_presence_confidence=0.2,
        min_tracking_confidence=0.2,
    )
    try:
        for p, lbl in zip(paths, labels):
            cpath = cache_path_for_video(
                cache_dir=cache_dir,
                video_path=p,
                num_frames=num_frames,
                vis_thr=vis_threshold,
                hand_thr=hand_threshold,
            )
            if cpath.exists():
                arr = np.load(cpath)["seq"].astype(np.float32)
                seqs.append(arr)
                used_paths.append(str(p))
                used_labels.append(float(lbl))
                continue
            try:
                seq = external_video_to_pose_sequence(
                    video_path=p,
                    num_frames=num_frames,
                    vis_threshold=vis_threshold,
                    hand_threshold=hand_threshold,
                    extractor=extractor,
                )
                np.savez_compressed(cpath, seq=seq)
                seqs.append(seq)
                used_paths.append(str(p))
                used_labels.append(float(lbl))
            except Exception as exc:  # pragma: no cover
                print(f"[warn] failed to extract pose+hands {p}: {exc}")
    finally:
        extractor.close()

    dim = len(KEYPOINT_NAMES) * 3
    x = np.stack(seqs) if seqs else np.zeros((0, num_frames, dim), dtype=np.float32)
    y = np.array(used_labels, dtype=np.float32)
    return x, y, used_paths


def load_challenge_pose_train_dataset(num_frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, y_scaled, groups = load_challenge_dataset(num_frames=num_frames, frame_size=8)
    keypoint_name_to_index = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
    seqs = []
    for _, ts in iterate_shots(train=True):
        seqs.append(
            challenge_timeseries_to_pose_sequence(
                timeseries=ts,
                keypoint_name_to_index=keypoint_name_to_index,
                num_frames=num_frames,
            )
        )
    x = np.stack(seqs).astype(np.float32)
    return x, y_scaled.astype(np.float32), groups


def load_challenge_pose_test_dataset(num_frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    keypoint_name_to_index = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
    seqs = []
    ids = []
    pids = []
    for meta, ts in iterate_shots(train=False):
        ids.append(meta["id"])
        pids.append(meta["participant_id"])
        seqs.append(
            challenge_timeseries_to_pose_sequence(
                timeseries=ts,
                keypoint_name_to_index=keypoint_name_to_index,
                num_frames=num_frames,
            )
        )
    x = np.stack(seqs).astype(np.float32)
    return x, np.array(ids), np.array(pids)


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_test - mean) / std


def train_full_and_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    device: str,
    pretrained_encoder_state: Dict[str, torch.Tensor] | None,
) -> np.ndarray:
    set_seed(seed)
    x_train_std, x_test_std = standardize_train_test(x_train, x_test)

    x_train_t = torch.tensor(x_train_std, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test_std, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    encoder = TemporalEncoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    if pretrained_encoder_state is not None:
        encoder.load_state_dict(pretrained_encoder_state)
    model = ChallengeRegressor(encoder=encoder, embed_dim=hidden_dim * 2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        test_pred = model(x_test_t.to(device)).cpu().numpy()
    return test_pred


class SequenceMAEWithClassifier(nn.Module):
    def __init__(self, encoder: TemporalEncoder, embed_dim: int, num_frames: int, input_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_frames * input_dim),
        )
        self.cls_head = nn.Linear(embed_dim, 1)

    def forward_recon(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon.reshape(x.shape[0], self.num_frames, self.input_dim)

    def forward_cls(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.cls_head(z).squeeze(-1)


def pretrain_external_mae_bce(
    x_unlabeled: np.ndarray,
    x_external_labeled: np.ndarray,
    y_external: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    device: str,
    mask_ratio: float,
    cls_weight: float,
    cls_schedule: str,
) -> Dict[str, object]:
    set_seed(seed)
    x_unlabeled_t = torch.tensor(x_unlabeled, dtype=torch.float32)
    x_external_t = torch.tensor(x_external_labeled, dtype=torch.float32)
    y_external_t = torch.tensor(y_external, dtype=torch.float32)
    num_frames = x_unlabeled.shape[1]

    encoder = TemporalEncoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    model = SequenceMAEWithClassifier(
        encoder=encoder,
        embed_dim=hidden_dim * 2,
        num_frames=num_frames,
        input_dim=input_dim,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    unlabeled_loader = DataLoader(TensorDataset(x_unlabeled_t), batch_size=batch_size, shuffle=True)
    external_loader = DataLoader(TensorDataset(x_external_t, y_external_t), batch_size=batch_size, shuffle=True)

    pos = float(y_external.sum())
    neg = float(len(y_external) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    mae_losses: List[float] = []
    cls_losses: List[float] = []
    epoch_cls_weights: List[float] = []
    for epoch in range(1, epochs + 1):
        if cls_schedule == "linear_decay" and epochs > 1:
            epoch_cls_weight = cls_weight * (1.0 - float(epoch - 1) / float(epochs - 1))
        else:
            epoch_cls_weight = cls_weight
        epoch_cls_weights.append(float(epoch_cls_weight))

        model.train()
        total_mae = 0.0
        total_cls = 0.0
        total_n = 0

        ext_iter = iter(external_loader)
        for (xb_unlabeled,) in unlabeled_loader:
            xb_unlabeled = xb_unlabeled.to(device)
            mask = torch.rand_like(xb_unlabeled) < mask_ratio
            xb_masked = xb_unlabeled.clone()
            xb_masked[mask] = 0.0
            recon = model.forward_recon(xb_masked)
            mae_loss = ((recon[mask] - xb_unlabeled[mask]) ** 2).mean()

            try:
                xb_ext, yb_ext = next(ext_iter)
            except StopIteration:
                ext_iter = iter(external_loader)
                xb_ext, yb_ext = next(ext_iter)
            xb_ext = xb_ext.to(device)
            yb_ext = yb_ext.to(device)
            logits = model.forward_cls(xb_ext)
            cls_loss = bce_loss_fn(logits, yb_ext)

            loss = mae_loss + epoch_cls_weight * cls_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_mae += float(mae_loss.item()) * xb_unlabeled.size(0)
            total_cls += float(cls_loss.item()) * xb_unlabeled.size(0)
            total_n += int(xb_unlabeled.size(0))

        epoch_mae = total_mae / max(total_n, 1)
        epoch_cls = total_cls / max(total_n, 1)
        mae_losses.append(epoch_mae)
        cls_losses.append(epoch_cls)
        print(
            f"pretrain_epoch={epoch} "
            f"mae_loss={epoch_mae:.12f} "
            f"cls_loss={epoch_cls:.12f} "
            f"cls_weight={epoch_cls_weight:.12f}"
        )

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model.forward_cls(x_external_t.to(device))).cpu().numpy()
    pred_labels = (probs >= 0.5).astype(np.float32)
    train_acc = float((pred_labels == y_external).mean())

    return {
        "encoder_state_dict": encoder.state_dict(),
        "losses": mae_losses,
        "cls_losses": cls_losses,
        "train_acc": train_acc,
        "pretrain_mode": "mae_bce",
        "cls_weight": float(cls_weight),
        "cls_schedule": cls_schedule,
        "cls_epoch_weights": epoch_cls_weights,
    }


def correlation_report(a: pd.DataFrame, b: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for col in TARGET_COLS:
        out[col] = float(np.corrcoef(a[col].values, b[col].values)[0, 1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose+hands-domain SSL pretraining submission generator")
    parser.add_argument("--external-root", type=str, default="Basketball_51 dataset")
    parser.add_argument("--max-external", type=int, default=1132)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--use-diff", action="store_true")
    parser.add_argument("--vis-threshold", type=float, default=0.2)
    parser.add_argument("--hand-threshold", type=float, default=0.2)
    parser.add_argument("--cache-dir", type=str, default="output/pose_hands_cache_mediapipe")
    parser.add_argument("--pose-model-path", type=str, default="output/models/pose_landmarker_lite.task")
    parser.add_argument("--hand-model-path", type=str, default="output/models/hand_landmarker.task")
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr-pretrain", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-mode", type=str, default="mae", choices=["mae", "mae_bce"])
    parser.add_argument("--pretrain-mask-ratio", type=float, default=0.3)
    parser.add_argument("--pretrain-cls-weight", type=float, default=0.25)
    parser.add_argument("--pretrain-cls-schedule", type=str, default="constant", choices=["constant", "linear_decay"])
    parser.add_argument("--include-challenge-train-unlabeled", action="store_true")
    parser.add_argument("--include-challenge-test-unlabeled", action="store_true")
    parser.add_argument("--base-submission", type=int, default=771)
    parser.add_argument("--blend-weights", type=float, nargs="*", default=[0.2, 0.35, 0.5])
    parser.add_argument("--output-dir", type=str, default="output/video_pose_hands_ssl_submission")
    args = parser.parse_args()

    out_dir = PROJECT_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = PROJECT_DIR / args.cache_dir
    pose_model_path = ensure_pose_landmarker_model(PROJECT_DIR / args.pose_model_path)
    hand_model_path = ensure_hand_landmarker_model(PROJECT_DIR / args.hand_model_path)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    x_external, y_external, used_paths = load_external_pose_dataset(
        external_root=PROJECT_DIR / args.external_root,
        cache_dir=cache_dir,
        max_external=args.max_external,
        num_frames=args.num_frames,
        seed=args.seed,
        vis_threshold=args.vis_threshold,
        hand_threshold=args.hand_threshold,
        pose_model_path=pose_model_path,
        hand_model_path=hand_model_path,
    )
    x_external = maybe_add_diffs(x_external, args.use_diff)
    print(f"external_samples={len(x_external)} external_dim={x_external.shape[2]}")
    print(f"external_positive_rate={float(y_external.mean()):.12f}")

    x_train, y_train, groups = load_challenge_pose_train_dataset(num_frames=args.num_frames)
    x_test, test_ids, _ = load_challenge_pose_test_dataset(num_frames=args.num_frames)
    x_train = maybe_add_diffs(x_train, args.use_diff)
    x_test = maybe_add_diffs(x_test, args.use_diff)
    print(f"challenge_train_samples={len(x_train)} challenge_test_samples={len(x_test)}")
    print(f"challenge_dim={x_train.shape[2]}")

    extra_unlabeled_blocks = []
    if args.include_challenge_train_unlabeled:
        extra_unlabeled_blocks.append(x_train.copy())
    if args.include_challenge_test_unlabeled:
        extra_unlabeled_blocks.append(x_test.copy())

    if extra_unlabeled_blocks:
        x_pretrain_raw = np.concatenate([x_external] + extra_unlabeled_blocks, axis=0)
    else:
        x_pretrain_raw = x_external

    pre_mean = x_pretrain_raw.mean(axis=(0, 1), keepdims=True)
    pre_std = x_pretrain_raw.std(axis=(0, 1), keepdims=True) + 1e-6
    x_pretrain_std = (x_pretrain_raw - pre_mean) / pre_std
    x_external_std = x_pretrain_std[: len(x_external)]
    print(f"pretrain_corpus_samples={len(x_pretrain_std)}")

    if args.pretrain_mode == "mae_bce":
        pretrain = pretrain_external_mae_bce(
            x_unlabeled=x_pretrain_std,
            x_external_labeled=x_external_std,
            y_external=y_external,
            input_dim=x_train.shape[2],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            batch_size=args.batch_size,
            epochs=args.pretrain_epochs,
            lr=args.lr_pretrain,
            seed=args.seed,
            device=device,
            mask_ratio=args.pretrain_mask_ratio,
            cls_weight=args.pretrain_cls_weight,
            cls_schedule=args.pretrain_cls_schedule,
        )
    else:
        pretrain = pretrain_external_mae(
            x_unlabeled=x_pretrain_std,
            input_dim=x_train.shape[2],
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            batch_size=args.batch_size,
            epochs=args.pretrain_epochs,
            lr=args.lr_pretrain,
            seed=args.seed,
            device=device,
            mask_ratio=args.pretrain_mask_ratio,
        )
    print(f"pretrain_mode={pretrain['pretrain_mode']}")
    print(f"pretrain_last_loss={pretrain['losses'][-1]:.12f}")
    if "cls_losses" in pretrain:
        print(f"pretrain_last_cls_loss={pretrain['cls_losses'][-1]:.12f}")
    if not np.isnan(float(pretrain.get("train_acc", np.nan))):
        print(f"pretrain_train_acc={float(pretrain['train_acc']):.12f}")

    cv_pretrained = run_groupkfold_regression(
        x=x_train,
        y=y_train,
        groups=groups,
        input_dim=x_train.shape[2],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.finetune_epochs,
        lr=args.lr_finetune,
        seed=args.seed,
        device=device,
        pretrained_encoder_state=pretrain["encoder_state_dict"],
    )
    print(
        "cv_pretrained "
        f"mse_angle={cv_pretrained['avg']['mse_angle']:.12f} "
        f"mse_depth={cv_pretrained['avg']['mse_depth']:.12f} "
        f"mse_left_right={cv_pretrained['avg']['mse_left_right']:.12f} "
        f"mse_total={cv_pretrained['avg']['mse_total']:.12f}"
    )

    test_pred = train_full_and_predict(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        input_dim=x_train.shape[2],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.finetune_epochs,
        lr=args.lr_finetune,
        seed=args.seed,
        device=device,
        pretrained_encoder_state=pretrain["encoder_state_dict"],
    )

    direct_df = pd.DataFrame(
        {
            "id": test_ids,
            "scaled_angle": np.clip(test_pred[:, 0], 0.0, 1.0),
            "scaled_depth": np.clip(test_pred[:, 1], 0.0, 1.0),
            "scaled_left_right": np.clip(test_pred[:, 2], 0.0, 1.0),
        }
    ).sort_values("id").reset_index(drop=True)

    sub_num_direct = get_next_submission_number()
    sub_path_direct = SUBMISSION_DIR / f"submission_{sub_num_direct}.csv"
    direct_df.to_csv(sub_path_direct, index=False)
    print(
        f"saved_submission={sub_num_direct} name=pose_hands_ssl_direct "
        f"angle_std={float(direct_df['scaled_angle'].std()):.12f} "
        f"depth_mean={float(direct_df['scaled_depth'].mean()):.12f}"
    )

    base_path = SUBMISSION_DIR / f"submission_{args.base_submission}.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"Base submission not found: {base_path}")
    base_df = pd.read_csv(base_path).sort_values("id").reset_index(drop=True)
    if not np.array_equal(base_df["id"].values, direct_df["id"].values):
        raise ValueError("ID mismatch between base and direct pose+hands submission")

    corr_with_base = correlation_report(base_df, direct_df)
    print(
        "corr_with_base "
        f"angle={corr_with_base['scaled_angle']:.12f} "
        f"depth={corr_with_base['scaled_depth']:.12f} "
        f"left_right={corr_with_base['scaled_left_right']:.12f}"
    )

    blend_records = []
    for w in args.blend_weights:
        out = base_df.copy()
        for c in TARGET_COLS:
            out[c] = np.clip((1.0 - w) * base_df[c].values + w * direct_df[c].values, 0.0, 1.0)
        sub_num = get_next_submission_number()
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        out.to_csv(sub_path, index=False)
        rec = {
            "submission_num": int(sub_num),
            "submission_file": str(sub_path),
            "blend_weight_direct": float(w),
            "angle_std": float(out["scaled_angle"].std()),
            "depth_mean": float(out["scaled_depth"].mean()),
            "left_right_std": float(out["scaled_left_right"].std()),
            "corr_with_base": correlation_report(base_df, out),
        }
        blend_records.append(rec)
        print(
            f"saved_submission={sub_num} name=pose_hands_ssl_blend_w{w:.3f} "
            f"angle_std={rec['angle_std']:.12f} "
            f"depth_mean={rec['depth_mean']:.12f}"
        )

    d_depth_lock = direct_df.copy()
    d_depth_lock["scaled_depth"] = np.clip(
        direct_df["scaled_depth"].values + (base_df["scaled_depth"].mean() - direct_df["scaled_depth"].mean()),
        0.0,
        1.0,
    )
    sub_num_depth_lock = get_next_submission_number()
    sub_path_depth_lock = SUBMISSION_DIR / f"submission_{sub_num_depth_lock}.csv"
    d_depth_lock.to_csv(sub_path_depth_lock, index=False)
    print(
        f"saved_submission={sub_num_depth_lock} name=pose_hands_ssl_direct_depth_lock "
        f"angle_std={float(d_depth_lock['scaled_angle'].std()):.12f} "
        f"depth_mean={float(d_depth_lock['scaled_depth'].mean()):.12f}"
    )

    run_meta = {
        "config": vars(args),
        "device": device,
        "keypoint_names": KEYPOINT_NAMES,
        "external_samples_used": int(len(x_external)),
        "external_positive_rate": float(y_external.mean()),
        "external_paths_used": used_paths,
        "pretrain": {
            "mode": pretrain["pretrain_mode"],
            "losses": [float(x) for x in pretrain["losses"]],
            "train_acc": float(pretrain.get("train_acc", np.nan)),
            "cls_weight": float(pretrain.get("cls_weight", np.nan)),
            "cls_schedule": pretrain.get("cls_schedule", "constant"),
            "cls_losses": [float(x) for x in pretrain.get("cls_losses", [])],
            "cls_epoch_weights": [float(x) for x in pretrain.get("cls_epoch_weights", [])],
        },
        "cv_pretrained": cv_pretrained,
        "base_submission": int(args.base_submission),
        "direct_submission": {
            "submission_num": int(sub_num_direct),
            "submission_file": str(sub_path_direct),
            "angle_std": float(direct_df["scaled_angle"].std()),
            "depth_mean": float(direct_df["scaled_depth"].mean()),
            "left_right_std": float(direct_df["scaled_left_right"].std()),
            "corr_with_base": corr_with_base,
        },
        "blend_submissions": blend_records,
        "depth_lock_submission": {
            "submission_num": int(sub_num_depth_lock),
            "submission_file": str(sub_path_depth_lock),
            "angle_std": float(d_depth_lock["scaled_angle"].std()),
            "depth_mean": float(d_depth_lock["scaled_depth"].mean()),
            "left_right_std": float(d_depth_lock["scaled_left_right"].std()),
            "corr_with_base": correlation_report(base_df, d_depth_lock),
        },
    }

    metrics_path = out_dir / "run_metrics.json"
    metrics_path.write_text(json.dumps(run_meta, indent=2))
    print(f"metrics_path={metrics_path}")


if __name__ == "__main__":
    main()

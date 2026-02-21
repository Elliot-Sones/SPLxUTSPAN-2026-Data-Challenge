"""
Pulse Features Transfer Pipeline

Goal:
- Convert complex motion into 1D pulse waveforms (motion-energy over time).
- Build SHOT7M2 free-throw-like pulse templates.
- Extract target-frame pulse-shape features from competition shots.
- Evaluate honest per-player LOO and generate conservative submission blends.

Core idea:
- Pulse(t) = mean joint speed magnitude across selected joints at frame t.
- We compare per-shot pulse windows to SHOT7M2 templates (correlation, distance,
  rise/decay, width, jitter) and use these as additive features.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
OUTPUT_DIR = PROJECT_DIR / "output"
SHOT7M2_DIR = PROJECT_DIR / "external_data" / "shot7m2_sample"

TARGETS = ["angle", "depth", "left_right"]
TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
HOOP = np.array([5.25, -25.0, 10.0], dtype=float)

JOINT_MAP = {
    "mid_hip": 0,
    "left_hip": 1,
    "left_knee": 2,
    "left_ankle": 3,
    "right_hip": 6,
    "right_knee": 7,
    "right_ankle": 8,
    "left_shoulder": 16,
    "left_elbow": 17,
    "left_wrist": 18,
    "right_shoulder": 23,
    "right_elbow": 24,
    "right_wrist": 25,
    "neck": 19,
}
MAPPED_JOINTS = list(JOINT_MAP.keys())
SHOT7M2_INDICES = [JOINT_MAP[j] for j in MAPPED_JOINTS]

SEGMENTS = ["full", "arm", "leg", "trunk"]


def _segment_indices() -> dict[str, np.ndarray]:
    idx = {name: i for i, name in enumerate(MAPPED_JOINTS)}
    arm = np.array(
        [
            idx["left_shoulder"],
            idx["left_elbow"],
            idx["left_wrist"],
            idx["right_shoulder"],
            idx["right_elbow"],
            idx["right_wrist"],
        ],
        dtype=int,
    )
    leg = np.array(
        [
            idx["left_hip"],
            idx["left_knee"],
            idx["left_ankle"],
            idx["right_hip"],
            idx["right_knee"],
            idx["right_ankle"],
        ],
        dtype=int,
    )
    trunk = np.array([idx["mid_hip"], idx["neck"]], dtype=int)
    full = np.arange(len(MAPPED_JOINTS), dtype=int)
    return {"full": full, "arm": arm, "leg": leg, "trunk": trunk}


SEGMENT_IDXS = _segment_indices()


def parse_triplet_int(s: str) -> tuple[int, int, int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 integers, got: {s}")
    return vals[0], vals[1], vals[2]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pulse features transfer pipeline")
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--frames-per-scale", type=int, default=1500)
    p.add_argument("--seed", type=int, default=20260216)

    p.add_argument("--shoot-threshold", type=float, default=0.3)
    p.add_argument("--context-frames", type=int, default=30)
    p.add_argument("--peak-search-half-window-shot7m2", type=int, default=10)

    p.add_argument("--axis-perm", type=str, default="0,2,1")
    p.add_argument("--axis-signs", type=str, default="1,1,1")

    p.add_argument("--window-radius", type=int, default=20)
    p.add_argument("--target-search-radius", type=int, default=25)
    p.add_argument("--smooth-kernel", type=int, default=5)

    p.add_argument("--n-pls-hoop", type=int, default=15)
    p.add_argument("--n-pls-pulse-grid", type=str, default="1,3,5")
    p.add_argument("--bw-quantile", type=float, default=0.45)
    p.add_argument("--alpha", type=float, default=10.0)

    p.add_argument("--weight-dribble", type=float, default=1.0)
    p.add_argument("--weight-move", type=float, default=1.0)
    p.add_argument("--weight-sprint", type=float, default=1.0)
    p.add_argument("--weight-temperature", type=float, default=0.3)
    p.add_argument("--min-frame-weight", type=float, default=0.05)

    p.add_argument("--base-submission", type=int, default=2503)
    p.add_argument("--blend-weights", type=str, default="0.01,0.02")
    p.add_argument("--skip-submissions", action="store_true")
    return p.parse_args()


def get_next_submission_number() -> int:
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = []
            for fp in existing:
                parts = fp.stem.split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    nums.append(int(parts[1]))
            next_num = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def parse_json_safe(s: str, frame: int) -> float:
    s = str(s).replace("nan", "null").replace("NaN", "null")
    arr = json.loads(s)
    v = arr[frame]
    if v is None:
        for o in range(1, 10):
            if frame - o >= 0 and arr[frame - o] is not None:
                return float(arr[frame - o])
            if frame + o < len(arr) and arr[frame + o] is not None:
                return float(arr[frame + o])
        return 0.0
    return float(v)


def parse_json_array_240(s: str) -> np.ndarray:
    s = str(s).replace("nan", "null").replace("NaN", "null")
    arr = json.loads(s)
    out = np.zeros(240, dtype=float)
    for i in range(min(len(arr), 240)):
        v = arr[i]
        out[i] = 0.0 if v is None else float(v)
    return out


def apply_axis_transform(x: np.ndarray, perm: tuple[int, int, int], signs: tuple[int, int, int]) -> np.ndarray:
    y = x[..., perm]
    return y * np.asarray(signs, dtype=float)


def smooth_signal(x: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return x
    k = int(kernel)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    w = np.ones(k, dtype=float) / float(k)
    return np.convolve(xp, w, mode="valid")


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def extract_window(x: np.ndarray, center: int, radius: int) -> np.ndarray:
    t = len(x)
    out = np.zeros(2 * radius + 1, dtype=float)
    for i, rel in enumerate(range(-radius, radius + 1)):
        idx = center + rel
        if idx < 0:
            idx = 0
        elif idx >= t:
            idx = t - 1
        out[i] = x[idx]
    return out


def normalize_window(w: np.ndarray) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(w)))
    if peak < 1e-12:
        return np.zeros_like(w), 0.0
    return w / peak, peak


def pulse_shape_stats(norm_w: np.ndarray, center: int) -> tuple[float, float, float, float]:
    above_half = np.where(norm_w >= 0.5)[0]
    width_half = float(len(above_half))

    left = norm_w[: center + 1]
    right = norm_w[center:]

    low_left = np.where(left <= 0.2)[0]
    rise = float(center - low_left[-1]) if len(low_left) > 0 else float(center)

    low_right = np.where(right <= 0.2)[0]
    decay = float(low_right[0]) if len(low_right) > 0 else float(len(right) - 1)

    tail = norm_w[center:]
    jitter = float(np.std(np.diff(tail))) if len(tail) > 2 else 0.0
    return width_half, rise, decay, jitter


def compute_pulse_bank(seqs_14: np.ndarray, smooth_kernel: int) -> dict[str, np.ndarray]:
    """
    seqs_14: (N, T, 14, 3)
    returns per-segment pulse bank: dict segment -> (N,T)
    """
    n = seqs_14.shape[0]
    t = seqs_14.shape[1]
    out = {seg: np.zeros((n, t), dtype=float) for seg in SEGMENTS}

    for i in range(n):
        seq = seqs_14[i]
        vel = np.diff(seq, axis=0, prepend=seq[0:1])
        speed = np.linalg.norm(vel, axis=2)  # (240,14)

        for seg in SEGMENTS:
            idxs = SEGMENT_IDXS[seg]
            pulse = speed[:, idxs].mean(axis=1)
            pulse = smooth_signal(pulse, smooth_kernel)
            out[seg][i] = pulse
    return out


def find_local_peak(signal: np.ndarray, center: int, radius: int) -> int:
    s = max(0, center - radius)
    e = min(len(signal), center + radius + 1)
    if e <= s:
        return int(np.clip(center, 0, len(signal) - 1))
    return int(s + np.argmax(signal[s:e]))


def extract_pulse_features_for_target(
    pulse_bank: dict[str, np.ndarray],
    target_frame: int,
    templates: dict[str, np.ndarray],
    window_radius: int,
    target_search_radius: int,
) -> np.ndarray:
    n = pulse_bank["full"].shape[0]
    t = pulse_bank["full"].shape[1]
    feat_rows = []
    c = window_radius

    for i in range(n):
        row = []
        peak_global = find_local_peak(pulse_bank["full"][i], target_frame, target_search_radius)

        peaks = {}
        pre_means = {}
        post_means = {}

        for seg in SEGMENTS:
            pulse = pulse_bank[seg][i]
            peak = find_local_peak(pulse, peak_global, 6)
            peaks[seg] = pulse[peak]

            w = extract_window(pulse, peak, window_radius)
            norm_w, peak_raw = normalize_window(w)
            tpl = templates[seg]

            corr = pearson_corr(norm_w, tpl)
            mse = float(np.mean((norm_w - tpl) ** 2))
            cos = cosine_sim(norm_w, tpl)
            area = float(np.sum(norm_w))
            width_half, rise, decay, jitter = pulse_shape_stats(norm_w, c)

            pre = pulse[max(0, peak - 10) : peak]
            post = pulse[peak + 1 : min(t, peak + 11)]
            pre_mean = float(np.mean(pre)) if len(pre) > 0 else 0.0
            post_mean = float(np.mean(post)) if len(post) > 0 else 0.0
            pre_means[seg] = pre_mean
            post_means[seg] = post_mean

            row.extend(
                [
                    corr,
                    mse,
                    cos,
                    peak_raw,
                    width_half,
                    rise,
                    decay,
                    jitter,
                    area,
                    float(peak - target_frame),
                    pre_mean,
                    post_mean,
                    pre_mean - post_mean,
                ]
            )

        # Cross-segment efficiency ratios
        arm_peak = max(float(peaks["arm"]), 1e-8)
        leg_peak = max(float(peaks["leg"]), 1e-8)
        trunk_peak = max(float(peaks["trunk"]), 1e-8)
        full_peak = max(float(peaks["full"]), 1e-8)

        row.extend(
            [
                arm_peak / leg_peak,
                arm_peak / trunk_peak,
                full_peak / leg_peak,
                float(pre_means["arm"] - post_means["arm"]),
                float(pre_means["leg"] - post_means["leg"]),
            ]
        )

        feat_rows.append(row)

    return np.asarray(feat_rows, dtype=float)


def load_competition_14j_sequences(
    df: pd.DataFrame,
    axis_perm: tuple[int, int, int],
    axis_signs: tuple[int, int, int],
) -> np.ndarray:
    n = len(df)
    seqs = np.zeros((n, 240, len(MAPPED_JOINTS), 3), dtype=float)

    for j_idx, name in enumerate(MAPPED_JOINTS):
        x_arr = np.stack(df[f"{name}_x"].apply(parse_json_array_240).to_list(), axis=0)
        y_arr = np.stack(df[f"{name}_y"].apply(parse_json_array_240).to_list(), axis=0)
        z_arr = np.stack(df[f"{name}_z"].apply(parse_json_array_240).to_list(), axis=0)
        seqs[:, :, j_idx, 0] = x_arr
        seqs[:, :, j_idx, 1] = y_arr
        seqs[:, :, j_idx, 2] = z_arr

    return apply_axis_transform(seqs, axis_perm, axis_signs)


def build_hoop_features_from_seqs(seqs_14: np.ndarray, frame: int) -> np.ndarray:
    f = int(np.clip(frame, 0, 239))
    prev_f = max(0, f - 1)
    next_f = min(239, f + 1)

    pos = seqs_14[:, f]
    vel = (seqs_14[:, next_f] - seqs_14[:, prev_f]) / max(next_f - prev_f, 1)
    acc = seqs_14[:, next_f] - 2.0 * seqs_14[:, f] + seqs_14[:, prev_f]

    hoop_rel = pos - HOOP[np.newaxis, np.newaxis, :]
    return np.hstack([hoop_rel.reshape(len(seqs_14), -1), vel.reshape(len(seqs_14), -1), acc.reshape(len(seqs_14), -1)])


def load_shot7m2_pulse_templates(
    args: argparse.Namespace,
    axis_perm: tuple[int, int, int],
    axis_signs: tuple[int, int, int],
) -> dict:
    print("Loading SHOT7M2 and building pulse templates...")
    poses = np.load(SHOT7M2_DIR / "train" / "train_dictionary_poses.npy", allow_pickle=True).item()
    actions = np.load(SHOT7M2_DIR / "train" / "train_dictionary_actions.npy", allow_pickle=True).item()

    labels = actions["label_array"]
    vocab = actions["vocabulary"]
    fnm = actions["frame_number_map"]

    shoot_idx = vocab.index("action_Shoot")
    dribble_idx = vocab.index("action_Dribble")
    move_idx = vocab.index("action_Move")
    sprint_idx = vocab.index("action_Sprint")

    windows_by_segment = {seg: [] for seg in SEGMENTS}
    frame_weights = []
    frame_confs = []
    frame_mob = []

    peak_search_half = int(max(1, args.peak_search_half_window_shot7m2))

    for key, seq in poses["sequences"]["keypoints"].items():
        start, end = fnm[key]
        shoot = labels[shoot_idx, start:end]
        drib = labels[dribble_idx, start:end]
        move = labels[move_idx, start:end]
        sprint = labels[sprint_idx, start:end]

        shoot_frames = np.where(shoot > args.shoot_threshold)[0]
        if len(shoot_frames) == 0:
            continue

        seq_14 = seq[:, 0, SHOT7M2_INDICES, :].astype(float)
        seq_14 = apply_axis_transform(seq_14, axis_perm, axis_signs)
        pulse_bank = compute_pulse_bank(seq_14[np.newaxis, ...], args.smooth_kernel)

        for f in shoot_frames:
            c0 = max(0, f - args.context_frames)
            c1 = min(len(shoot), f + args.context_frames + 1)

            d_mean = float(np.clip(np.mean(drib[c0:c1]), 0.0, 1.0))
            m_mean = float(np.clip(np.mean(move[c0:c1]), 0.0, 1.0))
            s_mean = float(np.clip(np.mean(sprint[c0:c1]), 0.0, 1.0))

            mobility = (
                args.weight_dribble * d_mean
                + args.weight_move * m_mean
                + args.weight_sprint * s_mean
            )
            ft_like = np.exp(-mobility / max(args.weight_temperature, 1e-8))
            conf = max(float(shoot[f]), 0.0)
            weight = conf * (args.min_frame_weight + (1.0 - args.min_frame_weight) * ft_like)

            peak = find_local_peak(pulse_bank["full"][0], int(f), peak_search_half)

            for seg in SEGMENTS:
                w = extract_window(pulse_bank[seg][0], peak, args.window_radius)
                nw, _ = normalize_window(w)
                windows_by_segment[seg].append(nw)

            frame_weights.append(weight)
            frame_confs.append(conf)
            frame_mob.append(mobility)

    if len(frame_weights) == 0:
        raise RuntimeError("No SHOT7M2 shoot frames were collected for pulse templates.")

    weights = np.asarray(frame_weights, dtype=float)
    confs = np.asarray(frame_confs, dtype=float)
    mobs = np.asarray(frame_mob, dtype=float)

    total = len(weights)
    max_frames = min(total, int(args.scale * args.frames_per_scale))
    keep_idx = np.argsort(-weights)[:max_frames]
    keep_weights = weights[keep_idx]

    templates = {}
    for seg in SEGMENTS:
        w = np.asarray(windows_by_segment[seg], dtype=float)[keep_idx]
        templates[seg] = np.average(w, axis=0, weights=keep_weights)

    stats = {
        "total_shooting_frames": int(total),
        "kept_frames": int(max_frames),
        "kept_ratio": float(max_frames / total),
        "weight_mean": float(np.mean(weights)),
        "weight_median": float(np.median(weights)),
        "weight_max": float(np.max(weights)),
        "kept_weight_mean": float(np.mean(keep_weights)),
        "kept_conf_mean": float(np.mean(confs[keep_idx])),
        "kept_mobility_mean": float(np.mean(mobs[keep_idx])),
        "template_length": int(2 * args.window_radius + 1),
    }

    print(f"  total shoot frames: {stats['total_shooting_frames']}")
    print(f"  kept frames (scale={args.scale}): {stats['kept_frames']}")
    print(f"  kept ratio: {stats['kept_ratio']:.6f}")
    print(f"  kept conf mean: {stats['kept_conf_mean']:.6f}")
    print(f"  kept mobility mean: {stats['kept_mobility_mean']:.6f}")

    return {"templates": templates, "stats": stats}


def run_per_player_loo(
    hoop_features: np.ndarray,
    pulse_features: np.ndarray | None,
    targets: np.ndarray,
    player_ids: np.ndarray,
    use_pulse: bool,
    n_pls_hoop: int,
    n_pls_pulse: int,
    bw_quantile: float,
    alpha: float,
) -> np.ndarray:
    n = len(targets)
    preds = np.zeros(n, dtype=float)

    for pid in sorted(np.unique(player_ids)):
        mask = player_ids == pid
        idx = np.where(mask)[0]
        n_player = len(idx)

        xh = hoop_features[mask]
        y = targets[mask]
        xp = pulse_features[mask] if (use_pulse and pulse_features is not None) else None

        for i_local in range(n_player):
            i_global = idx[i_local]
            loo = np.ones(n_player, dtype=bool)
            loo[i_local] = False

            xh_tr = xh[loo]
            xh_te = xh[~loo]
            y_tr = y[loo]

            sc_h = StandardScaler()
            xh_tr_s = sc_h.fit_transform(xh_tr)
            xh_te_s = sc_h.transform(xh_te)

            parts_tr = [xh_tr_s]
            parts_te = [xh_te_s]

            nph = min(n_pls_hoop, xh_tr_s.shape[1], xh_tr_s.shape[0] - 1)
            if nph > 0:
                pls_h = PLSRegression(n_components=nph)
                pls_h.fit(xh_tr_s, y_tr)
                parts_tr.append(pls_h.transform(xh_tr_s))
                parts_te.append(pls_h.transform(xh_te_s))

            if use_pulse and xp is not None:
                xp_tr = xp[loo]
                xp_te = xp[~loo]

                sc_p = StandardScaler()
                xp_tr_s = sc_p.fit_transform(xp_tr)
                xp_te_s = sc_p.transform(xp_te)

                npp = min(n_pls_pulse, xp_tr_s.shape[1], xp_tr_s.shape[0] - 1)
                if npp > 0:
                    pls_p = PLSRegression(n_components=npp)
                    pls_p.fit(xp_tr_s, y_tr)
                    parts_tr.append(pls_p.transform(xp_tr_s))
                    parts_te.append(pls_p.transform(xp_te_s))

            x_tr = np.hstack(parts_tr)
            x_te = np.hstack(parts_te)

            d = np.linalg.norm(x_tr - x_te, axis=1)
            bw = np.quantile(d, bw_quantile)
            w = np.exp(-0.5 * (d / max(bw, 1e-8)) ** 2)
            ws = np.diag(np.sqrt(w))

            ridge = Ridge(alpha=alpha)
            ridge.fit(ws @ x_tr, ws @ y_tr)
            preds[i_global] = ridge.predict(x_te)[0]

    return preds


def predict_test_target(
    xh_train: np.ndarray,
    xp_train: np.ndarray,
    y_train: np.ndarray,
    train_pids: np.ndarray,
    xh_test: np.ndarray,
    xp_test: np.ndarray,
    test_pids: np.ndarray,
    n_pls_hoop: int,
    n_pls_pulse: int,
    bw_quantile: float,
    alpha: float,
) -> np.ndarray:
    preds = np.zeros(len(xh_test), dtype=float)

    for pid in sorted(np.unique(train_pids)):
        tr_mask = train_pids == pid
        te_mask = test_pids == pid
        te_idx = np.where(te_mask)[0]
        if len(te_idx) == 0:
            continue

        xh = xh_train[tr_mask]
        xp = xp_train[tr_mask]
        y = y_train[tr_mask]
        xh_te = xh_test[te_mask]
        xp_te = xp_test[te_mask]

        sc_h = StandardScaler()
        xh_s = sc_h.fit_transform(xh)
        xh_te_s = sc_h.transform(xh_te)

        sc_p = StandardScaler()
        xp_s = sc_p.fit_transform(xp)
        xp_te_s = sc_p.transform(xp_te)

        parts_tr = [xh_s]
        parts_te = [xh_te_s]

        nph = min(n_pls_hoop, xh_s.shape[1], xh_s.shape[0] - 1)
        if nph > 0:
            pls_h = PLSRegression(n_components=nph)
            pls_h.fit(xh_s, y)
            parts_tr.append(pls_h.transform(xh_s))
            parts_te.append(pls_h.transform(xh_te_s))

        npp = min(n_pls_pulse, xp_s.shape[1], xp_s.shape[0] - 1)
        if npp > 0:
            pls_p = PLSRegression(n_components=npp)
            pls_p.fit(xp_s, y)
            parts_tr.append(pls_p.transform(xp_s))
            parts_te.append(pls_p.transform(xp_te_s))

        x_tr = np.hstack(parts_tr)
        x_te = np.hstack(parts_te)

        for i in range(len(x_te)):
            d = np.linalg.norm(x_tr - x_te[i], axis=1)
            bw = np.quantile(d, bw_quantile)
            w = np.exp(-0.5 * (d / max(bw, 1e-8)) ** 2)
            ws = np.diag(np.sqrt(w))
            ridge = Ridge(alpha=alpha)
            ridge.fit(ws @ x_tr, ws @ y)
            preds[te_idx[i]] = ridge.predict(x_te[i : i + 1])[0]

    return preds


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    axis_perm = parse_triplet_int(args.axis_perm)
    axis_signs = parse_triplet_int(args.axis_signs)
    n_pls_pulse_grid = parse_int_list(args.n_pls_pulse_grid)
    blend_weights = parse_float_list(args.blend_weights)

    print("=" * 80)
    print("PULSE FEATURES TRANSFER")
    print("=" * 80)
    print(f"axis_perm={axis_perm}, axis_signs={axis_signs}")
    print(f"scale={args.scale}, frames_per_scale={args.frames_per_scale}")
    print(f"window_radius={args.window_radius}, target_search_radius={args.target_search_radius}")
    print(f"n_pls_pulse_grid={n_pls_pulse_grid}")

    shot_data = load_shot7m2_pulse_templates(args, axis_perm, axis_signs)
    templates = shot_data["templates"]

    print("\nLoading competition data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    train_pids = train_df["participant_id"].values
    test_pids = test_df["participant_id"].values

    scalers = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}
    y_scaled = {
        t: scalers[t].transform(train_df[t].values.reshape(-1, 1)).ravel() for t in TARGETS
    }

    print("Building 14-joint sequences...")
    seq_train = load_competition_14j_sequences(train_df, axis_perm, axis_signs)
    seq_test = load_competition_14j_sequences(test_df, axis_perm, axis_signs)

    print("Computing pulse banks...")
    pulse_train = compute_pulse_bank(seq_train, args.smooth_kernel)
    pulse_test = compute_pulse_bank(seq_test, args.smooth_kernel)

    train_hoop = {}
    train_pulse_feats = {}
    test_hoop = {}
    test_pulse_feats = {}

    print("\nExtracting target-specific features...")
    for target in TARGETS:
        frame = TARGET_FRAMES[target]
        print(f"  {target}: frame={frame}")
        train_hoop[target] = build_hoop_features_from_seqs(seq_train, frame)
        test_hoop[target] = build_hoop_features_from_seqs(seq_test, frame)

        train_pulse_feats[target] = extract_pulse_features_for_target(
            pulse_train,
            frame,
            templates,
            args.window_radius,
            args.target_search_radius,
        )
        test_pulse_feats[target] = extract_pulse_features_for_target(
            pulse_test,
            frame,
            templates,
            args.window_radius,
            args.target_search_radius,
        )

        print(
            f"    hoop={train_hoop[target].shape} pulse={train_pulse_feats[target].shape}"
        )

    print("\n" + "=" * 80)
    print("HONEST LOO EVALUATION")
    print("=" * 80)

    configs = [("baseline_hoop_only", False, 0)]
    for npp in n_pls_pulse_grid:
        configs.append((f"hoop_plus_pulse_{npp}pls", True, npp))

    results = {}

    for name, use_pulse, npp in configs:
        t0 = time.time()
        m = {}
        print(f"\nConfig: {name}")
        for target in TARGETS:
            pred = run_per_player_loo(
                hoop_features=train_hoop[target],
                pulse_features=train_pulse_feats[target] if use_pulse else None,
                targets=y_scaled[target],
                player_ids=train_pids,
                use_pulse=use_pulse,
                n_pls_hoop=args.n_pls_hoop,
                n_pls_pulse=npp,
                bw_quantile=args.bw_quantile,
                alpha=args.alpha,
            )
            mse = float(np.mean((pred - y_scaled[target]) ** 2))
            m[target] = mse
            print(f"  {target}: {mse:.12f}")
        m["mean"] = float(np.mean([m[t] for t in TARGETS]))
        m["time_s"] = float(time.time() - t0)
        results[name] = m
        print(f"  mean: {m['mean']:.12f} [{m['time_s']:.3f}s]")

    baseline = results["baseline_hoop_only"]["mean"]
    print("\nComparison vs baseline:")
    for cfg, vals in results.items():
        delta_pct = (vals["mean"] - baseline) / baseline * 100.0
        vals["delta_pct"] = float(delta_pct)
        print(f"  {cfg}: mean={vals['mean']:.12f}, delta_pct={delta_pct:+.12f}")

    non_base = [k for k in results.keys() if k != "baseline_hoop_only"]
    best_name = min(non_base, key=lambda k: results[k]["mean"]) if non_base else "baseline_hoop_only"
    best_npp = 0 if best_name == "baseline_hoop_only" else int(best_name.split("_")[-1].replace("pls", ""))
    print(f"\nBest config: {best_name} (n_pls_pulse={best_npp})")

    submissions_created = []
    if not args.skip_submissions:
        print("\n" + "=" * 80)
        print("GENERATING SUBMISSIONS")
        print("=" * 80)

        base_path = SUBMISSION_DIR / f"submission_{args.base_submission}.csv"
        if not base_path.exists():
            raise FileNotFoundError(f"Base submission not found: {base_path}")
        base = pd.read_csv(base_path)

        standalone = base.copy()
        for target, col in zip(TARGETS, TARGET_COLS):
            preds = predict_test_target(
                xh_train=train_hoop[target],
                xp_train=train_pulse_feats[target],
                y_train=y_scaled[target],
                train_pids=train_pids,
                xh_test=test_hoop[target],
                xp_test=test_pulse_feats[target],
                test_pids=test_pids,
                n_pls_hoop=args.n_pls_hoop,
                n_pls_pulse=best_npp,
                bw_quantile=args.bw_quantile,
                alpha=args.alpha,
            )
            standalone[col] = np.clip(preds, 0.0, 1.0)

        sub_num = get_next_submission_number()
        standalone_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        standalone.to_csv(standalone_path, index=False)
        submissions_created.append(
            {
                "submission_number": int(sub_num),
                "path": str(standalone_path),
                "type": "standalone_pulse_transfer",
                "blend_weight_pulse": 1.0,
                "best_config": best_name,
            }
        )
        print(f"  standalone: submission_{sub_num}.csv")

        for bw in blend_weights:
            blended = base.copy()
            for col in TARGET_COLS:
                blended[col] = np.clip(
                    bw * standalone[col].values + (1.0 - bw) * base[col].values,
                    0.0,
                    1.0,
                )

            sub_num = get_next_submission_number()
            blend_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
            blended.to_csv(blend_path, index=False)
            submissions_created.append(
                {
                    "submission_number": int(sub_num),
                    "path": str(blend_path),
                    "type": "blend_with_base",
                    "blend_weight_pulse": float(bw),
                    "blend_weight_base": float(1.0 - bw),
                    "base_submission": int(args.base_submission),
                    "source_standalone_submission": int(submissions_created[0]["submission_number"]),
                }
            )
            print(
                f"  blend: submission_{sub_num}.csv "
                f"(pulse={bw:.6f}, base={1.0-bw:.6f})"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_json = OUTPUT_DIR / f"pulse_features_transfer_run_{ts}.json"
    run_md = OUTPUT_DIR / f"pulse_features_transfer_details_{ts}.md"

    payload = {
        "timestamp": ts,
        "command": " ".join(sys.argv),
        "config": {
            "scale": args.scale,
            "frames_per_scale": args.frames_per_scale,
            "seed": args.seed,
            "shoot_threshold": args.shoot_threshold,
            "context_frames": args.context_frames,
            "peak_search_half_window_shot7m2": args.peak_search_half_window_shot7m2,
            "axis_perm": axis_perm,
            "axis_signs": axis_signs,
            "window_radius": args.window_radius,
            "target_search_radius": args.target_search_radius,
            "smooth_kernel": args.smooth_kernel,
            "n_pls_hoop": args.n_pls_hoop,
            "n_pls_pulse_grid": n_pls_pulse_grid,
            "bw_quantile": args.bw_quantile,
            "alpha": args.alpha,
            "weight_dribble": args.weight_dribble,
            "weight_move": args.weight_move,
            "weight_sprint": args.weight_sprint,
            "weight_temperature": args.weight_temperature,
            "min_frame_weight": args.min_frame_weight,
            "base_submission": args.base_submission,
            "blend_weights": blend_weights,
            "skip_submissions": args.skip_submissions,
        },
        "shot7m2_stats": shot_data["stats"],
        "results": results,
        "best_config_name": best_name,
        "best_n_pls_pulse": int(best_npp),
        "submissions_created": submissions_created,
    }
    run_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Pulse Features Transfer Run",
        "",
        f"- Timestamp: `{ts}`",
        f"- Command: `{payload['command']}`",
        "",
        "## SHOT7M2 Pulse Template Stats",
    ]
    for k, v in shot_data["stats"].items():
        lines.append(f"- {k}: `{v}`")

    lines += ["", "## Honest LOO Results"]
    for cfg, vals in results.items():
        lines.append(
            "- "
            + cfg
            + f": angle={vals['angle']}, depth={vals['depth']}, left_right={vals['left_right']}, "
            + f"mean={vals['mean']}, delta_pct={vals['delta_pct']}"
        )

    lines += [
        "",
        "## Best Config",
        f"- name: `{best_name}`",
        f"- n_pls_pulse: `{best_npp}`",
        "",
        "## Submissions",
    ]
    if submissions_created:
        for row in submissions_created:
            lines.append(f"- {row}")
    else:
        lines.append("- None (`--skip-submissions` used)")

    run_md.write_text("\n".join(lines) + "\n")

    print("\nArtifacts:")
    print(f"  {run_json}")
    print(f"  {run_md}")
    print("\nDone.")


if __name__ == "__main__":
    main()

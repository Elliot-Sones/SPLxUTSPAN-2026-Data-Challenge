"""
Energy Wave Transfer Pipeline

Implements a dynamic biomechanical signal inspired by "energy wave" sequencing:
1. Build a pro-template waveform from external NBA kinematic traces.
2. Extract per-shot 3D velocity-energy waveforms from competition skeleton data.
3. Compute Pro_Form_Match_Score and related wave-shape diagnostics.
4. Add these features to the existing hoop-relative locally weighted Ridge pipeline.
5. Evaluate with honest per-player LOO and optionally generate submission candidates.

Note:
- The workspace does not contain raw NBA videos, so this implementation uses available
  external kinematic traces (`external_data/spacejam/path_detail.csv`) as the transfer
  source for the pro-template waveform.
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
EXTERNAL_DIR = PROJECT_DIR / "external_data"

SPACEJAM_PATH_DETAIL = EXTERNAL_DIR / "spacejam" / "path_detail.csv"

TARGETS = ["angle", "depth", "left_right"]
TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
HOOP = np.array([5.25, -25.0, 10.0], dtype=float)

ENERGY_JOINTS = [
    "mid_hip",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]
JOINT_INDEX = {name: i for i, name in enumerate(ENERGY_JOINTS)}

PROXIMAL_JOINTS = ["mid_hip", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
TRUNK_JOINTS = ["neck", "left_shoulder", "right_shoulder"]
RIGHT_DISTAL_JOINTS = ["right_shoulder", "right_elbow", "right_wrist"]
LEFT_DISTAL_JOINTS = ["left_shoulder", "left_elbow", "left_wrist"]

ENERGY_FEATURE_NAMES = [
    "pro_form_match_score",
    "pro_corr_distance",
    "pro_wave_l2",
    "wave_peak_lag_norm",
    "wave_peak_height_z",
    "wave_rise_slope",
    "wave_decay_slope",
    "prox_to_distal_peak_lag_norm",
    "prox_peak_height_z",
    "dist_peak_height_z",
    "dist_to_total_peak_ratio",
    "double_peak_ratio",
    "num_local_peaks",
    "flat_peak_fraction",
    "template_peak_diff_z",
    "total_wave_auc_z",
    "shooting_side_right",
]

ENERGY_FEATURE_SETS: dict[str, list[str]] = {
    "full": ENERGY_FEATURE_NAMES,
    "core": [
        "pro_form_match_score",
        "pro_corr_distance",
        "wave_peak_lag_norm",
        "prox_to_distal_peak_lag_norm",
        "double_peak_ratio",
        "num_local_peaks",
        "flat_peak_fraction",
        "shooting_side_right",
    ],
    "match_only": [
        "pro_form_match_score",
        "pro_corr_distance",
        "wave_peak_lag_norm",
        "prox_to_distal_peak_lag_norm",
        "shooting_side_right",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Energy Wave transfer pipeline")
    p.add_argument("--scale", type=int, default=8)
    p.add_argument("--sequences-per-scale", type=int, default=8)
    p.add_argument("--wave-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260216)
    p.add_argument("--n-pls-hoop", type=int, default=15)
    p.add_argument("--n-pls-energy-grid", type=str, default="0,2,4,6")
    p.add_argument(
        "--energy-feature-set",
        type=str,
        default="core",
        choices=sorted(ENERGY_FEATURE_SETS.keys()),
    )
    p.add_argument("--include-raw-energy", action="store_true")
    p.add_argument("--bw-quantile", type=float, default=0.45)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--base-submission", type=int, default=2503)
    p.add_argument("--blend-weights", type=str, default="0.05,0.10,0.15,0.20,0.30")
    p.add_argument("--skip-submissions", action="store_true")
    return p.parse_args()


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def get_next_submission_number() -> int:
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums: list[int] = []
            for fp in existing:
                stem_parts = fp.stem.split("_")
                if len(stem_parts) == 2 and stem_parts[1].isdigit():
                    nums.append(int(stem_parts[1]))
            next_num = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{next_num}.csv").touch(exist_ok=True)
            return next_num
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def fill_nan_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    mask = np.isnan(x)
    if not np.any(mask):
        return x
    if np.all(mask):
        return np.zeros_like(x)
    valid_idx = np.where(~mask)[0]
    filled = x.copy()
    filled[mask] = np.interp(np.where(mask)[0], valid_idx, x[valid_idx])
    return filled


def parse_series_json(s: str) -> np.ndarray:
    s = str(s).replace("nan", "null").replace("NaN", "null")
    arr = json.loads(s)
    series = np.array([np.nan if v is None else float(v) for v in arr], dtype=float)
    return fill_nan_1d(series)


def parse_json_safe_frame(s: str, frame: int) -> float:
    series = parse_series_json(s)
    if frame < 0:
        return float(series[0])
    if frame >= len(series):
        return float(series[-1])
    return float(series[frame])


def resample_wave(wave: np.ndarray, target_len: int) -> np.ndarray:
    wave = np.asarray(wave, dtype=float)
    if wave.size == 0:
        return np.zeros(target_len, dtype=float)
    if len(wave) == target_len:
        return wave.copy()
    x_old = np.linspace(0.0, 1.0, len(wave))
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, wave)


def smooth_wave(wave: np.ndarray, window: int = 5) -> np.ndarray:
    wave = np.asarray(wave, dtype=float)
    if window <= 1 or len(wave) < 3:
        return wave.copy()
    window = min(window, len(wave) if len(wave) % 2 == 1 else len(wave) - 1)
    window = max(window, 3)
    if window % 2 == 0:
        window -= 1
    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(wave, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    if len(smoothed) != len(wave):
        smoothed = smoothed[: len(wave)]
    return smoothed


def zscore_wave(wave: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave, dtype=float)
    mean = float(np.mean(wave))
    std = float(np.std(wave))
    if std < 1e-8:
        return np.zeros_like(wave)
    return (wave - mean) / std


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-8 or sy < 1e-8:
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return c


def count_local_peaks(x: np.ndarray, min_prominence: float = 0.2) -> tuple[int, float]:
    """
    Returns:
    - number of local peaks
    - second_peak_to_first_peak ratio (0 if <2 peaks)
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return 0, 0.0

    peaks: list[float] = []
    x_std = float(np.std(x))
    prom = min_prominence * max(x_std, 1e-8)
    for i in range(1, len(x) - 1):
        if x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            neigh = max(x[i - 1], x[i + 1])
            if (x[i] - neigh) >= prom:
                peaks.append(float(x[i]))

    if not peaks:
        return 0, 0.0
    peaks.sort(reverse=True)
    if len(peaks) == 1:
        return 1, 0.0
    ratio = peaks[1] / max(peaks[0], 1e-8)
    return len(peaks), float(ratio)


def load_pro_template(scale: int, sequences_per_scale: int, wave_length: int) -> tuple[np.ndarray, dict]:
    """
    Build a pro-template waveform from external NBA kinematic traces.
    """
    if not SPACEJAM_PATH_DETAIL.exists():
        raise FileNotFoundError(f"Missing external trace data: {SPACEJAM_PATH_DETAIL}")

    df = pd.read_csv(SPACEJAM_PATH_DETAIL)
    df = df.sort_values(["pid", "t"]).reset_index(drop=True)

    pid_change = df["pid"].ne(df["pid"].shift(1))
    t_reset = df["t"].lt(df["t"].shift(1).fillna(df["t"]))
    seq_start = pid_change | t_reset
    seq_id = seq_start.cumsum() - 1
    df["seq_id"] = seq_id

    seq_infos = (
        df.groupby("seq_id")
        .agg(
            pid=("pid", "first"),
            rows=("t", "size"),
            rt=("rt", "first"),
        )
        .reset_index()
    )
    seq_infos = seq_infos.sort_values(["rows", "pid"], ascending=[False, True]).reset_index(drop=True)

    max_sequences = min(len(seq_infos), max(1, scale * sequences_per_scale))
    keep_seq_ids = set(seq_infos.loc[: max_sequences - 1, "seq_id"].tolist())

    waves: list[np.ndarray] = []
    for sid, grp in df.groupby("seq_id"):
        if sid not in keep_seq_ids:
            continue
        cv = fill_nan_1d(grp["cv"].to_numpy(dtype=float))
        # Kinetic proxy: proportional to speed^2.
        wave = np.square(np.clip(cv, a_min=0.0, a_max=None))
        wave = smooth_wave(wave, window=7)
        wave = resample_wave(wave, wave_length)
        wave = zscore_wave(wave)
        waves.append(wave)

    if not waves:
        raise RuntimeError("No sequences available for pro template.")

    wave_matrix = np.vstack(waves)
    template = np.mean(wave_matrix, axis=0)
    template = zscore_wave(template)

    template_stats = {
        "source_file": str(SPACEJAM_PATH_DETAIL),
        "total_sequences_available": int(len(seq_infos)),
        "used_sequences": int(len(waves)),
        "rows_mean_used": float(seq_infos.loc[: max_sequences - 1, "rows"].mean()),
        "rows_min_used": int(seq_infos.loc[: max_sequences - 1, "rows"].min()),
        "rows_max_used": int(seq_infos.loc[: max_sequences - 1, "rows"].max()),
        "template_peak_index": int(np.argmax(template)),
        "template_peak_value": float(np.max(template)),
    }
    return template, template_stats


def load_joint_timeseries(df: pd.DataFrame, joints: list[str]) -> np.ndarray:
    """
    Returns positions with shape (N, T, J, 3).
    """
    n = len(df)
    first_col = f"{joints[0]}_x"
    first_series = parse_series_json(df.iloc[0][first_col])
    t_len = len(first_series)

    positions = np.zeros((n, t_len, len(joints), 3), dtype=float)
    for j_idx, joint in enumerate(joints):
        for c_idx, coord in enumerate(["x", "y", "z"]):
            col = f"{joint}_{coord}"
            stacked = np.vstack([parse_series_json(v) for v in df[col].values])
            positions[:, :, j_idx, c_idx] = stacked
    return positions


def choose_shooting_side(vel: np.ndarray) -> bool:
    """
    Returns True for right shooting side, False for left.
    vel: (T, J, 3)
    """
    rw = JOINT_INDEX["right_wrist"]
    lw = JOINT_INDEX["left_wrist"]
    right_speed = np.linalg.norm(vel[:, rw, :], axis=1)
    left_speed = np.linalg.norm(vel[:, lw, :], axis=1)
    return float(np.max(right_speed)) >= float(np.max(left_speed))


def compute_energy_waves(shot_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    shot_pos: (T, J, 3)
    Returns:
    - total wave
    - proximal wave
    - distal wave
    - shooting_side_right
    """
    vel = np.gradient(shot_pos, axis=0)
    speed_sq = np.sum(vel * vel, axis=2)  # (T, J)

    proximal_idx = [JOINT_INDEX[j] for j in PROXIMAL_JOINTS]
    trunk_idx = [JOINT_INDEX[j] for j in TRUNK_JOINTS]
    side_right = choose_shooting_side(vel)
    distal_joints = RIGHT_DISTAL_JOINTS if side_right else LEFT_DISTAL_JOINTS
    distal_idx = [JOINT_INDEX[j] for j in distal_joints]

    total_idx = sorted(set(proximal_idx + trunk_idx + distal_idx))

    wave_total = np.mean(speed_sq[:, total_idx], axis=1)
    wave_prox = np.mean(speed_sq[:, proximal_idx], axis=1)
    wave_dist = np.mean(speed_sq[:, distal_idx], axis=1)
    return wave_total, wave_prox, wave_dist, side_right


def build_wave_feature_row(
    wave_total: np.ndarray,
    wave_prox: np.ndarray,
    wave_dist: np.ndarray,
    pro_template: np.ndarray,
) -> np.ndarray:
    total_z = zscore_wave(wave_total)
    prox_z = zscore_wave(wave_prox)
    dist_z = zscore_wave(wave_dist)
    temp_z = zscore_wave(pro_template)

    corr = safe_corr(total_z, temp_z)
    corr_dist = 1.0 - corr
    l2 = float(np.mean((total_z - temp_z) ** 2))

    peak_idx = int(np.argmax(total_z))
    template_peak_idx = int(np.argmax(temp_z))
    lag_norm = (peak_idx - template_peak_idx) / float(len(total_z))
    peak_height = float(total_z[peak_idx])

    if peak_idx > 0:
        pre_idx = int(np.argmin(total_z[: peak_idx + 1]))
    else:
        pre_idx = 0
    rise_denom = max(1, peak_idx - pre_idx)
    rise_slope = float((total_z[peak_idx] - total_z[pre_idx]) / rise_denom)

    if peak_idx < len(total_z) - 1:
        post_min_rel = int(np.argmin(total_z[peak_idx:]))
        post_min_idx = peak_idx + post_min_rel
        decay_denom = max(1, post_min_idx - peak_idx)
        decay_slope = float((total_z[post_min_idx] - total_z[peak_idx]) / decay_denom)
    else:
        decay_slope = 0.0

    prox_peak_idx = int(np.argmax(prox_z))
    dist_peak_idx = int(np.argmax(dist_z))
    prox_dist_lag = (dist_peak_idx - prox_peak_idx) / float(len(total_z))
    prox_peak_h = float(prox_z[prox_peak_idx])
    dist_peak_h = float(dist_z[dist_peak_idx])
    dist_to_total_ratio = float(dist_peak_h / max(abs(peak_height), 1e-8))

    n_peaks, second_ratio = count_local_peaks(total_z, min_prominence=0.2)
    flat_peak_fraction = float(np.mean(total_z >= 0.8 * peak_height)) if peak_height > 0 else 0.0

    template_peak_diff = float(peak_height - temp_z[template_peak_idx])
    auc_z = float(np.mean(total_z))

    return np.array(
        [
            corr,
            corr_dist,
            l2,
            lag_norm,
            peak_height,
            rise_slope,
            decay_slope,
            prox_dist_lag,
            prox_peak_h,
            dist_peak_h,
            dist_to_total_ratio,
            second_ratio,
            float(n_peaks),
            flat_peak_fraction,
            template_peak_diff,
            auc_z,
        ],
        dtype=float,
    )


def build_energy_feature_matrix(
    df: pd.DataFrame,
    pro_template: np.ndarray,
    wave_length: int,
) -> np.ndarray:
    positions = load_joint_timeseries(df, ENERGY_JOINTS)  # (N, T, J, 3)
    n = len(df)
    x = np.zeros((n, len(ENERGY_FEATURE_NAMES)), dtype=float)

    for i in range(n):
        wave_total, wave_prox, wave_dist, side_right = compute_energy_waves(positions[i])
        wave_total = zscore_wave(resample_wave(smooth_wave(wave_total, window=5), wave_length))
        wave_prox = zscore_wave(resample_wave(smooth_wave(wave_prox, window=5), wave_length))
        wave_dist = zscore_wave(resample_wave(smooth_wave(wave_dist, window=5), wave_length))

        row = build_wave_feature_row(wave_total, wave_prox, wave_dist, pro_template)
        x[i, :-1] = row
        x[i, -1] = 1.0 if side_right else 0.0
    return x


def select_energy_feature_set(x: np.ndarray, feature_set: str) -> tuple[np.ndarray, list[str]]:
    selected = ENERGY_FEATURE_SETS[feature_set]
    name_to_idx = {name: i for i, name in enumerate(ENERGY_FEATURE_NAMES)}
    indices = [name_to_idx[name] for name in selected]
    return x[:, indices], selected


def build_hoop_features(df: pd.DataFrame, frame: int) -> np.ndarray:
    n = len(df)
    kp_names = sorted(
        set(c.rsplit("_", 1)[0] for c in df.columns if c.endswith(("_x", "_y", "_z")))
    )
    positions = np.zeros((n, len(kp_names), 3), dtype=float)
    for k_idx, kp in enumerate(kp_names):
        for c_idx, coord in enumerate(["x", "y", "z"]):
            col = f"{kp}_{coord}"
            positions[:, k_idx, c_idx] = df[col].apply(lambda s: parse_json_safe_frame(s, frame)).values
    hoop_rel = positions - HOOP[np.newaxis, np.newaxis, :]
    return hoop_rel.reshape(n, -1)


def run_per_player_loo(
    hoop_features: np.ndarray,
    energy_features: np.ndarray | None,
    targets: np.ndarray,
    player_ids: np.ndarray,
    use_energy: bool,
    include_raw_energy: bool,
    n_pls_hoop: int,
    n_pls_energy: int,
    bw_quantile: float,
    alpha: float,
) -> np.ndarray:
    n = len(targets)
    preds = np.zeros(n, dtype=float)
    players = sorted(np.unique(player_ids))

    for pid in players:
        mask = player_ids == pid
        idx = np.where(mask)[0]
        n_player = len(idx)

        xh = hoop_features[mask]
        y = targets[mask]
        xe = energy_features[mask] if use_energy and energy_features is not None else None

        for i_local in range(n_player):
            i_global = idx[i_local]
            loo_mask = np.ones(n_player, dtype=bool)
            loo_mask[i_local] = False

            xh_tr = xh[loo_mask]
            xh_te = xh[~loo_mask]
            y_tr = y[loo_mask]

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

            if use_energy and xe is not None:
                xe_tr = xe[loo_mask]
                xe_te = xe[~loo_mask]
                sc_e = StandardScaler()
                xe_tr_s = sc_e.fit_transform(xe_tr)
                xe_te_s = sc_e.transform(xe_te)
                if include_raw_energy:
                    parts_tr.append(xe_tr_s)
                    parts_te.append(xe_te_s)

                npe = min(n_pls_energy, xe_tr_s.shape[1], xe_tr_s.shape[0] - 1)
                if npe > 0:
                    pls_e = PLSRegression(n_components=npe)
                    pls_e.fit(xe_tr_s, y_tr)
                    parts_tr.append(pls_e.transform(xe_tr_s))
                    parts_te.append(pls_e.transform(xe_te_s))

            x_tr = np.hstack(parts_tr)
            x_te = np.hstack(parts_te)

            dists = np.linalg.norm(x_tr - x_te, axis=1)
            bw = float(np.quantile(dists, bw_quantile))
            weights = np.exp(-0.5 * (dists / max(bw, 1e-8)) ** 2)

            w = np.diag(np.sqrt(weights))
            model = Ridge(alpha=alpha)
            model.fit(w @ x_tr, w @ y_tr)
            preds[i_global] = float(model.predict(x_te)[0])
    return preds


def predict_test_target(
    xh_train: np.ndarray,
    xe_train: np.ndarray | None,
    y_train: np.ndarray,
    train_pids: np.ndarray,
    xh_test: np.ndarray,
    xe_test: np.ndarray | None,
    test_pids: np.ndarray,
    use_energy: bool,
    include_raw_energy: bool,
    n_pls_hoop: int,
    n_pls_energy: int,
    bw_quantile: float,
    alpha: float,
) -> np.ndarray:
    preds = np.zeros(len(xh_test), dtype=float)
    players = sorted(np.unique(train_pids))

    for pid in players:
        tr_mask = train_pids == pid
        te_mask = test_pids == pid
        te_idx = np.where(te_mask)[0]
        if len(te_idx) == 0:
            continue

        xh = xh_train[tr_mask]
        y = y_train[tr_mask]
        xh_te = xh_test[te_mask]

        sc_h = StandardScaler()
        xh_s = sc_h.fit_transform(xh)
        xh_te_s = sc_h.transform(xh_te)

        parts_train = [xh_s]
        parts_test = [xh_te_s]

        nph = min(n_pls_hoop, xh_s.shape[1], xh_s.shape[0] - 1)
        if nph > 0:
            pls_h = PLSRegression(n_components=nph)
            pls_h.fit(xh_s, y)
            parts_train.append(pls_h.transform(xh_s))
            parts_test.append(pls_h.transform(xh_te_s))

        if use_energy and xe_train is not None and xe_test is not None:
            xe = xe_train[tr_mask]
            xe_te = xe_test[te_mask]
            sc_e = StandardScaler()
            xe_s = sc_e.fit_transform(xe)
            xe_te_s = sc_e.transform(xe_te)
            if include_raw_energy:
                parts_train.append(xe_s)
                parts_test.append(xe_te_s)

            npe = min(n_pls_energy, xe_s.shape[1], xe_s.shape[0] - 1)
            if npe > 0:
                pls_e = PLSRegression(n_components=npe)
                pls_e.fit(xe_s, y)
                parts_train.append(pls_e.transform(xe_s))
                parts_test.append(pls_e.transform(xe_te_s))

        x_train_full = np.hstack(parts_train)
        x_test_full = np.hstack(parts_test)

        for i in range(len(x_test_full)):
            dists = np.linalg.norm(x_train_full - x_test_full[i], axis=1)
            bw = float(np.quantile(dists, bw_quantile))
            weights = np.exp(-0.5 * (dists / max(bw, 1e-8)) ** 2)
            w = np.diag(np.sqrt(weights))
            model = Ridge(alpha=alpha)
            model.fit(w @ x_train_full, w @ y)
            preds[te_idx[i]] = float(model.predict(x_test_full[i : i + 1])[0])
    return preds


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    n_pls_energy_grid = parse_int_list(args.n_pls_energy_grid)
    blend_weights = parse_float_list(args.blend_weights)

    print("=" * 80)
    print("ENERGY WAVE TRANSFER PIPELINE")
    print("=" * 80)
    print(f"scale={args.scale}")
    print(f"sequences_per_scale={args.sequences_per_scale}")
    print(f"wave_length={args.wave_length}")
    print(f"n_pls_energy_grid={n_pls_energy_grid}")
    print(f"energy_feature_set={args.energy_feature_set}")
    print(f"include_raw_energy={args.include_raw_energy}")

    pro_template, template_stats = load_pro_template(
        scale=args.scale,
        sequences_per_scale=args.sequences_per_scale,
        wave_length=args.wave_length,
    )
    print("Loaded pro template:")
    for k, v in template_stats.items():
        print(f"  {k}: {v}")

    print("\nLoading competition data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    player_ids = train_df["participant_id"].values
    test_pids = test_df["participant_id"].values

    scalers = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}
    scaled_targets = {
        t: scalers[t].transform(train_df[t].values.reshape(-1, 1)).ravel()
        for t in TARGETS
    }

    print("\nExtracting energy-wave features...")
    x_energy_train = build_energy_feature_matrix(
        train_df,
        pro_template=pro_template,
        wave_length=args.wave_length,
    )
    x_energy_test = build_energy_feature_matrix(
        test_df,
        pro_template=pro_template,
        wave_length=args.wave_length,
    )
    x_energy_train, selected_energy_names = select_energy_feature_set(
        x_energy_train,
        feature_set=args.energy_feature_set,
    )
    x_energy_test, _ = select_energy_feature_set(
        x_energy_test,
        feature_set=args.energy_feature_set,
    )
    print(f"  train energy shape: {x_energy_train.shape}")
    print(f"  test energy shape:  {x_energy_test.shape}")
    print(f"  selected energy features: {selected_energy_names}")

    print("\nExtracting hoop-relative features per target frame...")
    x_hoop_train: dict[str, np.ndarray] = {}
    x_hoop_test: dict[str, np.ndarray] = {}
    for target in TARGETS:
        frame = TARGET_FRAMES[target]
        print(f"  {target}: frame={frame}")
        xh_tr = build_hoop_features(train_df, frame)
        xh_te = build_hoop_features(test_df, frame)
        x_hoop_train[target] = xh_tr
        x_hoop_test[target] = xh_te
        print(f"    hoop shape: {xh_tr.shape}")

    print("\n" + "=" * 80)
    print("HONEST LOO EVALUATION")
    print("=" * 80)

    configs: list[tuple[str, bool, int]] = [("baseline_hoop_only", False, 0)]
    for npe in n_pls_energy_grid:
        configs.append((f"hoop_plus_energy_{npe}pls", True, npe))

    all_results: dict[str, dict] = {}
    for cfg_name, use_energy, npe in configs:
        t0 = time.time()
        result: dict[str, float] = {}
        print(f"\nConfig: {cfg_name}")
        for target in TARGETS:
            preds = run_per_player_loo(
                hoop_features=x_hoop_train[target],
                energy_features=x_energy_train if use_energy else None,
                targets=scaled_targets[target],
                player_ids=player_ids,
                use_energy=use_energy,
                include_raw_energy=args.include_raw_energy,
                n_pls_hoop=args.n_pls_hoop,
                n_pls_energy=npe,
                bw_quantile=args.bw_quantile,
                alpha=args.alpha,
            )
            mse = float(np.mean((preds - scaled_targets[target]) ** 2))
            result[target] = mse
            print(f"  {target}: {mse:.12f}")
        result["mean"] = float(np.mean([result[t] for t in TARGETS]))
        result["time_s"] = float(time.time() - t0)
        all_results[cfg_name] = result
        print(f"  mean: {result['mean']:.12f} [{result['time_s']:.3f}s]")

    baseline_mean = all_results["baseline_hoop_only"]["mean"]
    for cfg_name, result in all_results.items():
        delta_pct = (result["mean"] - baseline_mean) / baseline_mean * 100.0
        result["delta_pct"] = float(delta_pct)

    print("\nComparison vs baseline:")
    for cfg_name, result in all_results.items():
        print(
            f"  {cfg_name}: mean={result['mean']:.12f}, "
            f"delta_pct={result['delta_pct']:+.12f}"
        )

    non_baseline = [c for c in all_results.keys() if c != "baseline_hoop_only"]
    best_cfg = min(non_baseline, key=lambda x: all_results[x]["mean"])
    best_npe = int(best_cfg.split("_")[-1].replace("pls", ""))
    print(f"\nBest energy config: {best_cfg} (n_pls_energy={best_npe})")

    submissions_created: list[dict] = []
    if not args.skip_submissions:
        print("\n" + "=" * 80)
        print("GENERATING SUBMISSIONS")
        print("=" * 80)

        base_path = SUBMISSION_DIR / f"submission_{args.base_submission}.csv"
        if not base_path.exists():
            raise FileNotFoundError(f"Base submission not found: {base_path}")
        base_sub = pd.read_csv(base_path)
        standalone = base_sub.copy()

        for target, col in zip(TARGETS, TARGET_COLS):
            preds = predict_test_target(
                xh_train=x_hoop_train[target],
                xe_train=x_energy_train,
                y_train=scaled_targets[target],
                train_pids=player_ids,
                xh_test=x_hoop_test[target],
                xe_test=x_energy_test,
                test_pids=test_pids,
                use_energy=True,
                include_raw_energy=args.include_raw_energy,
                n_pls_hoop=args.n_pls_hoop,
                n_pls_energy=best_npe,
                bw_quantile=args.bw_quantile,
                alpha=args.alpha,
            )
            standalone[col] = np.clip(preds, 0.0, 1.0)

        sn = get_next_submission_number()
        standalone_path = SUBMISSION_DIR / f"submission_{sn}.csv"
        standalone.to_csv(standalone_path, index=False)
        submissions_created.append(
            {
                "submission_number": int(sn),
                "path": str(standalone_path),
                "type": "standalone_energy_wave",
                "best_config": best_cfg,
                "base_submission": int(args.base_submission),
            }
        )
        print(f"  standalone: submission_{sn}.csv")

        for bw in blend_weights:
            blended = base_sub.copy()
            for col in TARGET_COLS:
                blended[col] = np.clip(
                    bw * standalone[col].values + (1.0 - bw) * base_sub[col].values,
                    0.0,
                    1.0,
                )
            sn = get_next_submission_number()
            blend_path = SUBMISSION_DIR / f"submission_{sn}.csv"
            blended.to_csv(blend_path, index=False)
            submissions_created.append(
                {
                    "submission_number": int(sn),
                    "path": str(blend_path),
                    "type": "blend_with_base",
                    "blend_weight_energy": float(bw),
                    "blend_weight_base": float(1.0 - bw),
                    "base_submission": int(args.base_submission),
                    "best_config": best_cfg,
                }
            )
            print(
                f"  blend: submission_{sn}.csv "
                f"(energy={bw:.6f}, base={1.0-bw:.6f})"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_json = OUTPUT_DIR / f"energy_wave_transfer_run_{ts}.json"
    run_md = OUTPUT_DIR / f"energy_wave_transfer_details_{ts}.md"

    payload = {
        "timestamp": ts,
        "command": " ".join(sys.argv),
        "config": {
            "scale": args.scale,
            "sequences_per_scale": args.sequences_per_scale,
            "wave_length": args.wave_length,
            "seed": args.seed,
            "n_pls_hoop": args.n_pls_hoop,
            "n_pls_energy_grid": n_pls_energy_grid,
            "energy_feature_set": args.energy_feature_set,
            "include_raw_energy": args.include_raw_energy,
            "bw_quantile": args.bw_quantile,
            "alpha": args.alpha,
            "base_submission": args.base_submission,
            "blend_weights": blend_weights,
            "skip_submissions": args.skip_submissions,
        },
        "template_stats": template_stats,
        "energy_feature_names": selected_energy_names,
        "energy_feature_shapes": {
            "train": list(x_energy_train.shape),
            "test": list(x_energy_test.shape),
        },
        "results": all_results,
        "best_config": best_cfg,
        "submissions_created": submissions_created,
    }

    with open(run_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Energy Wave Transfer Run Details",
        "",
        f"- timestamp: `{ts}`",
        f"- command: `{payload['command']}`",
        "",
        "## Configuration",
        f"- scale: `{args.scale}`",
        f"- sequences_per_scale: `{args.sequences_per_scale}`",
        f"- wave_length: `{args.wave_length}`",
        f"- seed: `{args.seed}`",
        f"- n_pls_hoop: `{args.n_pls_hoop}`",
        f"- n_pls_energy_grid: `{n_pls_energy_grid}`",
        f"- energy_feature_set: `{args.energy_feature_set}`",
        f"- include_raw_energy: `{args.include_raw_energy}`",
        f"- bw_quantile: `{args.bw_quantile}`",
        f"- alpha: `{args.alpha}`",
        "",
        "## Pro Template Stats",
    ]
    for k, v in template_stats.items():
        lines.append(f"- {k}: `{v}`")

    lines.extend(["", "## Selected Energy Features"])
    for name in selected_energy_names:
        lines.append(f"- {name}")

    lines.extend(["", "## LOO Results"])
    for cfg_name, result in all_results.items():
        lines.append(f"- {cfg_name}:")
        for key in ["angle", "depth", "left_right", "mean", "delta_pct", "time_s"]:
            if key in result:
                lines.append(f"  - {key}: `{result[key]}`")

    lines.extend(
        [
            "",
            f"## Best Config",
            f"- name: `{best_cfg}`",
            f"- n_pls_energy: `{best_npe}`",
            "",
            "## Generated Submissions",
        ]
    )
    if submissions_created:
        for row in submissions_created:
            lines.append(f"- submission_{row['submission_number']}.csv: `{row}`")
    else:
        lines.append("- None (`--skip-submissions` used)")

    with open(run_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\nSaved artifacts:")
    print(f"  {run_json}")
    print(f"  {run_md}")


if __name__ == "__main__":
    main()

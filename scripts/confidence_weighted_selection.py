"""
Confidence-Weighted Row-Level Model Selection

Key insight: our Ridge model is probably very good for MOST test examples but
BAD for a few hard ones. If we identify which test examples are "hard" for Ridge
and substitute a different model's prediction for just those rows, we can improve
without the dilution of global blending.

Approach:
1. Load predictions from multiple diverse models (submissions).
2. For each test example, compute a "confidence score" based on how close it is
   to training examples in feature space (per-player, per-target).
3. For atypical/hard examples, blend more aggressively with diverse models.
   For typical/easy examples, stick with Sub 2716 (current best).
4. Per-target, per-player confidence-weighted blending.
5. Also try: for each test row, use the model whose predictions have lowest
   disagreement with the consensus of diverse models (i.e., median-closest).

Generates multiple candidate submissions with different confidence thresholds.
"""

import json
import time
import fcntl
import sys
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
TARGETS = ["angle", "depth", "left_right"]
TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0
FEET_TO_METERS = 0.3048

TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

# Model submissions to load
MODEL_SUBS = {
    "ridge_base": 2503,
    "best_blend": 2716,
    "bigru_10seed": 2887,
    "knn_bootstrap": 2784,
    "rf_velocity": 2780,
    "temporal_cnn": 1557,
    "trajectory": 1507,
    "energy_wave": 2602,
    "pulse": 2608,
}


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
# DATA LOADING
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
        ids, pids, targets = [], [], []
        for idx, row in df.iterrows():
            for col_i, col in enumerate(keypoint_cols):
                arr = parse_array_string(row[col])
                X_3d[idx, :, col_i // 3, col_i % 3] = arr
            ids.append(row['id'])
            pids.append(row['participant_id'])
            if is_train:
                targets.append([row['angle'], row['depth'], row['left_right']])
        result = {'X_3d': X_3d, 'pids': np.array(pids),
                  'ids': np.array(ids), 'kp_names': kp_names, 'kp_index': kp_index}
        if is_train:
            result['y'] = np.array(targets, dtype=np.float32)
        return result

    return process(train_df, True), process(test_df, False)


def load_submission(sub_num):
    """Load a submission CSV."""
    path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    need = {"id"} | set(TARGET_COLS)
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"submission_{sub_num}.csv missing columns: {sorted(missing)}")
    return df[["id"] + TARGET_COLS].copy()


# ==============================================================
# FEATURE EXTRACTION (compact - for distance computation only)
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


def extract_compact_features(ts_3d, ts_hr, kp_index, frame):
    """Extract compact feature set at a specific frame for distance computation.
    Returns ~80 features: positions + velocities for key joints at target frame."""
    f = int(np.clip(frame, 0, 239))
    feats = []

    key_joints = ['right_wrist', 'right_elbow', 'right_shoulder',
                  'left_wrist', 'left_shoulder',
                  'right_hip', 'left_hip', 'mid_hip',
                  'right_knee', 'left_knee', 'neck', 'nose']

    # Hoop-relative positions + velocities at target frame
    for jname in key_joints:
        idx = kp_index.get(jname)
        if idx is None:
            feats.extend([0.0] * 6)
            continue
        for coord in range(3):
            feats.append(ts_hr[f, idx, coord])
            vel = np.gradient(ts_hr[:, idx, coord], DT)
            feats.append(vel[f])

    # Arm mechanics
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
    else:
        feats.extend([0.0] * 4)

    return np.array(feats, dtype=np.float32)


def extract_all_compact_features(data, target):
    """Extract compact features for all shots at target-specific frame."""
    frame = TARGET_FRAMES[target]
    n = len(data['pids'])
    kp_index = data['kp_index']
    all_feats = []
    for i in range(n):
        ts_3d = data['X_3d'][i]
        ts_hr = compute_hoop_transform(ts_3d, kp_index)
        feats = extract_compact_features(ts_3d, ts_hr, kp_index, frame)
        all_feats.append(feats)
    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# ==============================================================
# CONFIDENCE SCORING
# ==============================================================

def compute_confidence_scores(train_data, test_data, k=5):
    """
    For each test example, compute confidence = how typical it is relative to
    training data in feature space. Per-player, per-target.

    Returns dict[target] -> array of shape (n_test,) with values in [0, 1].
    Higher = more typical/confident, lower = more atypical/hard.
    """
    pids_train = train_data['pids']
    pids_test = test_data['pids']
    n_test = len(pids_test)

    confidence = {}

    for target in TARGETS:
        print(f"  Computing confidence for {target}...")
        X_train = extract_all_compact_features(train_data, target)
        X_test = extract_all_compact_features(test_data, target)

        conf = np.zeros(n_test, dtype=float)

        for pid in sorted(np.unique(pids_train)):
            tr_mask = pids_train == pid
            te_mask = pids_test == pid
            if not np.any(te_mask):
                continue

            X_tr = X_train[tr_mask]
            X_te = X_test[te_mask]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            k_eff = max(1, min(k, len(X_tr_s)))

            # Train-to-train reference distances (exclude self)
            k_ref = max(2, min(k_eff + 1, len(X_tr_s)))
            nn_ref = NearestNeighbors(n_neighbors=k_ref, metric="euclidean")
            nn_ref.fit(X_tr_s)
            d_tr, _ = nn_ref.kneighbors(X_tr_s)
            if k_ref > 1:
                d_tr_ref = d_tr[:, 1:k_eff+1].mean(axis=1)
            else:
                d_tr_ref = d_tr[:, 0]
            d_tr_ref_sorted = np.sort(d_tr_ref)

            # Test-to-train distances
            nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
            nn.fit(X_tr_s)
            d_te, idx_te = nn.kneighbors(X_te_s)
            d_te_mean = d_te.mean(axis=1)

            # Confidence = 1 - percentile of test dist in train dist distribution
            # Low percentile = test is close (like train) = HIGH confidence
            pct = np.searchsorted(d_tr_ref_sorted, d_te_mean, side="right") / max(len(d_tr_ref_sorted), 1)
            conf[te_mask] = 1.0 - pct  # Invert so high = confident

        confidence[target] = conf

    return confidence


def compute_ensemble_disagreement(all_preds, base_col):
    """
    For each test row & target, compute how much the models disagree.
    Returns dict[target_col] -> array of shape (n_test,) with std of predictions.
    """
    disagreement = {}
    for col in TARGET_COLS:
        stack = np.stack([p[col].values for p in all_preds.values()], axis=1)
        disagreement[col] = np.std(stack, axis=1)
    return disagreement


def compute_diverse_consensus(all_preds, exclude_key="best_blend"):
    """
    Median of all diverse models (excluding the best blend itself).
    This is the "alternative opinion" for each row.
    """
    consensus = {}
    for col in TARGET_COLS:
        diverse_preds = []
        for key, df in all_preds.items():
            if key != exclude_key:
                diverse_preds.append(df[col].values)
        if diverse_preds:
            stack = np.stack(diverse_preds, axis=1)
            consensus[col] = np.median(stack, axis=1)
        else:
            consensus[col] = all_preds[exclude_key][col].values.copy()
    return consensus


# ==============================================================
# SUBMISSION GENERATION
# ==============================================================

def save_submission(sub_df, description):
    """Save a submission with proper numbering."""
    sub_num = get_next_submission_number()
    path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub_df.to_csv(path, index=False)
    print(f"  Sub {sub_num}: {description}")
    return sub_num


def generate_confidence_blends(base_df, all_preds, confidence,
                                diverse_consensus, disagreement,
                                pids_test):
    """
    Generate multiple candidate submissions using different confidence thresholds.

    Strategy: For each test row:
    - If confidence > threshold -> use 100% base (Sub 2716)
    - If confidence < threshold -> blend with diverse consensus
    The blend weight for hard rows varies by config.
    """
    configs = []

    # Config 1: Conservative - only touch bottom 10% confidence rows, 10% diverse
    configs.append({
        "name": "conservative_10pct_10w",
        "conf_threshold_quantile": 0.10,
        "hard_diverse_weight": 0.10,
        "per_target": False,
    })

    # Config 2: Conservative - bottom 15% confidence, 15% diverse
    configs.append({
        "name": "conservative_15pct_15w",
        "conf_threshold_quantile": 0.15,
        "hard_diverse_weight": 0.15,
        "per_target": False,
    })

    # Config 3: Moderate - bottom 20% confidence, 15% diverse
    configs.append({
        "name": "moderate_20pct_15w",
        "conf_threshold_quantile": 0.20,
        "hard_diverse_weight": 0.15,
        "per_target": False,
    })

    # Config 4: Moderate - bottom 20% confidence, 20% diverse
    configs.append({
        "name": "moderate_20pct_20w",
        "conf_threshold_quantile": 0.20,
        "hard_diverse_weight": 0.20,
        "per_target": False,
    })

    # Config 5: Aggressive - bottom 30% confidence, 20% diverse
    configs.append({
        "name": "aggressive_30pct_20w",
        "conf_threshold_quantile": 0.30,
        "hard_diverse_weight": 0.20,
        "per_target": False,
    })

    # Config 6: Per-target independent thresholds (bottom 15% each, 10% weight)
    configs.append({
        "name": "per_target_15pct_10w",
        "conf_threshold_quantile": 0.15,
        "hard_diverse_weight": 0.10,
        "per_target": True,
    })

    # Config 7: Per-target 20% threshold, 15% weight
    configs.append({
        "name": "per_target_20pct_15w",
        "conf_threshold_quantile": 0.20,
        "hard_diverse_weight": 0.15,
        "per_target": True,
    })

    # Config 8: Continuous blending (no hard threshold) - weight proportional to uncertainty
    configs.append({
        "name": "continuous_blend_max15",
        "continuous": True,
        "max_diverse_weight": 0.15,
    })

    # Config 9: Continuous blending, more aggressive
    configs.append({
        "name": "continuous_blend_max20",
        "continuous": True,
        "max_diverse_weight": 0.20,
    })

    # Config 10: Per-player per-target (only touch hardest player-target combos)
    configs.append({
        "name": "per_player_target_10pct_15w",
        "conf_threshold_quantile": 0.10,
        "hard_diverse_weight": 0.15,
        "per_player_target": True,
    })

    # Config 11: Disagreement-based (use disagreement instead of OOD confidence)
    configs.append({
        "name": "disagreement_top15pct_15w",
        "disagreement_quantile": 0.85,
        "hard_diverse_weight": 0.15,
        "use_disagreement": True,
    })

    # Config 12: Combined confidence + disagreement
    configs.append({
        "name": "combined_conf_disagree_15w",
        "conf_threshold_quantile": 0.20,
        "disagreement_quantile": 0.80,
        "hard_diverse_weight": 0.15,
        "combined": True,
    })

    submissions = []

    for cfg in configs:
        sub_df = base_df[["id"]].copy()
        n_modified_total = 0
        details = {}

        for t_idx, (target, col) in enumerate(zip(TARGETS, TARGET_COLS)):
            base_vals = base_df[col].values.copy()
            diverse_vals = diverse_consensus[col]
            conf = confidence[target]
            disagree = disagreement[col]

            result = base_vals.copy()
            n_modified = 0

            if cfg.get("continuous"):
                # Continuous: blend weight = (1 - confidence) * max_weight
                # Confidence is [0,1]; low confidence = more diverse blending
                max_w = cfg["max_diverse_weight"]
                # Normalize confidence to [0, 1] range per target
                conf_min, conf_max = conf.min(), conf.max()
                if conf_max - conf_min > 1e-8:
                    conf_norm = (conf - conf_min) / (conf_max - conf_min)
                else:
                    conf_norm = np.ones_like(conf)
                # Higher confidence -> lower diverse weight
                diverse_w = (1.0 - conf_norm) * max_w
                result = (1.0 - diverse_w) * base_vals + diverse_w * diverse_vals
                n_modified = np.sum(diverse_w > 0.001)

            elif cfg.get("use_disagreement"):
                # Use ensemble disagreement as the signal
                thresh = np.quantile(disagree, cfg["disagreement_quantile"])
                hard_mask = disagree >= thresh
                w = cfg["hard_diverse_weight"]
                result[hard_mask] = (1.0 - w) * base_vals[hard_mask] + w * diverse_vals[hard_mask]
                n_modified = np.sum(hard_mask)

            elif cfg.get("combined"):
                # Both low confidence AND high disagreement
                conf_thresh = np.quantile(conf, cfg["conf_threshold_quantile"])
                disagree_thresh = np.quantile(disagree, cfg["disagreement_quantile"])
                # Union: either low confidence OR high disagreement
                hard_mask = (conf <= conf_thresh) | (disagree >= disagree_thresh)
                w = cfg["hard_diverse_weight"]
                result[hard_mask] = (1.0 - w) * base_vals[hard_mask] + w * diverse_vals[hard_mask]
                n_modified = np.sum(hard_mask)

            elif cfg.get("per_player_target"):
                # Per-player: find hardest rows within each player
                for pid in sorted(np.unique(pids_test)):
                    pid_mask = pids_test == pid
                    pid_conf = conf[pid_mask]
                    if len(pid_conf) == 0:
                        continue
                    thresh = np.quantile(pid_conf, cfg["conf_threshold_quantile"])
                    hard_in_player = pid_conf <= thresh
                    # Map back to full indices
                    pid_indices = np.where(pid_mask)[0]
                    hard_indices = pid_indices[hard_in_player]
                    w = cfg["hard_diverse_weight"]
                    result[hard_indices] = (1.0 - w) * base_vals[hard_indices] + w * diverse_vals[hard_indices]
                    n_modified += len(hard_indices)

            else:
                # Standard threshold-based
                if cfg.get("per_target"):
                    thresh = np.quantile(conf, cfg["conf_threshold_quantile"])
                else:
                    # Use mean confidence across targets for global threshold
                    mean_conf = np.mean([confidence[t] for t in TARGETS], axis=0)
                    thresh = np.quantile(mean_conf, cfg["conf_threshold_quantile"])
                    conf = mean_conf

                hard_mask = conf <= thresh
                w = cfg["hard_diverse_weight"]
                result[hard_mask] = (1.0 - w) * base_vals[hard_mask] + w * diverse_vals[hard_mask]
                n_modified = np.sum(hard_mask)

            # Clip to [0, 1]
            result = np.clip(result, 0.0, 1.0)
            sub_df[col] = result
            n_modified_total += n_modified
            details[target] = {
                "n_modified": int(n_modified),
                "mean_delta": float(np.mean(np.abs(result - base_vals))),
                "max_delta": float(np.max(np.abs(result - base_vals))),
            }

        # Compute diversity vs base
        for col in TARGET_COLS:
            diff = sub_df[col].values - base_df[col].values
            details[f"{col}_rmsd_vs_base"] = float(np.sqrt(np.mean(diff**2)))

        sub_num = save_submission(sub_df, cfg["name"])
        submissions.append({
            "sub_num": sub_num,
            "config": cfg["name"],
            "n_modified_total": n_modified_total,
            "details": details,
        })

    return submissions


def generate_median_closest_selection(base_df, all_preds, pids_test):
    """
    For each test row, pick the submission whose prediction is closest to the
    median of ALL diverse models. The idea: if one model is an outlier for a
    specific row, replace it with the consensus.
    """
    sub_df = base_df[["id"]].copy()

    sub_keys = list(all_preds.keys())
    n_subs = len(sub_keys)

    for col in TARGET_COLS:
        stack = np.stack([all_preds[k][col].values for k in sub_keys], axis=1)
        median = np.median(stack, axis=1)

        # For each row, find which submission is closest to median
        diffs = np.abs(stack - median[:, None])  # (n_test, n_subs)
        best_idx = np.argmin(diffs, axis=1)

        result = np.array([stack[i, best_idx[i]] for i in range(len(best_idx))])
        sub_df[col] = np.clip(result, 0.0, 1.0)

    sub_num = save_submission(sub_df, "median_closest_per_row")
    diff_vs_base = {}
    for col in TARGET_COLS:
        d = sub_df[col].values - base_df[col].values
        diff_vs_base[col] = float(np.sqrt(np.mean(d**2)))
    return sub_num, diff_vs_base


def generate_weighted_median(base_df, all_preds, confidence, pids_test):
    """
    Weighted median: for confident rows, weight Sub 2716 heavily.
    For hard rows, give more weight to diverse models.
    """
    sub_df = base_df[["id"]].copy()

    sub_keys = list(all_preds.keys())

    for t_idx, (target, col) in enumerate(zip(TARGETS, TARGET_COLS)):
        conf = confidence[target]
        # Normalize confidence to [0, 1]
        conf_min, conf_max = conf.min(), conf.max()
        if conf_max - conf_min > 1e-8:
            conf_norm = (conf - conf_min) / (conf_max - conf_min)
        else:
            conf_norm = np.ones_like(conf)

        stack = np.stack([all_preds[k][col].values for k in sub_keys], axis=1)

        result = np.zeros(len(conf))
        for i in range(len(conf)):
            # Weights: base model gets weight proportional to confidence
            weights = np.ones(len(sub_keys))
            for j, key in enumerate(sub_keys):
                if key == "best_blend":
                    # Base gets 1 + 4*confidence (range: 1 to 5)
                    weights[j] = 1.0 + 4.0 * conf_norm[i]
                elif key == "ridge_base":
                    weights[j] = 1.0 + 2.0 * conf_norm[i]
                else:
                    # Diverse models get weight inversely proportional to confidence
                    weights[j] = 1.0 + 2.0 * (1.0 - conf_norm[i])
            weights /= weights.sum()
            result[i] = np.dot(weights, stack[i])

        sub_df[col] = np.clip(result, 0.0, 1.0)

    sub_num = save_submission(sub_df, "confidence_weighted_avg")
    diff_vs_base = {}
    for col in TARGET_COLS:
        d = sub_df[col].values - base_df[col].values
        diff_vs_base[col] = float(np.sqrt(np.mean(d**2)))
    return sub_num, diff_vs_base


def generate_per_player_hardest_swap(base_df, all_preds, confidence, pids_test):
    """
    For each player-target combo, find the single hardest row and try
    swapping it with the diverse median. Very conservative - only touches
    ~5 rows per target (1 per player).
    """
    submissions = []

    for swap_w in [0.20, 0.35, 0.50]:
        sub_df = base_df[["id"]].copy()
        n_swapped = 0

        for t_idx, (target, col) in enumerate(zip(TARGETS, TARGET_COLS)):
            conf = confidence[target]
            base_vals = base_df[col].values.copy()

            # Compute diverse consensus (exclude best_blend and ridge_base)
            diverse_preds = []
            for key, df in all_preds.items():
                if key not in ("best_blend", "ridge_base"):
                    diverse_preds.append(df[col].values)
            diverse_median = np.median(np.stack(diverse_preds, axis=1), axis=1)

            result = base_vals.copy()

            for pid in sorted(np.unique(pids_test)):
                pid_mask = pids_test == pid
                pid_indices = np.where(pid_mask)[0]
                pid_conf = conf[pid_indices]

                if len(pid_conf) == 0:
                    continue

                # Find the single hardest row for this player
                hardest_local_idx = np.argmin(pid_conf)
                hardest_global_idx = pid_indices[hardest_local_idx]

                # Swap this row
                result[hardest_global_idx] = (
                    (1.0 - swap_w) * base_vals[hardest_global_idx]
                    + swap_w * diverse_median[hardest_global_idx]
                )
                n_swapped += 1

            sub_df[col] = np.clip(result, 0.0, 1.0)

        sub_num = save_submission(sub_df, f"hardest_swap_w{swap_w:.2f}")
        submissions.append({
            "sub_num": sub_num,
            "config": f"hardest_swap_w{swap_w:.2f}",
            "n_swapped": n_swapped,
        })

    return submissions


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()

    print("=" * 80)
    print("CONFIDENCE-WEIGHTED ROW-LEVEL MODEL SELECTION")
    print("=" * 80)

    # Step 1: Load all model predictions
    print("\n[1/5] Loading model predictions...")
    all_preds = {}
    for key, sub_num in MODEL_SUBS.items():
        try:
            df = load_submission(sub_num)
            all_preds[key] = df
            print(f"  Loaded Sub {sub_num} ({key}): {len(df)} rows")
        except Exception as e:
            print(f"  SKIP Sub {sub_num} ({key}): {e}")

    base_df = all_preds["best_blend"]
    n_test = len(base_df)
    print(f"\n  Base: Sub 2716 (best_blend), {n_test} test rows")
    print(f"  Loaded {len(all_preds)} model predictions total")

    # Verify all have same IDs
    ref_ids = base_df["id"].values
    for key, df in all_preds.items():
        if not np.array_equal(df["id"].values, ref_ids):
            print(f"  WARNING: {key} has different IDs! Reordering...")
            df = df.set_index("id").reindex(ref_ids).reset_index()
            all_preds[key] = df

    # Step 2: Compute pairwise diversity between models
    print("\n[2/5] Computing model diversity...")
    keys = list(all_preds.keys())
    print(f"\n  Pairwise Pearson r vs Sub 2716 (best_blend):")
    for col in TARGET_COLS:
        base_vals = base_df[col].values
        print(f"\n  {col}:")
        for key in keys:
            if key == "best_blend":
                continue
            other_vals = all_preds[key][col].values
            r = np.corrcoef(base_vals, other_vals)[0, 1]
            rmsd = np.sqrt(np.mean((base_vals - other_vals)**2))
            print(f"    {key:20s}: r={r:.4f}, RMSD={rmsd:.6f}")

    # Step 3: Load raw data and compute confidence scores
    print("\n[3/5] Loading raw data and computing confidence scores...")
    train_data, test_data = load_data()
    pids_test = test_data['pids']

    confidence = compute_confidence_scores(train_data, test_data, k=5)

    print("\n  Confidence score statistics:")
    for target in TARGETS:
        conf = confidence[target]
        print(f"    {target}: mean={conf.mean():.3f}, std={conf.std():.3f}, "
              f"min={conf.min():.3f}, max={conf.max():.3f}")
        # Show per-player stats
        for pid in sorted(np.unique(pids_test)):
            mask = pids_test == pid
            pc = conf[mask]
            print(f"      P{pid}: mean={pc.mean():.3f}, min={pc.min():.3f}, "
                  f"n_below_0.3={np.sum(pc < 0.3)}/{len(pc)}")

    # Step 4: Compute ensemble disagreement
    print("\n[4/5] Computing ensemble disagreement...")
    disagreement = compute_ensemble_disagreement(all_preds, "best_blend")

    print("\n  Disagreement statistics (std across models):")
    for col in TARGET_COLS:
        d = disagreement[col]
        print(f"    {col}: mean={d.mean():.6f}, max={d.max():.6f}, "
              f"p90={np.quantile(d, 0.90):.6f}")

    # Correlation between confidence and disagreement
    print("\n  Correlation between OOD confidence and disagreement:")
    for target, col in zip(TARGETS, TARGET_COLS):
        r = np.corrcoef(confidence[target], disagreement[col])[0, 1]
        print(f"    {target}: r={r:.4f}")

    # Step 5: Generate submissions
    print("\n[5/5] Generating candidate submissions...")

    # 5a: Diverse consensus (excluding base)
    diverse_consensus = compute_diverse_consensus(all_preds, exclude_key="best_blend")

    # 5b: Confidence-threshold blends
    print("\n  --- Confidence-threshold blends ---")
    blend_subs = generate_confidence_blends(
        base_df, all_preds, confidence, diverse_consensus, disagreement, pids_test
    )

    # 5c: Median-closest selection
    print("\n  --- Median-closest per-row selection ---")
    mc_num, mc_diff = generate_median_closest_selection(base_df, all_preds, pids_test)

    # 5d: Confidence-weighted average
    print("\n  --- Confidence-weighted average ---")
    cwa_num, cwa_diff = generate_weighted_median(base_df, all_preds, confidence, pids_test)

    # 5e: Per-player hardest swap
    print("\n  --- Per-player hardest row swap ---")
    swap_subs = generate_per_player_hardest_swap(base_df, all_preds, confidence, pids_test)

    # ==============================================================
    # REPORT
    # ==============================================================
    elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"\nBase: Sub 2716 (LB 0.006343)")

    print("\n--- Confidence-Threshold Blend Submissions ---")
    print(f"{'Sub':>6s}  {'Config':40s}  {'Modified':>8s}  {'Mean Delta':>12s}  {'RMSD vs Base':>14s}")
    for s in blend_subs:
        mean_delta = np.mean([s["details"][t]["mean_delta"] for t in TARGETS])
        rmsd_vals = [s["details"].get(f"{col}_rmsd_vs_base", 0) for col in TARGET_COLS]
        mean_rmsd = np.mean(rmsd_vals)
        print(f"  {s['sub_num']:4d}  {s['config']:40s}  {s['n_modified_total']:8d}  "
              f"{mean_delta:12.6f}  {mean_rmsd:14.6f}")

    print(f"\n--- Special Submissions ---")
    print(f"  Sub {mc_num}: median_closest_per_row")
    for col in TARGET_COLS:
        print(f"    {col} RMSD vs base: {mc_diff[col]:.6f}")

    print(f"\n  Sub {cwa_num}: confidence_weighted_avg")
    for col in TARGET_COLS:
        print(f"    {col} RMSD vs base: {cwa_diff[col]:.6f}")

    print(f"\n--- Per-Player Hardest Swap ---")
    for s in swap_subs:
        print(f"  Sub {s['sub_num']}: {s['config']} ({s['n_swapped']} rows swapped)")

    # Show which test rows are hardest
    print("\n--- Hardest Test Examples (lowest confidence) ---")
    for target in TARGETS:
        conf = confidence[target]
        order = np.argsort(conf)
        print(f"\n  {target} (5 hardest):")
        for rank, idx in enumerate(order[:5]):
            pid = pids_test[idx]
            test_id = test_data['ids'][idx]
            # Show how much models disagree on this row
            col = TARGET_COLS[TARGETS.index(target)]
            sub_vals = [all_preds[k][col].values[idx] for k in all_preds]
            print(f"    #{rank+1}: idx={idx}, P{pid}, conf={conf[idx]:.3f}, "
                  f"pred_range=[{min(sub_vals):.4f}, {max(sub_vals):.4f}], "
                  f"spread={max(sub_vals)-min(sub_vals):.4f}")

    # Show the ALL submission numbers for easy reference
    print("\n--- All Generated Submissions ---")
    all_sub_nums = [s["sub_num"] for s in blend_subs]
    all_sub_nums.append(mc_num)
    all_sub_nums.append(cwa_num)
    all_sub_nums.extend([s["sub_num"] for s in swap_subs])

    for s in blend_subs:
        print(f"  Sub {s['sub_num']}: {s['config']}")
    print(f"  Sub {mc_num}: median_closest_per_row")
    print(f"  Sub {cwa_num}: confidence_weighted_avg")
    for s in swap_subs:
        print(f"  Sub {s['sub_num']}: {s['config']}")

    print(f"\nTotal: {len(all_sub_nums)} submissions generated")
    print(f"Range: Sub {min(all_sub_nums)} - Sub {max(all_sub_nums)}")


if __name__ == "__main__":
    main()

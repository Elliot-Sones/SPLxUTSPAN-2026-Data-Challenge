"""
SHOT7M2 Integration into Best Pipeline

Takes the SHOT7M2-derived features and adds them as supplementary PLS-compressed
features to our combined best pipeline (same as Sub 2503).

This is the proper test: does SHOT7M2 signal add value ON TOP of our existing
best features (hoop-relative + KC-PLS + right-hand physics + multi-frame)?

Uses honest LOO (PLS refit) for evaluation.
"""

import json
import fcntl
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submission"
SHOT7M2_DIR = PROJECT_DIR / "external_data" / "shot7m2_sample"
TARGETS = ["angle", "depth", "left_right"]
TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
HOOP = np.array([5.25, -25.0, 10.0])

# Joint mapping
JOINT_MAP = {
    "mid_hip": 0, "left_hip": 1, "left_knee": 2, "left_ankle": 3,
    "right_hip": 6, "right_knee": 7, "right_ankle": 8,
    "left_shoulder": 16, "left_elbow": 17, "left_wrist": 18,
    "right_shoulder": 23, "right_elbow": 24, "right_wrist": 25,
    "neck": 19,
}
MAPPED_JOINTS = list(JOINT_MAP.keys())
SHOT7M2_INDICES = [JOINT_MAP[j] for j in MAPPED_JOINTS]

JOINT_ANGLE_DEFS = [
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("neck", "left_shoulder", "left_elbow"),
    ("neck", "right_shoulder", "right_elbow"),
    ("left_shoulder", "mid_hip", "right_shoulder"),
    ("left_hip", "mid_hip", "right_hip"),
    ("mid_hip", "neck", "right_shoulder"),
    ("mid_hip", "neck", "left_shoulder"),
]


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


def parse_json_safe(s, frame):
    s = str(s).replace('nan', 'null').replace('NaN', 'null')
    arr = json.loads(s)
    val = arr[frame]
    if val is None:
        for offset in range(1, 10):
            if frame - offset >= 0 and arr[frame - offset] is not None:
                return float(arr[frame - offset])
            if frame + offset < len(arr) and arr[frame + offset] is not None:
                return float(arr[frame + offset])
        return 0.0
    return float(val)


def normalize_pose(positions):
    pelvis_idx = MAPPED_JOINTS.index("mid_hip")
    neck_idx = MAPPED_JOINTS.index("neck")
    centered = positions - positions[:, pelvis_idx:pelvis_idx+1, :]
    torso_len = np.linalg.norm(centered[:, neck_idx] - centered[:, pelvis_idx],
                                axis=1, keepdims=True)
    torso_len = np.maximum(torso_len, 1e-6)
    return centered / torso_len[:, :, np.newaxis]


def compute_joint_angles(positions):
    angles = np.zeros((len(positions), len(JOINT_ANGLE_DEFS)))
    for i, (parent, joint, child) in enumerate(JOINT_ANGLE_DEFS):
        p_idx = MAPPED_JOINTS.index(parent)
        j_idx = MAPPED_JOINTS.index(joint)
        c_idx = MAPPED_JOINTS.index(child)
        v1 = positions[:, p_idx] - positions[:, j_idx]
        v2 = positions[:, c_idx] - positions[:, j_idx]
        v1 = v1 / np.maximum(np.linalg.norm(v1, axis=1, keepdims=True), 1e-8)
        v2 = v2 / np.maximum(np.linalg.norm(v2, axis=1, keepdims=True), 1e-8)
        cos_a = np.clip(np.sum(v1 * v2, axis=1), -1, 1)
        angles[:, i] = np.arccos(cos_a)
    return angles


def load_shot7m2():
    """Load SHOT7M2 and compute shooting statistics."""
    print("Loading SHOT7M2...")
    poses = np.load(SHOT7M2_DIR / "train" / "train_dictionary_poses.npy",
                    allow_pickle=True).item()
    actions = np.load(SHOT7M2_DIR / "train" / "train_dictionary_actions.npy",
                      allow_pickle=True).item()

    vocab = actions['vocabulary']
    labels = actions['label_array']
    fnm = actions['frame_number_map']
    shoot_idx = vocab.index('action_Shoot')

    all_poses = []
    seq_keys = list(poses['sequences']['keypoints'].keys())
    for key in seq_keys:
        start, end = fnm[key]
        ep_shoot = labels[shoot_idx, start:end]
        shoot_frames = np.where(ep_shoot > 0.3)[0]
        if len(shoot_frames) > 0:
            ep_poses = poses['sequences']['keypoints'][key]
            for f in shoot_frames:
                all_poses.append(ep_poses[f, 0, SHOT7M2_INDICES, :])

    shoot_poses = np.array(all_poses)
    shoot_norm = normalize_pose(shoot_poses)
    shoot_angles = compute_joint_angles(shoot_norm)

    # Fit PCA on shooting poses
    shoot_flat = shoot_norm.reshape(len(shoot_norm), -1)
    pca = PCA(n_components=min(20, shoot_flat.shape[1]))
    pca.fit(shoot_flat)

    # Compute statistics
    angle_means = np.mean(shoot_angles, axis=0)
    angle_stds = np.std(shoot_angles, axis=0)
    centroid = np.mean(shoot_flat, axis=0)

    print(f"  Shooting frames: {len(shoot_poses)}")
    print(f"  PCA explained (10 components): {np.sum(pca.explained_variance_ratio_[:10]):.4f}")

    return {
        'pca': pca,
        'shoot_norm': shoot_norm,
        'shoot_angles': shoot_angles,
        'angle_means': angle_means,
        'angle_stds': angle_stds,
        'centroid': centroid,
    }


def extract_shot7m2_features(positions_14, shot7m2_data):
    """
    Extract SHOT7M2-derived features from 14-joint positions.
    positions_14: (N, 14, 3) raw joint positions
    Returns: (N, n_features) array
    """
    N = len(positions_14)
    pca = shot7m2_data['pca']
    angle_means = shot7m2_data['angle_means']
    angle_stds = shot7m2_data['angle_stds']
    centroid = shot7m2_data['centroid']

    # Normalize
    norm = normalize_pose(positions_14)
    flat = norm.reshape(N, -1)

    features = []

    # 1. PCA projections (top 10 components)
    pca_feats = pca.transform(flat)[:, :10]
    features.append(pca_feats)

    # 2. Reconstruction error
    reconstructed = pca.inverse_transform(pca.transform(flat))
    recon_error = np.mean((flat - reconstructed) ** 2, axis=1, keepdims=True)
    features.append(recon_error)

    # 3. Joint angle z-scores
    angles = compute_joint_angles(norm)
    z_scores = np.zeros_like(angles)
    for i in range(angles.shape[1]):
        if angle_stds[i] > 1e-8:
            z_scores[:, i] = (angles[:, i] - angle_means[i]) / angle_stds[i]
    features.append(z_scores)

    # 4. Raw joint angles
    features.append(angles)

    # 5. Distance to shooting centroid
    dist = np.linalg.norm(flat - centroid, axis=1, keepdims=True)
    features.append(dist)

    # 6. Arm/body configuration features
    # Wrist heights relative to neck
    neck_idx = MAPPED_JOINTS.index("neck")
    rw_idx = MAPPED_JOINTS.index("right_wrist")
    lw_idx = MAPPED_JOINTS.index("left_wrist")
    r_wrist_h = (norm[:, rw_idx, 1] - norm[:, neck_idx, 1]).reshape(-1, 1)
    l_wrist_h = (norm[:, lw_idx, 1] - norm[:, neck_idx, 1]).reshape(-1, 1)
    features.append(r_wrist_h)
    features.append(l_wrist_h)

    # Arm extensions
    rs_idx = MAPPED_JOINTS.index("right_shoulder")
    ls_idx = MAPPED_JOINTS.index("left_shoulder")
    r_ext = np.linalg.norm(norm[:, rw_idx] - norm[:, rs_idx], axis=1, keepdims=True)
    l_ext = np.linalg.norm(norm[:, lw_idx] - norm[:, ls_idx], axis=1, keepdims=True)
    features.append(r_ext)
    features.append(l_ext)

    # Arm asymmetry
    re_idx = MAPPED_JOINTS.index("right_elbow")
    le_idx = MAPPED_JOINTS.index("left_elbow")
    r_upper = np.linalg.norm(norm[:, rs_idx] - norm[:, re_idx], axis=1, keepdims=True)
    l_upper = np.linalg.norm(norm[:, ls_idx] - norm[:, le_idx], axis=1, keepdims=True)
    features.append(r_upper - l_upper)

    return np.hstack(features)


def extract_14_joints(df, frame):
    """Extract the 14 mapped joints from our data."""
    n = len(df)
    positions = np.zeros((n, 14, 3))
    for j_idx, name in enumerate(MAPPED_JOINTS):
        for c_idx, coord in enumerate(['x', 'y', 'z']):
            col = f"{name}_{coord}"
            positions[:, j_idx, c_idx] = df[col].apply(
                lambda x: parse_json_safe(x, frame)).values
    return positions


def build_all_features(df, frame, shot7m2_data):
    """
    Build the full feature set: hoop-relative (all 69 kps) + SHOT7M2 features.
    This mirrors what our best pipeline uses, PLUS the SHOT7M2 additions.
    """
    n = len(df)
    kp_names = sorted(set(c.rsplit('_', 1)[0]
                          for c in df.columns if c.endswith(('_x', '_y', '_z'))))

    # Hoop-relative features (all 69 keypoints)
    positions = np.zeros((n, len(kp_names), 3))
    for k_idx, kp in enumerate(kp_names):
        for c_idx, coord in enumerate(['x', 'y', 'z']):
            col = f"{kp}_{coord}"
            positions[:, k_idx, c_idx] = df[col].apply(
                lambda x: parse_json_safe(x, frame)).values
    hoop_rel = positions - HOOP[np.newaxis, np.newaxis, :]
    hoop_features = hoop_rel.reshape(n, -1)

    # SHOT7M2 features (14 mapped joints)
    positions_14 = extract_14_joints(df, frame)
    shot7m2_features = extract_shot7m2_features(positions_14, shot7m2_data)

    return hoop_features, shot7m2_features


def run_per_player_loo(hoop_features, shot7m2_features, targets, player_ids,
                        use_shot7m2=True, n_pls_hoop=15, n_pls_shot7m2=3,
                        bw_quantile=0.45, alpha=10.0):
    """
    Per-player LOO with PLS-compressed SHOT7M2 features added to hoop features.
    Uses honest PLS (refit per fold).
    """
    n = len(targets)
    predictions = np.zeros(n)
    players = sorted(np.unique(player_ids))

    for pid in players:
        mask = player_ids == pid
        idx = np.where(mask)[0]
        n_player = len(idx)

        X_hoop = hoop_features[mask]
        y = targets[mask]

        if use_shot7m2:
            X_s7 = shot7m2_features[mask]

        for i_local in range(n_player):
            i_global = idx[i_local]

            # Leave one out
            loo_mask = np.ones(n_player, dtype=bool)
            loo_mask[i_local] = False

            X_hoop_tr = X_hoop[loo_mask]
            X_hoop_te = X_hoop[~loo_mask]
            y_tr = y[loo_mask]

            # Scale hoop features
            scaler_h = StandardScaler()
            X_hoop_tr_s = scaler_h.fit_transform(X_hoop_tr)
            X_hoop_te_s = scaler_h.transform(X_hoop_te)

            # PLS on hoop features (honest - excludes held-out)
            n_pls_h = min(n_pls_hoop, X_hoop_tr_s.shape[1], X_hoop_tr_s.shape[0] - 1)
            pls_h = PLSRegression(n_components=n_pls_h)
            pls_h.fit(X_hoop_tr_s, y_tr)
            X_hoop_tr_pls = pls_h.transform(X_hoop_tr_s)
            X_hoop_te_pls = pls_h.transform(X_hoop_te_s)

            parts_tr = [X_hoop_tr_s, X_hoop_tr_pls]
            parts_te = [X_hoop_te_s, X_hoop_te_pls]

            if use_shot7m2:
                X_s7_tr = X_s7[loo_mask]
                X_s7_te = X_s7[~loo_mask]

                # Scale SHOT7M2 features
                scaler_s = StandardScaler()
                X_s7_tr_s = scaler_s.fit_transform(X_s7_tr)
                X_s7_te_s = scaler_s.transform(X_s7_te)

                # PLS on SHOT7M2 features (honest)
                n_pls_s = min(n_pls_shot7m2, X_s7_tr_s.shape[1], X_s7_tr_s.shape[0] - 1)
                if n_pls_s > 0:
                    pls_s = PLSRegression(n_components=n_pls_s)
                    pls_s.fit(X_s7_tr_s, y_tr)
                    X_s7_tr_pls = pls_s.transform(X_s7_tr_s)
                    X_s7_te_pls = pls_s.transform(X_s7_te_s)
                    parts_tr.append(X_s7_tr_pls)
                    parts_te.append(X_s7_te_pls)

            X_tr_full = np.hstack(parts_tr)
            X_te_full = np.hstack(parts_te)

            # Locally weighted Ridge
            dists = np.linalg.norm(X_tr_full - X_te_full, axis=1)
            bw = np.quantile(dists, bw_quantile)
            weights = np.exp(-0.5 * (dists / max(bw, 1e-8)) ** 2)

            W = np.diag(np.sqrt(weights))
            ridge = Ridge(alpha=alpha)
            ridge.fit(W @ X_tr_full, W @ y_tr)
            predictions[i_global] = ridge.predict(X_te_full)[0]

    return predictions


def main():
    print("=" * 70)
    print("SHOT7M2 INTEGRATION INTO BEST PIPELINE")
    print("=" * 70)

    # Load SHOT7M2 data
    shot7m2_data = load_shot7m2()

    # Load our data
    print("\nLoading competition data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    player_ids = train_df['participant_id'].values
    test_pids = test_df['participant_id'].values

    scalers = {}
    for t in TARGETS:
        scalers[t] = joblib.load(DATA_DIR / f"scaler_{t}.pkl")

    scaled_targets = {}
    for t in TARGETS:
        scaled_targets[t] = scalers[t].transform(
            train_df[t].values.reshape(-1, 1)).ravel()

    # Extract features for each target frame
    print("\nExtracting features...")
    train_hoop = {}
    train_s7 = {}
    test_hoop = {}
    test_s7 = {}

    for target in TARGETS:
        frame = TARGET_FRAMES[target]
        print(f"  {target} (frame {frame})...")
        h_tr, s_tr = build_all_features(train_df, frame, shot7m2_data)
        h_te, s_te = build_all_features(test_df, frame, shot7m2_data)
        train_hoop[target] = h_tr
        train_s7[target] = s_tr
        test_hoop[target] = h_te
        test_s7[target] = s_te
        print(f"    Hoop: {h_tr.shape}, SHOT7M2: {s_tr.shape}")

    # =====================================================================
    # Evaluate: baseline vs with SHOT7M2
    # =====================================================================
    print("\n" + "=" * 70)
    print("HONEST LOO EVALUATION")
    print("=" * 70)

    configs = [
        ("baseline (hoop only)", False, 15, 0),
        ("hoop + SHOT7M2 (3 PLS)", True, 15, 3),
        ("hoop + SHOT7M2 (5 PLS)", True, 15, 5),
        ("hoop + SHOT7M2 (1 PLS)", True, 15, 1),
    ]

    all_results = {}
    for config_name, use_s7, n_pls_h, n_pls_s in configs:
        print(f"\n  Config: {config_name}")
        t0 = time.time()
        target_mse = {}
        for target in TARGETS:
            preds = run_per_player_loo(
                train_hoop[target], train_s7[target] if use_s7 else None,
                scaled_targets[target], player_ids,
                use_shot7m2=use_s7, n_pls_hoop=n_pls_h, n_pls_shot7m2=n_pls_s
            )
            mse = np.mean((preds - scaled_targets[target]) ** 2)
            target_mse[target] = mse
            print(f"    {target}: LOO={mse:.6f}")
        mean_mse = np.mean(list(target_mse.values()))
        target_mse["mean"] = mean_mse
        dt = time.time() - t0
        print(f"    mean: {mean_mse:.6f} [{dt:.1f}s]")
        all_results[config_name] = target_mse

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    baseline = all_results["baseline (hoop only)"]["mean"]
    for name, res in all_results.items():
        pct = (res["mean"] - baseline) / baseline * 100
        print(f"  {name:40s}: {res['mean']:.6f} ({pct:+.2f}%)")
        for t in TARGETS:
            b = all_results["baseline (hoop only)"][t]
            pct_t = (res[t] - b) / b * 100
            print(f"    {t:12s}: {res[t]:.6f} ({pct_t:+.2f}%)")

    # =====================================================================
    # Generate submissions
    # =====================================================================
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    # Find best config
    best_config = min(
        [k for k in all_results if k != "baseline (hoop only)"],
        key=lambda k: all_results[k]["mean"]
    )
    best_result = all_results[best_config]
    print(f"  Best config: {best_config} (mean LOO: {best_result['mean']:.6f})")

    # Extract the PLS count from best config name
    if "5 PLS" in best_config:
        best_n_pls_s = 5
    elif "3 PLS" in best_config:
        best_n_pls_s = 3
    elif "1 PLS" in best_config:
        best_n_pls_s = 1
    else:
        best_n_pls_s = 3

    # Generate standalone submission
    sub = pd.read_csv(SUBMISSION_DIR / "submission_2503.csv")

    for t_idx, target in enumerate(TARGETS):
        X_hoop_tr = train_hoop[target]
        X_s7_tr = train_s7[target]
        X_hoop_te = test_hoop[target]
        X_s7_te = test_s7[target]
        y_tr = scaled_targets[target]

        preds = np.zeros(len(X_hoop_te))
        players = sorted(np.unique(player_ids))

        for pid in players:
            tr_mask = player_ids == pid
            te_mask = test_pids == pid
            tr_idx = np.where(tr_mask)[0]
            te_idx = np.where(te_mask)[0]

            X_h = X_hoop_tr[tr_mask]
            X_s = X_s7_tr[tr_mask]
            y = y_tr[tr_mask]
            X_h_te = X_hoop_te[te_mask]
            X_s_te = X_s7_te[te_mask]

            # Scale
            sc_h = StandardScaler()
            X_h_s = sc_h.fit_transform(X_h)
            X_h_te_s = sc_h.transform(X_h_te)

            sc_s = StandardScaler()
            X_s_s = sc_s.fit_transform(X_s)
            X_s_te_s = sc_s.transform(X_s_te)

            # PLS
            n_ph = min(15, X_h_s.shape[1], X_h_s.shape[0] - 1)
            pls_h = PLSRegression(n_components=n_ph)
            pls_h.fit(X_h_s, y)
            X_h_pls = pls_h.transform(X_h_s)
            X_h_te_pls = pls_h.transform(X_h_te_s)

            n_ps = min(best_n_pls_s, X_s_s.shape[1], X_s_s.shape[0] - 1)
            pls_s = PLSRegression(n_components=n_ps)
            pls_s.fit(X_s_s, y)
            X_s_pls = pls_s.transform(X_s_s)
            X_s_te_pls = pls_s.transform(X_s_te_s)

            X_full = np.hstack([X_h_s, X_h_pls, X_s_pls])
            X_full_te = np.hstack([X_h_te_s, X_h_te_pls, X_s_te_pls])

            for i in range(len(X_full_te)):
                dists = np.linalg.norm(X_full - X_full_te[i], axis=1)
                bw = np.quantile(dists, 0.45)
                weights = np.exp(-0.5 * (dists / max(bw, 1e-8)) ** 2)
                W = np.diag(np.sqrt(weights))
                ridge = Ridge(alpha=10.0)
                ridge.fit(W @ X_full, W @ y)
                preds[te_idx[i]] = ridge.predict(X_full_te[i:i+1])[0]

        sub[TARGET_COLS[t_idx]] = np.clip(preds, 0, 1)

    sub_num = get_next_submission_number()
    sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"  Sub {sub_num}: Standalone hoop + SHOT7M2 ({best_config})")

    # Generate blends with Sub 2503
    sub_2503 = pd.read_csv(SUBMISSION_DIR / "submission_2503.csv")
    for bw in [0.05, 0.10, 0.15, 0.20, 0.30]:
        sub_blend = sub_2503.copy()
        for col in TARGET_COLS:
            sub_blend[col] = bw * sub[col] + (1 - bw) * sub_2503[col]
            sub_blend[col] = sub_blend[col].clip(0, 1)

        sub_num = get_next_submission_number()
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub_blend.to_csv(sub_path, index=False)
        print(f"  Sub {sub_num}: {int(bw*100)}% SHOT7M2 + {int((1-bw)*100)}% Sub2503")

    # Also generate per-target blends (SHOT7M2 might help some targets more)
    # Use SHOT7M2 only for target where it helps most
    print("\n  Per-target selective blends:")
    for target in TARGETS:
        b = all_results["baseline (hoop only)"][target]
        s = all_results[best_config][target]
        pct = (s - b) / b * 100
        print(f"    {target}: baseline={b:.6f}, SHOT7M2={s:.6f} ({pct:+.2f}%)")

    print("\nDone!")


if __name__ == "__main__":
    main()

"""
SHOT7M2 Feature Extraction Pipeline

Extract multiple feature types from the SHOT7M2 dataset (7.2M frames of synthetic
basketball motion) and integrate them into our prediction pipeline.

SHOT7M2 has 26 joints, our data has 69 keypoints. 14 joints map between the two.
We normalize both to a body-size-invariant representation and extract:

1. Shooting PCA features: Project our poses into principal modes of shooting variation
2. Joint angle distribution z-scores: How unusual is each angle vs typical shooting form
3. Shooting form typicality: Mahalanobis distance from shooting pose centroid
4. Pose reconstruction error: How well does the shooting PCA basis reconstruct our pose
5. Shooting phase features: Position along the shooting arc (wind-up -> release -> follow-through)
6. Autoencoder features: Nonlinear pose embedding from a simple MLP autoencoder

All features are evaluated with honest LOO (PLS refit) to avoid the leakage artifact.
"""

import json
import fcntl
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from scipy.spatial.distance import mahalanobis
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

# =========================================================================
# JOINT MAPPING: 14 joints that map between our skeleton and SHOT7M2
# =========================================================================
# Our data column names -> SHOT7M2 joint indices
JOINT_MAP = {
    # Our name: SHOT7M2 index
    "mid_hip": 0,       # center
    "left_hip": 1,
    "left_knee": 2,
    "left_ankle": 3,
    "right_hip": 6,
    "right_knee": 7,
    "right_ankle": 8,
    "left_shoulder": 16,  # l_shoulder (not shoulder_blade)
    "left_elbow": 17,
    "left_wrist": 18,
    "right_shoulder": 23,  # r_shoulder
    "right_elbow": 24,
    "right_wrist": 25,
    "neck": 19,
}

# Joint names in order for feature extraction
MAPPED_JOINTS = list(JOINT_MAP.keys())
SHOT7M2_INDICES = [JOINT_MAP[j] for j in MAPPED_JOINTS]

# Joint angle definitions: (parent, joint, child) triplets using our joint names
JOINT_ANGLE_DEFS = [
    ("left_hip", "left_knee", "left_ankle"),     # left knee angle
    ("right_hip", "right_knee", "right_ankle"),   # right knee angle
    ("left_shoulder", "left_elbow", "left_wrist"),  # left elbow angle
    ("right_shoulder", "right_elbow", "right_wrist"),  # right elbow angle
    ("neck", "left_shoulder", "left_elbow"),      # left shoulder angle
    ("neck", "right_shoulder", "right_elbow"),    # right shoulder angle
    ("left_shoulder", "mid_hip", "right_shoulder"),  # torso lean
    ("left_hip", "mid_hip", "right_hip"),         # hip width angle
    ("mid_hip", "neck", "right_shoulder"),         # upper body angle R
    ("mid_hip", "neck", "left_shoulder"),          # upper body angle L
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


def normalize_pose(positions):
    """
    Normalize a pose to body-size-invariant representation.
    positions: (N, 14, 3) array of joint positions
    Returns: (N, 14, 3) normalized positions

    Normalization:
    1. Center on pelvis (mid_hip / center)
    2. Scale by torso length (hip to neck distance)
    3. Align facing direction (optional, skip for simplicity)
    """
    pelvis_idx = MAPPED_JOINTS.index("mid_hip")
    neck_idx = MAPPED_JOINTS.index("neck")

    # Center on pelvis
    centered = positions - positions[:, pelvis_idx:pelvis_idx+1, :]

    # Scale by torso length
    torso_len = np.linalg.norm(centered[:, neck_idx] - centered[:, pelvis_idx], axis=1, keepdims=True)
    torso_len = np.maximum(torso_len, 1e-6)  # avoid division by zero
    normalized = centered / torso_len[:, :, np.newaxis]

    return normalized


def compute_joint_angles(positions):
    """
    Compute joint angles from positions.
    positions: (N, 14, 3) array
    Returns: (N, n_angles) array of angles in radians
    """
    angles = np.zeros((len(positions), len(JOINT_ANGLE_DEFS)))
    for i, (parent, joint, child) in enumerate(JOINT_ANGLE_DEFS):
        p_idx = MAPPED_JOINTS.index(parent)
        j_idx = MAPPED_JOINTS.index(joint)
        c_idx = MAPPED_JOINTS.index(child)

        v1 = positions[:, p_idx] - positions[:, j_idx]
        v2 = positions[:, c_idx] - positions[:, j_idx]

        # Normalize
        v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
        v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)
        v1 = v1 / np.maximum(v1_norm, 1e-8)
        v2 = v2 / np.maximum(v2_norm, 1e-8)

        cos_angle = np.sum(v1 * v2, axis=1)
        cos_angle = np.clip(cos_angle, -1, 1)
        angles[:, i] = np.arccos(cos_angle)

    return angles


def load_shot7m2_shooting_poses():
    """Load all shooting frames from SHOT7M2 and extract the 14 mapped joints."""
    print("Loading SHOT7M2 data...")
    poses = np.load(SHOT7M2_DIR / "train" / "train_dictionary_poses.npy",
                    allow_pickle=True).item()
    actions = np.load(SHOT7M2_DIR / "train" / "train_dictionary_actions.npy",
                      allow_pickle=True).item()

    vocab = actions['vocabulary']
    labels = actions['label_array']
    fnm = actions['frame_number_map']

    shoot_idx = vocab.index('action_Shoot')
    shoot_labels = labels[shoot_idx]

    # Collect shooting frames with confidence > 0.3 (get more data)
    all_poses = []
    all_weights = []
    all_contexts = []  # store surrounding context for temporal features

    seq_keys = list(poses['sequences']['keypoints'].keys())
    for key in seq_keys:
        start, end = fnm[key]
        ep_shoot = shoot_labels[start:end]
        ep_poses = poses['sequences']['keypoints'][key]  # (1800, 1, 26, 3)

        # Find shooting segments
        shoot_mask = ep_shoot > 0.3
        if not np.any(shoot_mask):
            continue

        # Get shooting frames
        shoot_frames = np.where(shoot_mask)[0]
        for f in shoot_frames:
            pose = ep_poses[f, 0, SHOT7M2_INDICES, :]  # (14, 3)
            all_poses.append(pose)
            all_weights.append(ep_shoot[f])

        # Also get shooting segments with context (for temporal features)
        transitions = np.diff(shoot_mask.astype(int))
        seg_starts = np.where(transitions == 1)[0] + 1
        seg_ends = np.where(transitions == -1)[0] + 1

        if shoot_mask[0]:
            seg_starts = np.concatenate([[0], seg_starts])
        if shoot_mask[-1]:
            seg_ends = np.concatenate([seg_ends, [len(shoot_mask)]])

        for s, e in zip(seg_starts, seg_ends):
            # Get context: 30 frames before to 30 frames after
            ctx_start = max(0, s - 30)
            ctx_end = min(ep_poses.shape[0], e + 30)
            context_poses = ep_poses[ctx_start:ctx_end, 0, SHOT7M2_INDICES, :]  # (L, 14, 3)
            all_contexts.append({
                'poses': context_poses,
                'shoot_start': s - ctx_start,
                'shoot_end': e - ctx_start,
                'segment_len': e - s
            })

    shoot_poses = np.array(all_poses)
    shoot_weights = np.array(all_weights)

    print(f"  Shooting frames (conf>0.3): {len(shoot_poses)}")
    print(f"  Shooting segments: {len(all_contexts)}")
    print(f"  Mean segment length: {np.mean([c['segment_len'] for c in all_contexts]):.1f} frames")

    # Also collect ALL frames for general pose statistics
    all_general = []
    for key in seq_keys[:100]:  # sample 100 episodes for general stats
        ep_poses = poses['sequences']['keypoints'][key]
        # Sample every 10th frame
        sampled = ep_poses[::10, 0, SHOT7M2_INDICES, :]
        all_general.append(sampled)
    general_poses = np.concatenate(all_general)
    print(f"  General pose samples: {len(general_poses)}")

    return shoot_poses, shoot_weights, all_contexts, general_poses


def load_our_data(target_frame=153):
    """Load our competition data and extract the 14 mapped joints at the target frame."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    def parse_json_safe(s, frame):
        """Parse JSON string handling nan values."""
        s = str(s).replace('nan', 'null').replace('NaN', 'null')
        arr = json.loads(s)
        val = arr[frame]
        if val is None:
            # Try nearby frames
            for offset in range(1, 10):
                if frame - offset >= 0 and arr[frame - offset] is not None:
                    return float(arr[frame - offset])
                if frame + offset < len(arr) and arr[frame + offset] is not None:
                    return float(arr[frame + offset])
            return 0.0
        return float(val)

    def extract_joints(df, frame):
        """Extract 14 mapped joint positions at a specific frame."""
        n = len(df)
        positions = np.zeros((n, 14, 3))
        for j_idx, joint_name in enumerate(MAPPED_JOINTS):
            for c_idx, coord in enumerate(['x', 'y', 'z']):
                col = f"{joint_name}_{coord}"
                values = df[col].apply(lambda x: parse_json_safe(x, frame)).values
                positions[:, j_idx, c_idx] = values
        return positions

    train_poses = {}
    test_poses = {}
    for target, frame in TARGET_FRAMES.items():
        train_poses[target] = extract_joints(train_df, frame)
        test_poses[target] = extract_joints(test_df, frame)

    return train_df, test_df, train_poses, test_poses


def extract_all_features(our_poses, shoot_poses_norm, shoot_angles,
                         general_poses_norm, pca_model, angle_means, angle_stds,
                         cov_inv, centroid):
    """
    Extract all SHOT7M2-derived features for a set of our poses.
    our_poses: (N, 14, 3) raw poses
    Returns: dict of feature arrays
    """
    N = len(our_poses)
    features = {}

    # Normalize our poses
    our_norm = normalize_pose(our_poses)
    our_flat = our_norm.reshape(N, -1)  # (N, 42)

    # 1. PCA features: projection onto shooting variation modes
    pca_features = pca_model.transform(our_flat)
    for i in range(pca_features.shape[1]):
        features[f'shot7m2_pca_{i}'] = pca_features[:, i]

    # 2. PCA reconstruction error
    reconstructed = pca_model.inverse_transform(pca_features)
    recon_error = np.mean((our_flat - reconstructed) ** 2, axis=1)
    features['shot7m2_recon_error'] = recon_error

    # 3. Joint angle z-scores
    our_angles = compute_joint_angles(our_norm)
    for i in range(our_angles.shape[1]):
        if angle_stds[i] > 1e-8:
            features[f'shot7m2_angle_zscore_{i}'] = (our_angles[:, i] - angle_means[i]) / angle_stds[i]
        else:
            features[f'shot7m2_angle_zscore_{i}'] = np.zeros(N)

    # 4. Joint angle raw values (from our normalized skeleton)
    for i in range(our_angles.shape[1]):
        features[f'shot7m2_angle_raw_{i}'] = our_angles[:, i]

    # 5. Mahalanobis distance from shooting centroid
    try:
        mahal_dists = np.array([
            mahalanobis(our_flat[i], centroid, cov_inv)
            for i in range(N)
        ])
    except Exception:
        # Fallback to Euclidean if covariance is singular
        mahal_dists = np.linalg.norm(our_flat - centroid, axis=1)
    features['shot7m2_mahal_dist'] = mahal_dists

    # 6. Euclidean distance to shooting centroid
    features['shot7m2_eucl_dist'] = np.linalg.norm(our_flat - centroid, axis=1)

    # 7. Nearest-neighbor features in shooting PCA space
    shoot_pca = pca_model.transform(shoot_poses_norm.reshape(len(shoot_poses_norm), -1))
    for k in [5, 20]:
        nn_dists = np.zeros(N)
        nn_angle_sim = np.zeros(N)
        for i in range(N):
            dists = np.linalg.norm(shoot_pca - pca_features[i], axis=1)
            nearest_k = np.argsort(dists)[:k]
            nn_dists[i] = np.mean(dists[nearest_k])
            # Mean angle similarity to k nearest shooting poses
            nn_angle_sim[i] = np.mean(np.abs(
                shoot_angles[nearest_k] - our_angles[i]
            ))
        features[f'shot7m2_nn{k}_dist'] = nn_dists
        features[f'shot7m2_nn{k}_angle_diff'] = nn_angle_sim

    # 8. Limb ratio features (body proportions relative to shooting population)
    # Upper arm length ratio (shoulder to elbow)
    r_upper = np.linalg.norm(our_norm[:, MAPPED_JOINTS.index("right_shoulder")] -
                              our_norm[:, MAPPED_JOINTS.index("right_elbow")], axis=1)
    l_upper = np.linalg.norm(our_norm[:, MAPPED_JOINTS.index("left_shoulder")] -
                              our_norm[:, MAPPED_JOINTS.index("left_elbow")], axis=1)
    features['shot7m2_arm_asymmetry'] = r_upper - l_upper

    # Wrist height relative to head
    neck_y = our_norm[:, MAPPED_JOINTS.index("neck"), 1]
    r_wrist_y = our_norm[:, MAPPED_JOINTS.index("right_wrist"), 1]
    l_wrist_y = our_norm[:, MAPPED_JOINTS.index("left_wrist"), 1]
    features['shot7m2_r_wrist_height'] = r_wrist_y - neck_y
    features['shot7m2_l_wrist_height'] = l_wrist_y - neck_y

    # Arm extension (wrist distance from shoulder, normalized)
    r_ext = np.linalg.norm(our_norm[:, MAPPED_JOINTS.index("right_wrist")] -
                           our_norm[:, MAPPED_JOINTS.index("right_shoulder")], axis=1)
    l_ext = np.linalg.norm(our_norm[:, MAPPED_JOINTS.index("left_wrist")] -
                           our_norm[:, MAPPED_JOINTS.index("left_shoulder")], axis=1)
    features['shot7m2_r_arm_extension'] = r_ext
    features['shot7m2_l_arm_extension'] = l_ext

    # Knee bend (how bent are the knees)
    features['shot7m2_r_knee_angle'] = our_angles[:, JOINT_ANGLE_DEFS.index(
        ("right_hip", "right_knee", "right_ankle"))]
    features['shot7m2_l_knee_angle'] = our_angles[:, JOINT_ANGLE_DEFS.index(
        ("left_hip", "left_knee", "left_ankle"))]

    return features


def honest_loo_evaluate(train_features, train_targets, player_ids, n_pls=5):
    """
    Honest LOO with PLS refit (no leakage).
    Uses per-player models with PLS + Ridge, refitting PLS excluding held-out point.
    """
    n = len(train_targets)
    predictions = np.zeros(n)

    players = sorted(np.unique(player_ids))

    for pid in players:
        mask = player_ids == pid
        idx = np.where(mask)[0]
        n_player = len(idx)

        X_player = train_features[mask]
        y_player = train_targets[mask]

        for i_local in range(n_player):
            i_global = idx[i_local]

            # Leave one out
            X_train = np.delete(X_player, i_local, axis=0)
            y_train = np.delete(y_player, i_local)
            X_test = X_player[i_local:i_local+1]

            # Scale
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # PLS (honest - fit without held-out point)
            n_pls_actual = min(n_pls, X_train_s.shape[1], X_train_s.shape[0])
            if n_pls_actual < 1:
                predictions[i_global] = np.mean(y_train)
                continue

            pls = PLSRegression(n_components=n_pls_actual)
            pls.fit(X_train_s, y_train)
            X_train_pls = pls.transform(X_train_s)
            X_test_pls = pls.transform(X_test_s)

            # Combine original + PLS
            X_train_full = np.hstack([X_train_s, X_train_pls])
            X_test_full = np.hstack([X_test_s, X_test_pls])

            # Locally weighted Ridge
            # Compute distances
            dists = np.linalg.norm(X_train_full - X_test_full, axis=1)
            bw = np.quantile(dists, 0.45)  # wider bandwidth for robustness
            weights = np.exp(-0.5 * (dists / max(bw, 1e-8)) ** 2)

            # Weighted Ridge
            W = np.diag(np.sqrt(weights))
            ridge = Ridge(alpha=10.0)
            ridge.fit(W @ X_train_full, W @ y_train)
            predictions[i_global] = ridge.predict(X_test_full)[0]

    return predictions


def main():
    print("=" * 70)
    print("SHOT7M2 FEATURE EXTRACTION PIPELINE")
    print("=" * 70)

    # =====================================================================
    # STEP 1: Load and process SHOT7M2 data
    # =====================================================================
    shoot_poses, shoot_weights, contexts, general_poses = load_shot7m2_shooting_poses()

    # Normalize SHOT7M2 poses
    print("\nNormalizing SHOT7M2 poses...")
    shoot_norm = normalize_pose(shoot_poses)
    general_norm = normalize_pose(general_poses)

    # Compute shooting joint angles
    shoot_angles = compute_joint_angles(shoot_norm)
    angle_means = np.mean(shoot_angles, axis=0)
    angle_stds = np.std(shoot_angles, axis=0)
    print(f"  Shooting angle means: {angle_means}")
    print(f"  Shooting angle stds: {angle_stds}")

    # Fit PCA on shooting poses
    shoot_flat = shoot_norm.reshape(len(shoot_norm), -1)  # (N, 42)
    n_pca = min(20, shoot_flat.shape[1])
    pca = PCA(n_components=n_pca)
    pca.fit(shoot_flat)
    print(f"\n  PCA explained variance (first 10): {pca.explained_variance_ratio_[:10]}")
    print(f"  PCA cumulative (10): {np.cumsum(pca.explained_variance_ratio_[:10])[-1]:.4f}")
    print(f"  PCA cumulative (20): {np.sum(pca.explained_variance_ratio_):.4f}")

    # Compute centroid and covariance for Mahalanobis distance
    centroid = np.mean(shoot_flat, axis=0)
    cov = np.cov(shoot_flat.T)
    try:
        cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    # =====================================================================
    # STEP 2: Load our data
    # =====================================================================
    print("\nLoading our competition data...")
    train_df, test_df, train_poses, test_poses = load_our_data()
    player_ids = train_df['participant_id'].values
    test_pids = test_df['participant_id'].values

    # Load scalers for targets
    import joblib
    scalers = {}
    for target in TARGETS:
        scalers[target] = joblib.load(DATA_DIR / f"scaler_{target}.pkl")

    # Compute scaled targets
    scaled_targets = {}
    for target in TARGETS:
        raw = train_df[target].values
        scaled_targets[target] = scalers[target].transform(raw.reshape(-1, 1)).ravel()

    # =====================================================================
    # STEP 3: Extract features for each target frame
    # =====================================================================
    print("\nExtracting SHOT7M2 features...")
    train_features_by_target = {}
    test_features_by_target = {}

    for target in TARGETS:
        print(f"\n  Target: {target} (frame {TARGET_FRAMES[target]})")
        train_feats = extract_all_features(
            train_poses[target], shoot_norm, shoot_angles,
            general_norm, pca, angle_means, angle_stds, cov_inv, centroid
        )
        test_feats = extract_all_features(
            test_poses[target], shoot_norm, shoot_angles,
            general_norm, pca, angle_means, angle_stds, cov_inv, centroid
        )

        # Convert to arrays
        feat_names = sorted(train_feats.keys())
        train_arr = np.column_stack([train_feats[f] for f in feat_names])
        test_arr = np.column_stack([test_feats[f] for f in feat_names])

        print(f"    Features: {len(feat_names)} ({train_arr.shape})")
        print(f"    Feature names: {feat_names[:10]}...")

        train_features_by_target[target] = train_arr
        test_features_by_target[target] = test_arr

    # =====================================================================
    # STEP 4: Also load our existing hoop-relative features (for combination)
    # =====================================================================
    print("\nBuilding hoop-relative features...")
    hoop_train_features = {}
    hoop_test_features = {}

    for target in TARGETS:
        frame = TARGET_FRAMES[target]
        train_hoop = build_hoop_relative_features(train_df, frame)
        test_hoop = build_hoop_relative_features(test_df, frame)
        hoop_train_features[target] = train_hoop
        hoop_test_features[target] = test_hoop

    # =====================================================================
    # STEP 5: Evaluate with honest LOO
    # =====================================================================
    print("\n" + "=" * 70)
    print("EVALUATION: Honest LOO (PLS refit, no leakage)")
    print("=" * 70)

    configs = {
        "SHOT7M2 features only": {
            "use_shot7m2": True, "use_hoop": False
        },
        "Hoop-relative only (baseline)": {
            "use_shot7m2": False, "use_hoop": True
        },
        "Hoop-relative + SHOT7M2": {
            "use_shot7m2": True, "use_hoop": True
        },
    }

    results = {}
    for config_name, cfg in configs.items():
        print(f"\n  Config: {config_name}")
        target_mse = {}

        for target in TARGETS:
            # Build feature matrix
            parts = []
            if cfg["use_hoop"]:
                parts.append(hoop_train_features[target])
            if cfg["use_shot7m2"]:
                parts.append(train_features_by_target[target])

            X_train = np.hstack(parts) if len(parts) > 1 else parts[0]
            y_train = scaled_targets[target]

            # Honest LOO
            preds = honest_loo_evaluate(X_train, y_train, player_ids, n_pls=5)
            mse = np.mean((preds - y_train) ** 2)
            target_mse[target] = mse
            print(f"    {target}: LOO={mse:.6f}")

        mean_mse = np.mean(list(target_mse.values()))
        results[config_name] = {**target_mse, "mean": mean_mse}
        print(f"    mean: {mean_mse:.6f}")

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    baseline_mean = results["Hoop-relative only (baseline)"]["mean"]
    for config_name, res in results.items():
        pct = (res["mean"] - baseline_mean) / baseline_mean * 100
        print(f"  {config_name:40s}: mean={res['mean']:.6f} ({pct:+.2f}%)")

    # =====================================================================
    # STEP 6: Generate submissions if SHOT7M2 helps
    # =====================================================================
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS")
    print("=" * 70)

    # Generate standalone SHOT7M2 submission
    for config_name, cfg in configs.items():
        if config_name == "Hoop-relative only (baseline)":
            continue

        print(f"\n  Generating: {config_name}")
        sub = pd.read_csv(SUBMISSION_DIR / "submission_2503.csv")

        for t_idx, target in enumerate(TARGETS):
            parts_train = []
            parts_test = []
            if cfg["use_hoop"]:
                parts_train.append(hoop_train_features[target])
                parts_test.append(hoop_test_features[target])
            if cfg["use_shot7m2"]:
                parts_train.append(train_features_by_target[target])
                parts_test.append(test_features_by_target[target])

            X_train = np.hstack(parts_train) if len(parts_train) > 1 else parts_train[0]
            X_test = np.hstack(parts_test) if len(parts_test) > 1 else parts_test[0]
            y_train = scaled_targets[target]

            # Train full model per player
            preds = np.zeros(len(X_test))
            players = sorted(np.unique(player_ids))

            for pid in players:
                train_mask = player_ids == pid
                test_mask = test_pids == pid

                X_tr = X_train[train_mask]
                y_tr = y_train[train_mask]
                X_te = X_test[test_mask]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)

                n_pls_actual = min(5, X_tr_s.shape[1], X_tr_s.shape[0] - 1)
                pls = PLSRegression(n_components=n_pls_actual)
                pls.fit(X_tr_s, y_tr)
                X_tr_pls = pls.transform(X_tr_s)
                X_te_pls = pls.transform(X_te_s)

                X_tr_full = np.hstack([X_tr_s, X_tr_pls])
                X_te_full = np.hstack([X_te_s, X_te_pls])

                # Predict each test point with locally weighted Ridge
                for i in range(len(X_te_full)):
                    dists = np.linalg.norm(X_tr_full - X_te_full[i], axis=1)
                    bw = np.quantile(dists, 0.45)
                    weights = np.exp(-0.5 * (dists / max(bw, 1e-8)) ** 2)

                    W = np.diag(np.sqrt(weights))
                    ridge = Ridge(alpha=10.0)
                    ridge.fit(W @ X_tr_full, W @ y_tr)
                    test_idx = np.where(test_mask)[0]
                    preds[test_idx[i]] = ridge.predict(X_te_full[i:i+1])[0]

            sub[TARGET_COLS[t_idx]] = np.clip(preds, 0, 1)

        sub_num = get_next_submission_number()
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub.to_csv(sub_path, index=False)
        print(f"    Sub {sub_num}: {config_name} (honest LOO: {results[config_name]['mean']:.6f})")

    # Also generate blend submissions with Sub 2503
    sub_2503 = pd.read_csv(SUBMISSION_DIR / "submission_2503.csv")

    for blend_weight in [0.05, 0.10, 0.15, 0.20]:
        # Blend with the best SHOT7M2 config
        best_config = min(
            [k for k in results if k != "Hoop-relative only (baseline)"],
            key=lambda k: results[k]["mean"]
        )

        # Reload the standalone submission we just made
        # Generate fresh predictions for blend
        sub_new = sub_2503.copy()

        for t_idx, target in enumerate(TARGETS):
            parts_train = []
            parts_test = []
            cfg = configs[best_config]
            if cfg["use_hoop"]:
                parts_train.append(hoop_train_features[target])
                parts_test.append(hoop_test_features[target])
            if cfg["use_shot7m2"]:
                parts_train.append(train_features_by_target[target])
                parts_test.append(test_features_by_target[target])

            X_train = np.hstack(parts_train) if len(parts_train) > 1 else parts_train[0]
            X_test = np.hstack(parts_test) if len(parts_test) > 1 else parts_test[0]
            y_train = scaled_targets[target]

            preds = np.zeros(len(X_test))
            players = sorted(np.unique(player_ids))

            for pid in players:
                train_mask = player_ids == pid
                test_mask = test_pids == pid

                X_tr = X_train[train_mask]
                y_tr = y_train[train_mask]
                X_te = X_test[test_mask]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)

                n_pls_actual = min(5, X_tr_s.shape[1], X_tr_s.shape[0] - 1)
                pls = PLSRegression(n_components=n_pls_actual)
                pls.fit(X_tr_s, y_tr)
                X_tr_pls = pls.transform(X_tr_s)
                X_te_pls = pls.transform(X_te_s)

                X_tr_full = np.hstack([X_tr_s, X_tr_pls])
                X_te_full = np.hstack([X_te_s, X_te_pls])

                for i in range(len(X_te_full)):
                    dists = np.linalg.norm(X_tr_full - X_te_full[i], axis=1)
                    bw = np.quantile(dists, 0.45)
                    weights = np.exp(-0.5 * (dists / max(bw, 1e-8)) ** 2)

                    W = np.diag(np.sqrt(weights))
                    ridge = Ridge(alpha=10.0)
                    ridge.fit(W @ X_tr_full, W @ y_tr)
                    test_idx = np.where(test_mask)[0]
                    preds[test_idx[i]] = ridge.predict(X_te_full[i:i+1])[0]

            col = TARGET_COLS[t_idx]
            sub_new[col] = blend_weight * np.clip(preds, 0, 1) + (1 - blend_weight) * sub_2503[col]
            sub_new[col] = sub_new[col].clip(0, 1)

        sub_num = get_next_submission_number()
        sub_path = SUBMISSION_DIR / f"submission_{sub_num}.csv"
        sub_new.to_csv(sub_path, index=False)
        print(f"    Sub {sub_num}: {int(blend_weight*100)}% {best_config} + {int((1-blend_weight)*100)}% Sub2503")

    print("\nDone!")


def build_hoop_relative_features(df, frame):
    """Build hoop-relative features for our data at a specific frame."""
    n = len(df)
    kp_names = sorted(set(c.rsplit('_', 1)[0]
                          for c in df.columns if c.endswith(('_x', '_y', '_z'))))

    def parse_safe(s, f):
        s = str(s).replace('nan', 'null').replace('NaN', 'null')
        arr = json.loads(s)
        val = arr[f]
        if val is None:
            for offset in range(1, 10):
                if f - offset >= 0 and arr[f - offset] is not None:
                    return float(arr[f - offset])
                if f + offset < len(arr) and arr[f + offset] is not None:
                    return float(arr[f + offset])
            return 0.0
        return float(val)

    positions = np.zeros((n, len(kp_names), 3))
    for k_idx, kp in enumerate(kp_names):
        for c_idx, coord in enumerate(['x', 'y', 'z']):
            col = f"{kp}_{coord}"
            positions[:, k_idx, c_idx] = df[col].apply(
                lambda x: parse_safe(x, frame)).values

    # Hoop-relative: subtract hoop position
    hoop_rel = positions - HOOP[np.newaxis, np.newaxis, :]

    # Flatten: all keypoints x 3 coords
    features = hoop_rel.reshape(n, -1)

    return features


if __name__ == "__main__":
    main()

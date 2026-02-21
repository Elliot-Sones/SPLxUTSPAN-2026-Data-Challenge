"""
Body-Alignment Features for Basketball Shot Prediction

Key finding from correlation analysis:
- P5 LR: right wrist y-offset from shoulder center at frame 180 has r=0.51
- P5 LR: left wrist x-offset from shoulder center at frames 160-170 has r=0.37
- P5 LR: body facing angle at frames 165-170 has r=0.33
- All players angle: body_facing_xy at frame 153 has r=0.59 (global)

These are BODY-NORMALIZED alignment features - positions relative to the
body center rather than absolute or hoop-relative positions.

The existing pipeline uses hoop-relative coordinates, not body-relative.
This script adds them as new features to the per-example kernel Ridge.

Strategy:
- For each target, extract alignment features at the target frame AND
  the follow-through frame (+10 frames) for LR
- Add ~40 new features per shot
- Run same per-player kernel Ridge with PLS compression
- Honest LOO CV to validate before any submission
"""
import json
import time
import fcntl
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "output"
SUBMISSION_DIR = PROJECT_DIR / "submission"

TARGETS     = ["angle", "depth", "left_right"]
TARGET_COLS = ["scaled_angle", "scaled_depth", "scaled_left_right"]
# Per-target frames (confirmed winners)
TARGET_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}
# Follow-through frames (new - where alignment signal is strongest)
FOLLOWTHROUGH_FRAMES = {"angle": 158, "depth": 155, "left_right": 180}

HOOP_POS = np.array([5.25, -25.0, 10.0])
DT = 1.0 / 60.0

# COCO 17 joints
JOINTS_17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
J17_IDX = {name: i for i, name in enumerate(JOINTS_17)}


def get_next_submission_number():
    lock_path = SUBMISSION_DIR / ".submission_lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            existing = list(SUBMISSION_DIR.glob("submission_*.csv"))
            nums = [int(fp.stem.split("_")[1]) for fp in existing
                    if fp.stem.split("_")[1].isdigit()]
            nxt = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{nxt}.csv").touch(exist_ok=True)
            return nxt
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def parse_seq(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = str(s).replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)


def safe_grad(arr, dt=DT):
    arr = arr.copy().astype(np.float64)
    bad = np.isnan(arr) | np.isinf(arr)
    if np.any(bad) and not np.all(bad):
        good = ~bad
        arr[bad] = np.interp(np.where(bad)[0], np.where(good)[0], arr[good])
    return np.gradient(arr, dt)


def load_sequences(df):
    """Load (N, 240, 17, 3) joint sequences for 17 COCO joints."""
    n = len(df)
    seqs = np.zeros((n, 240, 17, 3), dtype=np.float32)
    for j_idx, jname in enumerate(JOINTS_17):
        for c_idx, coord in enumerate(["x", "y", "z"]):
            col = f"{jname}_{coord}"
            if col not in df.columns:
                continue
            for i, s in enumerate(df[col]):
                seqs[i, :, j_idx, c_idx] = parse_seq(s)
    return seqs  # (N, 240, 17, 3)


def compute_body_alignment_features(seqs, target):
    """
    Compute body-normalized alignment features.

    seqs: (N, 240, 17, 3)
    Returns: (N, n_feats) array
    """
    N = seqs.shape[0]
    frame_rel = TARGET_FRAMES[target]
    frame_ft  = FOLLOWTHROUGH_FRAMES[target]

    ls_i  = J17_IDX["left_shoulder"]
    rs_i  = J17_IDX["right_shoulder"]
    le_i  = J17_IDX["left_elbow"]
    re_i  = J17_IDX["right_elbow"]
    lw_i  = J17_IDX["left_wrist"]
    rw_i  = J17_IDX["right_wrist"]
    lh_i  = J17_IDX["left_hip"]
    rh_i  = J17_IDX["right_hip"]
    no_i  = J17_IDX["nose"]
    la_i  = J17_IDX["left_ankle"]
    ra_i  = J17_IDX["right_ankle"]

    all_feats = []

    for i in range(N):
        s = seqs[i]  # (240, 17, 3)
        feats = []

        for frame in [frame_rel, frame_ft]:
            f = int(np.clip(frame, 0, 239))
            # Body reference points
            mid_sho = (s[f, ls_i] + s[f, rs_i]) / 2   # (3,)
            mid_hip = (s[f, lh_i] + s[f, rh_i]) / 2   # (3,)

            # Right wrist offset from mid-shoulder
            rw_off = s[f, rw_i] - mid_sho
            feats.extend(rw_off.tolist())           # dx, dy, dz

            # Left wrist offset from mid-shoulder
            lw_off = s[f, lw_i] - mid_sho
            feats.extend(lw_off.tolist())           # dx, dy, dz

            # Right elbow offset from mid-shoulder
            re_off = s[f, re_i] - mid_sho
            feats.extend(re_off.tolist())           # dx, dy, dz

            # Shoulder rotation angle in xy plane
            sho_vec = s[f, rs_i] - s[f, ls_i]
            sho_ang = np.arctan2(sho_vec[1], sho_vec[0])
            feats.append(float(sho_ang))

            # Hip rotation angle in xy plane
            hip_vec = s[f, rh_i] - s[f, lh_i]
            hip_ang = np.arctan2(hip_vec[1], hip_vec[0])
            feats.append(float(hip_ang))

            # Body facing: nose relative to mid-hip
            nose_off = s[f, no_i] - mid_hip
            body_facing = np.arctan2(nose_off[1], nose_off[0])
            feats.append(float(body_facing))

            # Shoulder-hip twist (rotation mismatch)
            sho_twist = (s[f, rs_i, 0] - s[f, ls_i, 0])
            hip_twist  = (s[f, rh_i, 0] - s[f, lh_i, 0])
            feats.append(float(sho_twist - hip_twist))

            # Elbow-wrist z height difference (arm extension)
            feats.append(float(s[f, rw_i, 2] - s[f, re_i, 2]))
            feats.append(float(s[f, re_i, 2] - s[f, rs_i, 2]))

            # Ankle lateral spread (stance)
            feats.append(float(s[f, ra_i, 0] - s[f, la_i, 0]))

            # Wrist-wrist separation (guide hand vs shooting hand)
            ww_sep = s[f, rw_i] - s[f, lw_i]
            feats.extend(ww_sep[:2].tolist())  # x and y separation

        # Velocity of right wrist in y direction around release frame
        # (captures wrist lateral swing speed)
        rwy_series = s[:, rw_i, 1].astype(np.float64)
        bad = np.isnan(rwy_series) | np.isinf(rwy_series)
        if not np.all(bad) and np.any(bad):
            good = ~bad
            rwy_series[bad] = np.interp(np.where(bad)[0], np.where(good)[0], rwy_series[good])
        rwy_vel = np.gradient(rwy_series, DT)

        f_rel = int(np.clip(frame_rel, 0, 239))
        f_ft  = int(np.clip(frame_ft, 0, 239))
        feats.append(float(rwy_vel[f_rel]))
        feats.append(float(rwy_vel[f_ft]))
        feats.append(float(np.mean(rwy_vel[f_rel-5:f_rel+5])))

        # Body facing angle change from frame_rel to frame_ft (follow-through drift)
        mid_hip_rel = (s[f_rel, lh_i] + s[f_rel, rh_i]) / 2
        mid_hip_ft  = (s[f_ft,  lh_i] + s[f_ft,  rh_i]) / 2
        nose_rel = s[f_rel, no_i] - mid_hip_rel
        nose_ft  = s[f_ft,  no_i] - mid_hip_ft
        facing_rel = np.arctan2(nose_rel[1], nose_rel[0])
        facing_ft  = np.arctan2(nose_ft[1],  nose_ft[0])
        feats.append(float(facing_ft - facing_rel))

        all_feats.append(feats)

    X = np.array(all_feats, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def compute_hoop_relative_features(seqs, target):
    """
    Compact hoop-relative features matching the existing pipeline.
    seqs: (N, 240, 17, 3)
    Returns (N, n_feats)
    """
    N = seqs.shape[0]
    frame = TARGET_FRAMES[target]
    all_feats = []

    for i in range(N):
        s = seqs[i]  # (240, 17, 3)
        f = int(np.clip(frame, 0, 239))

        # Hoop-center transform (same as existing pipeline)
        mid_hip_pos = (s[120, J17_IDX["left_hip"]] + s[120, J17_IDX["right_hip"]]) / 2
        player_pos = mid_hip_pos.copy(); player_pos[2] = 0
        forward_2d = HOOP_POS[:2] - player_pos[:2]
        fn = np.linalg.norm(forward_2d)
        if fn > 1e-6: forward_2d /= fn
        else: forward_2d = np.array([0.0, -1.0])
        lateral_2d = np.array([-forward_2d[1], forward_2d[0]])

        R = np.eye(3, dtype=np.float32)
        R[0, :2] = forward_2d; R[1, :2] = lateral_2d

        centered = s - player_pos.reshape(1, 1, 3)
        # (240, 17, 3)
        hr = np.einsum("ij,fkj->fki", R, centered)

        feats = []
        key_joints = [
            "right_wrist", "right_elbow", "right_shoulder",
            "left_wrist", "left_shoulder",
            "left_hip", "right_hip",
            "left_knee", "right_knee", "nose"
        ]
        for jname in key_joints:
            ji = J17_IDX.get(jname)
            if ji is None:
                feats.extend([0.0] * 6)
                continue
            for c in range(3):
                feats.append(float(hr[f, ji, c]))
                vel = safe_grad(hr[:, ji, c])
                feats.append(float(vel[f]))

        # Multi-frame: also extract at follow-through
        ft_frame = int(np.clip(FOLLOWTHROUGH_FRAMES[target], 0, 239))
        for jname in ["right_wrist", "right_elbow"]:
            ji = J17_IDX.get(jname)
            if ji is None:
                feats.extend([0.0] * 3)
                continue
            for c in range(3):
                feats.append(float(hr[ft_frame, ji, c]))

        # Release window stats
        rw_i = J17_IDX["right_wrist"]
        for c in range(3):
            series = hr[140:185, rw_i, c]
            feats.append(float(np.nanmean(series)))
            feats.append(float(np.nanstd(series)))
            vel = safe_grad(hr[:, rw_i, c])
            feats.append(float(np.nanmax(np.abs(vel[140:185]))))

        all_feats.append(feats)

    X = np.array(all_feats, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def build_features(seqs, target):
    """Combine hoop-relative + body-alignment features."""
    X_hr   = compute_hoop_relative_features(seqs, target)
    X_body = compute_body_alignment_features(seqs, target)
    return np.hstack([X_hr, X_body])


def run_per_player_loo(X_all, y_all, pids, scalers, target, bw_quantile=0.3,
                        alpha=10.0, n_pls=15):
    """
    HONEST per-player LOO CV: PLS is refit within each fold (excluding held-out point).
    This avoids the PLS leakage bug documented in MEMORY.md.
    Leaky LOO inflates by ~3x; honest LOO matches LB within ~5%.
    """
    N = len(y_all)
    oof = np.zeros(N, dtype=np.float64)
    unique_pids = np.unique(pids)

    for pid in unique_pids:
        mask = pids == pid
        X_p  = X_all[mask]
        y_p  = y_all[mask]
        idx  = np.where(mask)[0]
        n_p  = len(X_p)

        for i in range(n_p):
            train_idx = [j for j in range(n_p) if j != i]
            X_tr_raw = X_p[train_idx]
            y_tr     = y_p[train_idx]
            X_te_raw = X_p[[i]]

            # Scaler fit on train only
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr_raw)
            X_te_s = sc.transform(X_te_raw)

            # PLS fit on train only (honest - no leakage)
            n_comp = min(n_pls, X_tr_s.shape[1], len(train_idx) - 1)
            pls = PLSRegression(n_components=n_comp, max_iter=500)
            pls.fit(X_tr_s, y_tr)
            X_tr_pls = pls.transform(X_tr_s)
            X_te_pls = pls.transform(X_te_s)

            # Kernel weights
            D_tr = cdist(X_tr_pls, X_tr_pls, metric="euclidean")
            d_i  = cdist(X_te_pls, X_tr_pls, metric="euclidean")[0]
            bw   = np.quantile(D_tr[D_tr > 0], bw_quantile) if np.any(D_tr > 0) else 1.0
            w    = np.maximum(np.exp(-0.5 * (d_i / bw) ** 2), 1e-6)

            reg = Ridge(alpha=alpha)
            reg.fit(X_tr_pls, y_tr, sample_weight=w)
            oof[idx[i]] = reg.predict(X_te_pls)[0]

    return oof


def run_test_predictions(X_train, y_train, X_test, pids_train, pids_test,
                          bw_quantile=0.3, alpha=10.0, n_pls=15):
    """Generate test predictions using all training data per player."""
    N_test = len(X_test)
    preds  = np.zeros(N_test, dtype=np.float64)
    unique_pids = np.unique(pids_train)

    for pid in unique_pids:
        tr_mask = pids_train == pid
        te_mask = pids_test == pid
        if not np.any(te_mask):
            continue

        X_tr = X_train[tr_mask]
        y_tr = y_train[tr_mask]
        X_te = X_test[te_mask]
        n_tr = len(X_tr)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        n_comp = min(n_pls, X_tr_s.shape[1], n_tr - 1)
        pls = PLSRegression(n_components=n_comp, max_iter=500)
        pls.fit(X_tr_s, y_tr)
        X_tr_pls = pls.transform(X_tr_s)
        X_te_pls = pls.transform(X_te_s)

        D_tr  = cdist(X_tr_pls, X_tr_pls, metric="euclidean")
        D_te  = cdist(X_te_pls, X_tr_pls, metric="euclidean")
        bw    = np.quantile(D_tr[D_tr > 0], bw_quantile) if np.any(D_tr > 0) else 1.0

        te_idx = np.where(te_mask)[0]
        for k, gi in enumerate(te_idx):
            d_k = D_te[k]
            w   = np.maximum(np.exp(-0.5 * (d_k / bw) ** 2), 1e-6)
            reg = Ridge(alpha=alpha)
            reg.fit(X_tr_pls, y_tr, sample_weight=w)
            preds[gi] = reg.predict(X_te_pls[[k]])[0]

    return preds


def main():
    t0 = time.time()
    print("=== Body-Alignment Feature Pipeline ===")

    # Load data
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df  = pd.read_csv(DATA_DIR  / "test.csv")
    scalers  = {t: joblib.load(DATA_DIR / f"scaler_{t}.pkl") for t in TARGETS}

    pids_train = train_df["participant_id"].values
    pids_test  = test_df["participant_id"].values

    # Scale targets
    y_train = {}
    for t in TARGETS:
        y_train[t] = scalers[t].transform(
            train_df[t].values.reshape(-1, 1)
        ).ravel().astype(np.float64)

    print("Parsing train sequences...")
    seqs_train = load_sequences(train_df)
    print("Parsing test sequences...")
    seqs_test  = load_sequences(test_df)

    results  = {}
    oof_all  = {}
    test_all = {}

    for target in TARGETS:
        print(f"\n--- {target} ---")
        t1 = time.time()

        print("  Building features...")
        X_tr = build_features(seqs_train, target)
        X_te = build_features(seqs_test,  target)
        print(f"  Feature shape: {X_tr.shape}")

        y_tr = y_train[target]

        print("  LOO CV (per-player kernel Ridge)...")
        oof = run_per_player_loo(X_tr, y_tr, pids_train, scalers, target)
        mse = float(np.mean((oof - y_tr) ** 2))
        print(f"  LOO MSE: {mse:.6f}  ({time.time()-t1:.1f}s)")
        results[target] = mse
        oof_all[target] = oof

        print("  Test predictions...")
        test_preds = run_test_predictions(X_tr, y_tr, X_te, pids_train, pids_test)
        test_all[target] = np.clip(test_preds, 0.0, 1.0)

    mean_mse = float(np.mean(list(results.values())))
    print(f"\n=== LOO Results ===")
    for t, v in results.items():
        print(f"  {t}: {v:.6f}")
    print(f"  mean:  {mean_mse:.6f}")
    print(f"  (honest LOO baseline: ~0.006830)")
    print(f"  Total time: {time.time()-t0:.1f}s")

    # Build submissions
    print("\nBuilding submissions...")

    # 1. Standalone
    sub1_num = get_next_submission_number()
    sub_sa = pd.DataFrame({"id": test_df["id"].values})
    for t, col in zip(TARGETS, TARGET_COLS):
        sub_sa[col] = test_all[t]
    sub_sa.to_csv(SUBMISSION_DIR / f"submission_{sub1_num}.csv", index=False)

    # 2. Blend with Sub3411 (5%)
    ref411 = pd.read_csv(SUBMISSION_DIR / "submission_3507.csv")  # current best
    sub2_num = get_next_submission_number()
    sub_blend = pd.DataFrame({"id": test_df["id"].values})
    w = 0.05
    for t, col in zip(TARGETS, TARGET_COLS):
        sub_blend[col] = w * test_all[t] + (1-w) * ref411[col].values
    sub_blend.to_csv(SUBMISSION_DIR / f"submission_{sub2_num}.csv", index=False)

    # 3. 3% blend
    sub3_num = get_next_submission_number()
    sub_blend3 = pd.DataFrame({"id": test_df["id"].values})
    w3 = 0.03
    for t, col in zip(TARGETS, TARGET_COLS):
        sub_blend3[col] = w3 * test_all[t] + (1-w3) * ref411[col].values
    sub_blend3.to_csv(SUBMISSION_DIR / f"submission_{sub3_num}.csv", index=False)

    print(f"\nSubmissions created:")
    print(f"  Sub {sub1_num}: Standalone body-alignment pipeline (CV={mean_mse:.6f})")
    print(f"  Sub {sub2_num}: 5% body-alignment + 95% Sub3507 (current best)")
    print(f"  Sub {sub3_num}: 3% body-alignment + 97% Sub3507")

    # Save results JSON
    import json as _json
    out = {
        "cv_scores": results,
        "mean_cv": mean_mse,
        "honest_loo_baseline": 0.006830,
        "total_time_s": time.time()-t0,
        "submissions": [
            {"num": sub1_num, "desc": f"Standalone body-alignment (CV={mean_mse:.6f})"},
            {"num": sub2_num, "desc": f"5% body-alignment + 95% Sub3507"},
            {"num": sub3_num, "desc": f"3% body-alignment + 97% Sub3507"},
        ]
    }
    ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"body_alignment_run_{ts}.json", "w") as f:
        _json.dump(out, f, indent=2)
    print(f"\nResults saved to output/body_alignment_run_{ts}.json")


if __name__ == "__main__":
    main()

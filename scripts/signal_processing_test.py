"""
Quick test: Signal processing features (Savitzky-Golay + wavelets)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
import pywt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
TARGETS = ["angle", "depth", "left_right"]

def parse_array_string(s):
    if pd.isna(s):
        return np.full(240, np.nan, dtype=np.float32)
    s = s.replace("nan", "null")
    return np.array(json.loads(s), dtype=np.float32)

def load_raw_data():
    df = pd.read_csv(DATA_DIR / "train.csv")
    meta_cols = ["id", "shot_id", "participant_id"] + TARGETS
    keypoint_cols = [c for c in df.columns if c not in meta_cols]

    n = len(df)
    X = np.zeros((n, 240, len(keypoint_cols)), dtype=np.float32)
    for idx, row in df.iterrows():
        for i, col in enumerate(keypoint_cols):
            X[idx, :, i] = parse_array_string(row[col])

    y = df[TARGETS].values.astype(np.float32)
    pids = df["participant_id"].values
    return X, y, pids

def apply_savgol(X, window=11, poly=3):
    X_out = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            s = X[i, :, j]
            valid = ~np.isnan(s)
            if valid.sum() >= window:
                s_clean = np.interp(np.arange(len(s)), np.where(valid)[0], s[valid])
                X_out[i, :, j] = savgol_filter(s_clean, window, poly)
            else:
                X_out[i, :, j] = s
    return X_out

def extract_wavelet_features(X, wavelet='db4', level=3):
    n_samples, n_frames, n_features = X.shape
    # 4 levels = 4 coefficient arrays, 3 stats each = 12 per feature
    out = []
    for i in range(n_samples):
        row = []
        for j in range(n_features):
            s = X[i, :, j]
            valid = ~np.isnan(s)
            if valid.sum() < 10:
                row.extend([0] * (level + 1) * 3)
                continue
            s_clean = np.interp(np.arange(len(s)), np.where(valid)[0], s[valid])
            try:
                coeffs = pywt.wavedec(s_clean, wavelet, level=level)
                for c in coeffs:
                    row.extend([np.mean(c), np.std(c), np.sum(c**2)])
            except:
                row.extend([0] * (level + 1) * 3)
        out.append(row)
    return np.array(out, dtype=np.float32)

def run_cv(X, y, pids, name):
    gkf = GroupKFold(n_splits=5)
    preds = np.zeros_like(y)

    for fold, (tr, va) in enumerate(gkf.split(X, y, pids)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(np.nan_to_num(X[tr]))
        X_va = scaler.transform(np.nan_to_num(X[va]))

        for t in range(3):
            m = lgb.LGBMRegressor(n_estimators=100, num_leaves=10, learning_rate=0.05,
                                  reg_alpha=0.5, reg_lambda=0.5, verbose=-1, n_jobs=-1)
            m.fit(X_tr, y[tr, t])
            preds[va, t] = m.predict(X_va)

    print(f"\n{name} Results:")
    for i, t in enumerate(TARGETS):
        mse = np.mean((y[:, i] - preds[:, i])**2)
        print(f"  {t}: MSE = {mse:.6f}")
    total = np.mean((y - preds)**2)
    print(f"  TOTAL: MSE = {total:.6f}")
    return total

if __name__ == "__main__":
    print("Loading data...")
    X_raw, y, pids = load_raw_data()
    print(f"Shape: {X_raw.shape}")

    # Test 1: Raw mean/std features
    print("\n--- Test 1: Raw mean/std ---")
    X_simple = np.concatenate([X_raw.mean(axis=1), X_raw.std(axis=1)], axis=1)
    X_simple = np.nan_to_num(X_simple)
    run_cv(X_simple, y, pids, "Raw mean/std")

    # Test 2: Savitzky-Golay denoised
    print("\n--- Test 2: Savitzky-Golay ---")
    X_sg = apply_savgol(X_raw)
    X_sg_feat = np.concatenate([X_sg.mean(axis=1), X_sg.std(axis=1)], axis=1)
    X_sg_feat = np.nan_to_num(X_sg_feat)
    run_cv(X_sg_feat, y, pids, "Savitzky-Golay")

    # Test 3: Wavelet features
    print("\n--- Test 3: Wavelet features ---")
    X_wav = extract_wavelet_features(X_raw)
    X_wav = np.nan_to_num(X_wav)
    print(f"Wavelet features shape: {X_wav.shape}")
    run_cv(X_wav, y, pids, "Wavelet")

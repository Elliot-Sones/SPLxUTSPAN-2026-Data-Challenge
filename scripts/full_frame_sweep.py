"""
Full Frame Sweep + Greedy Multi-Frame Selection

Problems with original TARGET_FRAMES:
1. Only searched 100-180 (arbitrary constraint)
2. Same frame for all players
3. Optimized on old Ridge+LGB+XGB ensemble, not current pipeline
4. Never tested multi-frame combinations with Ridge

This script:
1. Sweeps ALL 240 frames (every 5th) per player per target using Ridge LOO
2. Greedy forward selection: find best single frame, then best pair, triple, etc.
3. Compares single-frame vs multi-frame Ridge LOO
4. Generates submission with best configuration

Uses Ridge LOO (no PLS) for the sweep to avoid PLS leakage.
Then uses the full pipeline (with PLS) only for final submission generation.
"""

from __future__ import annotations

import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"

HOOP = np.array([5.25, -25.0, 10.0])
N_FRAMES = 240
TARGETS = ["angle", "depth", "left_right"]


def parse_array_string(s):
    s = str(s).replace("nan", "NaN").replace("null", "NaN")
    arr = np.array(json.loads(s), dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    meta_cols = {"id", "shot_id", "participant_id", "angle", "depth", "left_right"}
    kp_cols = [c for c in df.columns if c not in meta_cols]
    n_kp = sum(1 for c in kp_cols if c.endswith("_x"))
    n = len(df)

    X_3d = np.zeros((n, N_FRAMES, n_kp, 3), dtype=np.float32)
    for idx, row in df.iterrows():
        for col_i, col in enumerate(kp_cols):
            X_3d[idx, :, col_i // 3, col_i % 3] = parse_array_string(row[col])
        if (idx + 1) % 100 == 0:
            print(f"  Loaded {idx+1}/{n}...")

    return X_3d, df


def extract_hoop_relative(X_3d, frame_idx):
    """Extract hoop-relative features at one frame. (N, n_kp*3)"""
    kp = X_3d[:, frame_idx, :, :]
    hoop_rel = kp - HOOP[np.newaxis, np.newaxis, :]
    return hoop_rel.reshape(len(X_3d), -1)


def extract_multi_frame(X_3d, frame_indices):
    """Extract and concatenate hoop-relative features from multiple frames."""
    parts = [extract_hoop_relative(X_3d, f) for f in frame_indices]
    return np.concatenate(parts, axis=1)


def ridge_loo_mse_per_player(features, player_ids, target_values,
                              bw_quantile=0.3, alpha=10.0):
    """
    Per-player locally weighted Ridge LOO.
    No PLS (avoid leakage). Returns per-player MSE dict and global MSE.
    """
    n = len(target_values)
    predictions = np.full(n, np.nan)
    unique_players = np.unique(player_ids)
    player_mses = {}

    for pid in unique_players:
        mask = player_ids == pid
        idx = np.where(mask)[0]
        X_p = features[mask]
        y_p = target_values[mask]
        n_p = len(y_p)

        if n_p < 5:
            predictions[idx] = np.mean(y_p)
            player_mses[pid] = np.mean((predictions[idx] - y_p) ** 2)
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_p)

        D = pairwise_distances(X_scaled)
        dists_flat = D[np.triu_indices(n_p, k=1)]
        bw = np.quantile(dists_flat, bw_quantile) if len(dists_flat) > 0 else 1.0
        if bw < 1e-10:
            bw = 1.0

        preds = np.zeros(n_p)
        for i in range(n_p):
            weights = np.exp(-0.5 * (D[i] / bw) ** 2)
            weights[i] = 0  # LOO

            if weights.sum() < 1e-10:
                preds[i] = np.mean(np.delete(y_p, i))
                continue

            # Weighted Ridge
            sqrt_w = np.sqrt(weights)
            X_w = X_scaled * sqrt_w[:, np.newaxis]
            y_w = y_p * sqrt_w

            # Remove held-out
            X_train = np.delete(X_w, i, axis=0)
            y_train = np.delete(y_w, i, axis=0)

            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(X_train, y_train)
            preds[i] = ridge.predict(X_scaled[i:i+1])[0]

        predictions[idx] = preds
        player_mses[pid] = np.mean((preds - y_p) ** 2)

    global_mse = np.mean((predictions - target_values) ** 2)
    return global_mse, player_mses


def main():
    print("=" * 70)
    print("FULL FRAME SWEEP + GREEDY MULTI-FRAME SELECTION")
    print("=" * 70)

    print("\nLoading train data...")
    X_3d, df = load_data(DATA_DIR / "train.csv")
    players = df["participant_id"].values
    unique_players = sorted(np.unique(players))

    # Scale targets
    import joblib
    targets_scaled = {}
    for t in TARGETS:
        scaler_path = DATA_DIR / f"scaler_{t}.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            targets_scaled[t] = scaler.transform(df[t].values.reshape(-1, 1)).ravel()
        else:
            vals = df[t].values
            targets_scaled[t] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)

    # ============================================================
    # PHASE 1: Full sweep - every 5th frame, 0-239
    # ============================================================
    frame_step = 5
    frames = list(range(0, N_FRAMES, frame_step))
    n_frames = len(frames)

    print(f"\nPHASE 1: Sweeping {n_frames} frames (0-{N_FRAMES-1}, step {frame_step})")
    print("Per-player Ridge LOO (no PLS)\n")

    # Store: results[target][frame_idx] = (global_mse, {pid: mse})
    results = {t: {} for t in TARGETS}

    for fi, frame_idx in enumerate(frames):
        if fi % 10 == 0:
            print(f"  Frame {frame_idx:3d} ({fi+1}/{n_frames})...")

        feats = extract_hoop_relative(X_3d, frame_idx)

        for tname in TARGETS:
            y = targets_scaled[tname]
            global_mse, player_mses = ridge_loo_mse_per_player(feats, players, y)
            results[tname][frame_idx] = (global_mse, player_mses)

    # Print results
    print("\n" + "=" * 70)
    print("PHASE 1 RESULTS: Single-frame Ridge LOO")
    print("=" * 70)

    best_single_frames = {}  # {target: {pid: best_frame}}

    for tname in TARGETS:
        print(f"\n{'='*50}")
        print(f"TARGET: {tname}")
        print(f"{'='*50}")

        # Global best
        global_mses = {f: results[tname][f][0] for f in frames}
        sorted_frames = sorted(global_mses, key=global_mses.get)
        best_f = sorted_frames[0]
        best_mse = global_mses[best_f]

        print(f"\n  GLOBAL top 10:")
        for rank, f in enumerate(sorted_frames[:10]):
            gap = (global_mses[f] - best_mse) / best_mse * 100
            print(f"    #{rank+1}: frame {f:3d} MSE={global_mses[f]:.6f} (+{gap:.1f}%)")

        # Disjoint global peaks
        used = set()
        disjoint = []
        for f in sorted_frames:
            if any(abs(f - u) < 25 for u in used):
                continue
            used.add(f)
            disjoint.append(f)
            if len(disjoint) >= 6:
                break
        print(f"\n  GLOBAL disjoint peaks: {disjoint}")

        # Per-player best
        print(f"\n  PER-PLAYER best frames:")
        best_single_frames[tname] = {}
        for pid in unique_players:
            player_mses = {f: results[tname][f][1].get(pid, np.inf) for f in frames}
            sorted_p = sorted(player_mses, key=player_mses.get)
            best_p_f = sorted_p[0]
            best_p_mse = player_mses[best_p_f]
            best_single_frames[tname][pid] = best_p_f

            # Disjoint peaks for this player
            used_p = set()
            disjoint_p = []
            for f in sorted_p:
                if any(abs(f - u) < 25 for u in used_p):
                    continue
                used_p.add(f)
                disjoint_p.append((f, player_mses[f]))
                if len(disjoint_p) >= 4:
                    break

            peaks_str = " | ".join([
                f"f{f}({m:.6f})" for f, m in disjoint_p
            ])
            n_shots = (players == pid).sum()
            print(f"    P{pid} ({n_shots} shots): {peaks_str}")

    # ============================================================
    # PHASE 2: Greedy multi-frame selection per player per target
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Greedy forward selection of frame combinations")
    print("Per-player Ridge LOO")
    print("=" * 70)

    # For each player x target:
    # 1. Start with the best single frame
    # 2. Try adding each other frame, keep the one that improves most
    # 3. Repeat until no improvement

    max_frames_to_combine = 4
    best_combos = {}  # {target: {pid: (frames_list, mse)}}

    for tname in TARGETS:
        print(f"\n{'='*50}")
        print(f"TARGET: {tname}")
        print(f"{'='*50}")

        y = targets_scaled[tname]
        best_combos[tname] = {}

        for pid in unique_players:
            mask = players == pid
            n_shots = mask.sum()
            y_p = y[mask]

            # Get the single best frame for this player
            player_mses_single = {f: results[tname][f][1].get(pid, np.inf) for f in frames}
            best_f = min(player_mses_single, key=player_mses_single.get)
            best_mse = player_mses_single[best_f]

            selected = [best_f]
            current_mse = best_mse

            print(f"\n  P{pid} ({n_shots} shots): starting with f{best_f} (MSE={best_mse:.6f})")

            for step in range(1, max_frames_to_combine):
                # Try adding each unselected frame
                best_addition = None
                best_new_mse = current_mse

                # Only try frames that aren't too close to already selected
                candidates = [f for f in frames
                             if f not in selected and
                             all(abs(f - s) >= 15 for s in selected)]

                for f_candidate in candidates:
                    trial_frames = selected + [f_candidate]
                    trial_feats = extract_multi_frame(X_3d[mask], trial_frames)

                    # Quick Ridge LOO for this player only
                    mse_trial = _player_ridge_loo(trial_feats, y_p, bw_quantile=0.3, alpha=10.0)

                    if mse_trial < best_new_mse:
                        best_new_mse = mse_trial
                        best_addition = f_candidate

                if best_addition is not None and best_new_mse < current_mse * 0.995:
                    # At least 0.5% improvement to add a frame
                    improvement = (current_mse - best_new_mse) / current_mse * 100
                    selected.append(best_addition)
                    current_mse = best_new_mse
                    print(f"    +f{best_addition}: MSE={current_mse:.6f} ({improvement:+.1f}% improvement)")
                else:
                    print(f"    No further improvement found at step {step+1}")
                    break

            final_pct = (current_mse - best_mse) / best_mse * 100
            print(f"    FINAL: frames={selected}, MSE={current_mse:.6f} ({final_pct:+.1f}% vs single)")
            best_combos[tname][pid] = (selected, current_mse)

    # ============================================================
    # PHASE 3: Compare old vs new frame selection
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Old vs New comparison")
    print("=" * 70)

    OLD_FRAMES = {"angle": 153, "depth": 150, "left_right": 170}

    for tname in TARGETS:
        y = targets_scaled[tname]
        old_frame = OLD_FRAMES[tname]

        # Old: same frame for all players
        old_feats = extract_hoop_relative(X_3d, old_frame)
        old_mse, old_player_mses = ridge_loo_mse_per_player(old_feats, players, y)

        # New single-frame: per-player optimal
        new_single_preds = np.full(len(y), np.nan)
        for pid in unique_players:
            mask = players == pid
            best_f = best_single_frames[tname][pid]
            feats = extract_hoop_relative(X_3d[mask], best_f)
            _, pmses = ridge_loo_mse_per_player(feats, players[mask], y[mask])

        # New multi-frame: per-player greedy combos
        new_multi_preds = np.full(len(y), np.nan)

        print(f"\n{tname}:")
        print(f"  Old fixed frame {old_frame}: global MSE = {old_mse:.6f}")

        # Per-player comparison
        total_old = 0
        total_new_single = 0
        total_new_multi = 0
        total_n = 0

        for pid in unique_players:
            mask = players == pid
            idx = np.where(mask)[0]
            n_p = mask.sum()
            y_p = y[mask]

            # Old
            old_f_mse = old_player_mses.get(pid, np.inf)

            # New single
            new_f = best_single_frames[tname][pid]
            new_f_mse = results[tname][new_f][1].get(pid, np.inf)

            # New multi
            multi_frames, multi_mse = best_combos[tname][pid]

            improvement_single = (old_f_mse - new_f_mse) / old_f_mse * 100
            improvement_multi = (old_f_mse - multi_mse) / old_f_mse * 100

            total_old += old_f_mse * n_p
            total_new_single += new_f_mse * n_p
            total_new_multi += multi_mse * n_p
            total_n += n_p

            print(f"  P{pid}: old=f{old_frame}({old_f_mse:.6f}) -> "
                  f"new_single=f{new_f}({new_f_mse:.6f}, {improvement_single:+.1f}%) -> "
                  f"multi={multi_frames}({multi_mse:.6f}, {improvement_multi:+.1f}%)")

        wavg_old = total_old / total_n
        wavg_single = total_new_single / total_n
        wavg_multi = total_new_multi / total_n
        print(f"  WEIGHTED AVG: old={wavg_old:.6f} -> single={wavg_single:.6f} "
              f"({(wavg_old-wavg_single)/wavg_old*100:+.1f}%) -> multi={wavg_multi:.6f} "
              f"({(wavg_old-wavg_multi)/wavg_old*100:+.1f}%)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF BEST FRAME CONFIGURATIONS")
    print("=" * 70)

    for tname in TARGETS:
        print(f"\n{tname}:")
        for pid in unique_players:
            single_f = best_single_frames[tname][pid]
            multi_frames, multi_mse = best_combos[tname][pid]
            single_mse = results[tname][single_f][1].get(pid, np.inf)
            print(f"  P{pid}: single=f{single_f}({single_mse:.6f}) | multi={multi_frames}({multi_mse:.6f})")


def _player_ridge_loo(features, target_values, bw_quantile=0.3, alpha=10.0):
    """Ridge LOO for a single player's data."""
    n = len(target_values)
    if n < 5:
        return np.mean((target_values - np.mean(target_values)) ** 2)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    D = pairwise_distances(X_scaled)
    dists_flat = D[np.triu_indices(n, k=1)]
    bw = np.quantile(dists_flat, bw_quantile) if len(dists_flat) > 0 else 1.0
    if bw < 1e-10:
        bw = 1.0

    preds = np.zeros(n)
    for i in range(n):
        weights = np.exp(-0.5 * (D[i] / bw) ** 2)
        weights[i] = 0

        if weights.sum() < 1e-10:
            preds[i] = np.mean(np.delete(target_values, i))
            continue

        sqrt_w = np.sqrt(weights)
        X_w = X_scaled * sqrt_w[:, np.newaxis]
        y_w = target_values * sqrt_w

        X_train = np.delete(X_w, i, axis=0)
        y_train = np.delete(y_w, i, axis=0)

        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(X_train, y_train)
        preds[i] = ridge.predict(X_scaled[i:i+1])[0]

    return np.mean((preds - target_values) ** 2)


if __name__ == "__main__":
    main()

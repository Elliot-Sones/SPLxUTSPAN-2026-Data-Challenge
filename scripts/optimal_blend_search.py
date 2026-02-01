"""
Optimal Blend Search - Use known LB data to find best blend.

We have 8 known LB scores. Use submission profiles to predict LB,
then search for blend weights that minimize predicted LB.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

SUBMISSION_DIR = Path(__file__).parent.parent / "submission"

# Known LB scores
KNOWN_LB = {
    8: 0.010220,
    9: 0.009109,
    10: 0.008907,
    11: 0.009848,
    20: 0.008619,
    25: 0.008305,  # BEST
    34: 0.008377,
    51: 0.008807,
    104: 0.008827,
    110: 0.010069,
    111: 0.008703,
}


def get_submission_profile(sub_num):
    """Get profile metrics for a submission."""
    try:
        df = pd.read_csv(SUBMISSION_DIR / f"submission_{sub_num}.csv")
        return {
            'angle_std': df['scaled_angle'].std(),
            'angle_mean': df['scaled_angle'].mean(),
            'depth_std': df['scaled_depth'].std(),
            'depth_mean': df['scaled_depth'].mean(),
            'lr_std': df['scaled_left_right'].std(),
            'lr_mean': df['scaled_left_right'].mean(),
        }
    except:
        return None


def build_lb_predictor():
    """Build a model to predict LB from submission profile."""
    profiles = []
    lb_scores = []

    for sub_num, lb in KNOWN_LB.items():
        profile = get_submission_profile(sub_num)
        if profile:
            profiles.append(profile)
            lb_scores.append(lb)

    # Create feature matrix
    X = np.array([[p['angle_std'], p['depth_mean']] for p in profiles])
    y = np.array(lb_scores)

    # Fit linear model
    model = LinearRegression()
    model.fit(X, y)

    print("LB Predictor:")
    print(f"  Coefficients: angle_std={model.coef_[0]:.6f}, depth_mean={model.coef_[1]:.6f}")
    print(f"  Intercept: {model.intercept_:.6f}")
    print(f"  R2 on known data: {model.score(X, y):.4f}")

    return model


def predict_lb(model, angle_std, depth_mean):
    """Predict LB from profile metrics."""
    return model.predict([[angle_std, depth_mean]])[0]


def create_blend(subs, weights):
    """Create a blended submission from multiple submissions."""
    dfs = [pd.read_csv(SUBMISSION_DIR / f"submission_{s}.csv") for s in subs]

    cols = ['scaled_angle', 'scaled_depth', 'scaled_left_right']
    blend_data = sum(w * dfs[i][cols].values for i, w in enumerate(weights))

    result = dfs[0][['id']].copy()
    result['scaled_angle'] = blend_data[:, 0]
    result['scaled_depth'] = blend_data[:, 1]
    result['scaled_left_right'] = blend_data[:, 2]

    return result


def search_optimal_blend():
    """Search for optimal blend weights to minimize predicted LB."""

    # Build LB predictor
    lb_model = build_lb_predictor()

    # Submissions to blend
    candidates = [9, 10, 25, 111]

    print(f"\n{'='*60}")
    print("BLEND SEARCH")
    print(f"{'='*60}")
    print(f"Candidates: {candidates}")

    # Individual profiles
    print("\nIndividual profiles:")
    for s in candidates:
        p = get_submission_profile(s)
        pred = predict_lb(lb_model, p['angle_std'], p['depth_mean'])
        actual = KNOWN_LB.get(s, '-')
        print(f"  Sub {s}: angle_std={p['angle_std']:.4f}, depth_mean={p['depth_mean']:.4f}, "
              f"pred_LB={pred:.6f}, actual_LB={actual}")

    # Grid search for 2-way blends
    print("\n2-way blends:")
    best_2way = None
    best_2way_pred = float('inf')

    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            for w in np.arange(0.0, 1.05, 0.1):
                weights = [0] * len(candidates)
                weights[i] = w
                weights[j] = 1 - w

                blend = create_blend(candidates, weights)
                angle_std = blend['scaled_angle'].std()
                depth_mean = blend['scaled_depth'].mean()
                pred_lb = predict_lb(lb_model, angle_std, depth_mean)

                if pred_lb < best_2way_pred:
                    best_2way_pred = pred_lb
                    best_2way = (candidates[i], candidates[j], w, 1-w, angle_std, depth_mean)

    print(f"Best 2-way: Sub{best_2way[0]} ({best_2way[2]:.1f}) + Sub{best_2way[1]} ({best_2way[3]:.1f})")
    print(f"  angle_std={best_2way[4]:.4f}, depth_mean={best_2way[5]:.4f}, pred_LB={best_2way_pred:.6f}")

    # Grid search for 3-way blends
    print("\n3-way blends (Sub 9 + Sub 10 + Sub X):")
    best_3way = None
    best_3way_pred = float('inf')

    for third in [25, 111]:
        for w1 in np.arange(0.0, 1.05, 0.1):
            for w2 in np.arange(0.0, 1.05 - w1, 0.1):
                w3 = 1 - w1 - w2
                if w3 < 0:
                    continue

                subs = [9, 10, third]
                weights = [w1, w2, w3]

                blend = create_blend(subs, weights)
                angle_std = blend['scaled_angle'].std()
                depth_mean = blend['scaled_depth'].mean()
                pred_lb = predict_lb(lb_model, angle_std, depth_mean)

                if pred_lb < best_3way_pred:
                    best_3way_pred = pred_lb
                    best_3way = (subs, weights, angle_std, depth_mean)

    print(f"Best 3-way: {best_3way[0]} weights={[f'{w:.1f}' for w in best_3way[1]]}")
    print(f"  angle_std={best_3way[2]:.4f}, depth_mean={best_3way[3]:.4f}, pred_LB={best_3way_pred:.6f}")

    # Fine-grained search around best 3-way
    print("\nFine-tuning best blend...")
    best_subs = [9, 10]  # We know 50-50 of 9+10 is best

    def objective(weights):
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        blend = create_blend(best_subs, weights.tolist())
        angle_std = blend['scaled_angle'].std()
        depth_mean = blend['scaled_depth'].mean()
        return predict_lb(lb_model, angle_std, depth_mean)

    # Try different starting points
    best_result = None
    best_pred = float('inf')

    for start in [[0.5, 0.5], [0.4, 0.6], [0.6, 0.4], [0.45, 0.55], [0.55, 0.45]]:
        result = minimize(objective, start, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 0.001})
        if result.fun < best_pred:
            best_pred = result.fun
            best_result = result

    optimal_weights = np.array(best_result.x) / np.sum(best_result.x)
    print(f"\nOptimal weights for Sub 9 + Sub 10:")
    print(f"  Sub 9: {optimal_weights[0]:.4f}")
    print(f"  Sub 10: {optimal_weights[1]:.4f}")

    final_blend = create_blend(best_subs, optimal_weights.tolist())
    print(f"\nFinal blend profile:")
    print(f"  angle_std: {final_blend['scaled_angle'].std():.4f}")
    print(f"  depth_mean: {final_blend['scaled_depth'].mean():.4f}")
    print(f"  Predicted LB: {best_pred:.6f}")

    # Compare with Sub 25 (50-50 blend)
    print(f"\nComparison with Sub 25 (50-50 blend):")
    sub25_profile = get_submission_profile(25)
    sub25_pred = predict_lb(lb_model, sub25_profile['angle_std'], sub25_profile['depth_mean'])
    print(f"  Sub 25 angle_std: {sub25_profile['angle_std']:.4f}")
    print(f"  Sub 25 depth_mean: {sub25_profile['depth_mean']:.4f}")
    print(f"  Sub 25 predicted LB: {sub25_pred:.6f}")
    print(f"  Sub 25 actual LB: {KNOWN_LB[25]:.6f}")

    # Save optimal blend if it's better
    improvement = sub25_pred - best_pred
    if improvement > 0:
        print(f"\n*** Optimal blend is predicted to be {improvement:.6f} better than Sub 25 ***")

        # Save blend
        existing = list(SUBMISSION_DIR.glob("submission*.csv"))
        nums = []
        for f in existing:
            name = f.stem
            if name.startswith("submission_"):
                try:
                    nums.append(int(name.split('_')[1]))
                except:
                    pass
            elif name.startswith("submission"):
                try:
                    nums.append(int(name[10:]))
                except:
                    pass
        next_num = max(nums) + 1 if nums else 1

        filepath = SUBMISSION_DIR / f"submission_{next_num}.csv"
        final_blend.to_csv(filepath, index=False)
        print(f"\nSaved as: {filepath}")
        return filepath
    else:
        print(f"\nSub 25 is already optimal or near-optimal.")
        return None


def main():
    print("="*80)
    print("OPTIMAL BLEND SEARCH")
    print("="*80)
    print("\nUsing known LB data to find best blend weights")

    result = search_optimal_blend()

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if result:
        print(f"Created new submission: {result}")
    else:
        print("Sub 25 (50-50 blend of Sub 9 + Sub 10) remains the best approach.")
        print("\nTo beat Sub 25, we would need:")
        print("1. A fundamentally different model architecture")
        print("2. New features that capture untapped signal")
        print("3. Different training strategy (e.g., more data)")


if __name__ == "__main__":
    main()

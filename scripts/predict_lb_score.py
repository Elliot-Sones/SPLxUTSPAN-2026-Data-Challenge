"""
Predict LB Score from Submission Statistics

Use known submissions to model the CV-to-LB relationship and predict
what new submissions will score on LB.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from scipy import stats as sp_stats

PROJECT_DIR = Path(__file__).parent.parent
SUBMISSION_DIR = PROJECT_DIR / 'submission'

# Known LB scores from research
KNOWN_SUBMISSIONS = {
    8: {'lb': 0.010220},
    9: {'lb': 0.009109},
    10: {'lb': 0.008907},
    11: {'lb': 0.009848},
    20: {'lb': 0.008619},
    25: {'lb': 0.008305},  # Best
    34: {'lb': 0.008377},
    51: {'lb': 0.008807},
}


def load_submission_stats(sub_num):
    """Load a submission and compute statistics."""
    # Try different naming patterns
    patterns = [
        f"submission{sub_num}.csv",
        f"submission_{sub_num}.csv",
    ]

    sub_path = None
    for pattern in patterns:
        path = SUBMISSION_DIR / pattern
        if path.exists():
            sub_path = path
            break

    if sub_path is None:
        return None

    df = pd.read_csv(sub_path)

    # Normalize column names
    col_map = {}
    for col in df.columns:
        if 'angle' in col.lower():
            col_map[col] = 'angle'
        elif 'depth' in col.lower():
            col_map[col] = 'depth'
        elif 'left' in col.lower() or 'right' in col.lower():
            col_map[col] = 'left_right'

    df = df.rename(columns=col_map)

    stats = {'sub_num': sub_num}

    for col in ['angle', 'depth', 'left_right']:
        if col in df.columns:
            stats[f'{col}_mean'] = df[col].mean()
            stats[f'{col}_std'] = df[col].std()
            stats[f'{col}_min'] = df[col].min()
            stats[f'{col}_max'] = df[col].max()
            stats[f'{col}_range'] = df[col].max() - df[col].min()
            stats[f'{col}_median'] = df[col].median()
            stats[f'{col}_skew'] = df[col].skew()

    return stats


def main():
    print("=" * 70)
    print("PREDICTING LB SCORE FROM SUBMISSION STATISTICS")
    print("=" * 70)

    # Load all known submissions
    all_stats = []

    for sub_num, info in KNOWN_SUBMISSIONS.items():
        stats = load_submission_stats(sub_num)
        if stats:
            stats['lb_score'] = info['lb']
            all_stats.append(stats)
            print(f"Loaded sub{sub_num}: LB={info['lb']:.6f}")

    if len(all_stats) < 3:
        print("Not enough submissions with known LB scores")
        return

    df = pd.DataFrame(all_stats)

    print(f"\nLoaded {len(df)} submissions with known LB scores")

    # Analyze correlations
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS: What predicts LB score?")
    print("=" * 70)

    feature_cols = [c for c in df.columns if c not in ['sub_num', 'lb_score']]

    correlations = []
    for col in feature_cols:
        if df[col].std() > 0:
            corr, pval = sp_stats.pearsonr(df[col], df['lb_score'])
            correlations.append({
                'feature': col,
                'correlation': corr,
                'p_value': pval,
                'abs_corr': abs(corr)
            })

    corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)

    print("\nTop correlations with LB score:")
    print(f"{'Feature':<20} {'Correlation':>12} {'p-value':>10} {'Direction':<15}")
    print("-" * 60)

    for _, row in corr_df.head(10).iterrows():
        direction = "higher=worse" if row['correlation'] > 0 else "higher=better"
        sig = "*" if row['p_value'] < 0.1 else ""
        print(f"{row['feature']:<20} {row['correlation']:>+12.4f} {row['p_value']:>10.4f} {direction:<15} {sig}")

    # Build prediction model
    print("\n" + "=" * 70)
    print("BUILDING LB PREDICTION MODEL")
    print("=" * 70)

    # Use top correlated features
    top_features = corr_df[corr_df['abs_corr'] > 0.3]['feature'].tolist()[:5]

    if len(top_features) < 1:
        top_features = ['angle_std']  # Fallback

    print(f"\nUsing features: {top_features}")

    X = df[top_features].values
    y = df['lb_score'].values

    # Simple linear model
    model = Ridge(alpha=0.1)
    model.fit(X, y)

    # Leave-one-out CV for the model
    predictions = []
    for i in range(len(df)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]

        m = Ridge(alpha=0.1)
        m.fit(X_train, y_train)
        pred = m.predict(X_test)[0]
        predictions.append(pred)

    pred_error = np.mean(np.abs(np.array(predictions) - y))
    print(f"Model LOO-CV mean absolute error: {pred_error:.6f}")

    # Show predictions vs actual
    print("\nModel predictions vs actual:")
    for i, row in df.iterrows():
        print(f"  Sub{int(row['sub_num'])}: Actual={row['lb_score']:.6f}, Predicted={predictions[i]:.6f}, Error={abs(predictions[i]-row['lb_score']):.6f}")

    # Now predict for new submissions
    print("\n" + "=" * 70)
    print("PREDICTIONS FOR NEW SUBMISSIONS")
    print("=" * 70)

    new_subs = [85, 86]

    for sub_num in new_subs:
        stats = load_submission_stats(sub_num)
        if stats is None:
            print(f"\nSub{sub_num}: File not found")
            continue

        # Get features
        X_new = np.array([[stats.get(f, 0) for f in top_features]])

        # Predict
        pred_lb = model.predict(X_new)[0]

        print(f"\nSubmission {sub_num}:")
        print(f"  Features used:")
        for f in top_features:
            print(f"    {f}: {stats.get(f, 'N/A'):.4f}")

        print(f"  Predicted LB Score: {pred_lb:.6f}")
        print(f"  Prediction interval: [{pred_lb - pred_error:.6f}, {pred_lb + pred_error:.6f}]")

        # Compare to sub9
        if pred_lb < 0.009109:
            print(f"  vs Sub9 (0.009109): BETTER by {(0.009109 - pred_lb)/0.009109*100:.1f}%")
        else:
            print(f"  vs Sub9 (0.009109): WORSE by {(pred_lb - 0.009109)/0.009109*100:.1f}%")

        # Compare to best (sub25)
        if pred_lb < 0.008305:
            print(f"  vs Sub25 (0.008305): BETTER by {(0.008305 - pred_lb)/0.008305*100:.1f}%")
        else:
            print(f"  vs Sub25 (0.008305): WORSE by {(pred_lb - 0.008305)/0.008305*100:.1f}%")

    # Analysis
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    # What makes a good LB score?
    best_sub = df.loc[df['lb_score'].idxmin()]
    worst_sub = df.loc[df['lb_score'].idxmax()]

    print(f"\nBest submission (Sub{int(best_sub['sub_num'])}): LB={best_sub['lb_score']:.6f}")
    print(f"Worst submission (Sub{int(worst_sub['sub_num'])}): LB={worst_sub['lb_score']:.6f}")

    print("\nKey differences:")
    for f in top_features:
        print(f"  {f}: Best={best_sub[f]:.4f}, Worst={worst_sub[f]:.4f}")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    sub85_stats = load_submission_stats(85)
    sub86_stats = load_submission_stats(86)

    if sub85_stats and sub86_stats:
        pred85 = model.predict(np.array([[sub85_stats.get(f, 0) for f in top_features]]))[0]
        pred86 = model.predict(np.array([[sub86_stats.get(f, 0) for f in top_features]]))[0]

        print(f"\n  Sub85 predicted LB: {pred85:.6f}")
        print(f"  Sub86 predicted LB: {pred86:.6f}")

        if pred86 < pred85:
            print(f"\n  RECOMMENDATION: Submit Sub86 first")
            print(f"  Reason: Lower predicted LB score ({pred86:.6f} vs {pred85:.6f})")
        else:
            print(f"\n  RECOMMENDATION: Submit Sub85 first")
            print(f"  Reason: Lower predicted LB score ({pred85:.6f} vs {pred86:.6f})")


if __name__ == "__main__":
    main()

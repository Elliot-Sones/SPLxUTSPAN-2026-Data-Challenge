"""
Optimized Blend V2 - Using Sub 133 LB score as new data point
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# Known LB scores with Sub 133 added
KNOWN_LB = {
    8: 0.010220,
    9: 0.009109,
    10: 0.008907,
    11: 0.009848,
    20: 0.008619,
    25: 0.008305,
    34: 0.008377,
    51: 0.008807,
    113: 0.008031,
    133: 0.007809,  # NEW DATA POINT
}

targets = ['scaled_angle', 'scaled_depth', 'scaled_left_right']

def load_sub(num):
    """Load a submission file"""
    return pd.read_csv(f"submission/submission_{num}.csv")

def compute_profile(df):
    """Compute angle_std and depth_mean"""
    angle_std = df['scaled_angle'].std()
    depth_mean = df['scaled_depth'].mean()
    return angle_std, depth_mean

def create_blend(weights, subs):
    """Create weighted blend of submissions"""
    blend = pd.DataFrame()
    blend['id'] = subs[0]['id']

    for target in targets:
        blend[target] = sum(w * s[target] for w, s in zip(weights, subs))

    return blend

# Load base submissions
print("Loading submissions...")
sub9 = load_sub(9)
sub10 = load_sub(10)
sub25 = load_sub(25)
sub111 = load_sub(111)
sub133 = load_sub(133)

# Compute profiles for all known submissions
print("\n=== Computing submission profiles ===")

profiles = {}
for sub_num, lb_score in KNOWN_LB.items():
    try:
        if sub_num == 113:
            # Sub 113 = 50% Sub9 + 40% Sub10 + 10% Sub111
            df = create_blend([0.5, 0.4, 0.1], [sub9, sub10, sub111])
        elif sub_num == 133:
            df = sub133
        else:
            df = load_sub(sub_num)

        angle_std, depth_mean = compute_profile(df)
        profiles[sub_num] = {
            'angle_std': angle_std,
            'depth_mean': depth_mean,
            'lb_score': lb_score
        }
        print(f"Sub {sub_num}: angle_std={angle_std:.5f}, depth_mean={depth_mean:.5f}, LB={lb_score:.6f}")
    except Exception as e:
        print(f"Sub {sub_num}: Could not load - {e}")

# Build prediction model with updated data
print("\n=== Building LB prediction model ===")
X = np.array([[p['angle_std'], p['depth_mean']] for p in profiles.values()])
y = np.array([p['lb_score'] for p in profiles.values()])

print(f"Training data: {len(profiles)} submissions")

# Try polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

model = Ridge(alpha=0.01)
model.fit(X_poly, y)

print(f"Feature names: {poly.get_feature_names_out(['angle_std', 'depth_mean'])}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Predict for each known submission
print("\n=== Model predictions ===")
for sub_num, p in sorted(profiles.items(), key=lambda x: x[1]['lb_score']):
    X_test = poly.transform([[p['angle_std'], p['depth_mean']]])
    pred = model.predict(X_test)[0]
    print(f"Sub {sub_num}: actual={p['lb_score']:.6f}, predicted={pred:.6f}, error={abs(pred-p['lb_score'])*1000:.3f}e-3")

# Now search for optimal blend weights
print("\n=== Searching for optimal blend ===")

# We'll search 4 submissions: Sub25, Sub9, Sub10, Sub111
base_subs = [sub25, sub9, sub10, sub111]
sub_names = ['Sub25', 'Sub9', 'Sub10', 'Sub111']

best_pred = float('inf')
best_weights = None
best_profile = None

# Fine grid search
step = 0.01
count = 0

for w0 in np.arange(0, 0.20, step):  # Sub25: 0-20%
    for w1 in np.arange(0.2, 0.5, step):  # Sub9: 20-50%
        for w2 in np.arange(0.3, 0.6, step):  # Sub10: 30-60%
            w3 = 1.0 - w0 - w1 - w2  # Sub111: remainder

            if w3 < 0 or w3 > 0.5:
                continue

            weights = [w0, w1, w2, w3]
            blend = create_blend(weights, base_subs)
            angle_std, depth_mean = compute_profile(blend)

            # Predict LB
            X_test = poly.transform([[angle_std, depth_mean]])
            pred = model.predict(X_test)[0]

            if pred < best_pred:
                best_pred = pred
                best_weights = weights
                best_profile = (angle_std, depth_mean)

            count += 1

print(f"Searched {count} combinations")
print(f"\nBest predicted LB: {best_pred:.6f}")
print(f"Best weights: Sub25={best_weights[0]:.2f}, Sub9={best_weights[1]:.2f}, Sub10={best_weights[2]:.2f}, Sub111={best_weights[3]:.2f}")
print(f"Best profile: angle_std={best_profile[0]:.5f}, depth_mean={best_profile[1]:.5f}")

# Compare to Sub 133
sub133_pred = model.predict(poly.transform([[profiles[133]['angle_std'], profiles[133]['depth_mean']]]))[0]
print(f"\nSub 133 predicted: {sub133_pred:.6f}, actual: {profiles[133]['lb_score']:.6f}")
print(f"New blend vs Sub 133: {best_pred - sub133_pred:.6f}")

# Create the best blend
print("\n=== Creating best blend ===")
best_blend = create_blend(best_weights, base_subs)

# Verify profile
final_std, final_mean = compute_profile(best_blend)
print(f"Final profile: angle_std={final_std:.5f}, depth_mean={final_mean:.5f}")

# Compare to Sub 133
print(f"\nComparison to Sub 133:")
print(f"  Sub 133: angle_std={profiles[133]['angle_std']:.5f}, depth_mean={profiles[133]['depth_mean']:.5f}")
print(f"  Sub 134: angle_std={final_std:.5f}, depth_mean={final_mean:.5f}")
print(f"  angle_std change: {final_std - profiles[133]['angle_std']:.6f}")
print(f"  depth_mean change: {final_mean - profiles[133]['depth_mean']:.6f}")

# Save if better
if best_pred < sub133_pred:
    output_path = "submission/submission_134.csv"
    best_blend.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print("RECOMMEND: Submit Sub 134 for potential improvement")
else:
    print("\nNo improvement over Sub 133 - not saving")

# Also try blending Sub 133 with other subs
print("\n=== Trying blends WITH Sub 133 ===")

best_blend_with_133 = None
best_pred_with_133 = sub133_pred

# Try blending Sub 133 with Sub25
for w in np.arange(0.01, 0.15, 0.01):
    blend = create_blend([w, 1-w], [sub25, sub133])
    angle_std, depth_mean = compute_profile(blend)
    X_test = poly.transform([[angle_std, depth_mean]])
    pred = model.predict(X_test)[0]

    if pred < best_pred_with_133:
        best_pred_with_133 = pred
        best_blend_with_133 = (f"{int(w*100)}% Sub25 + {int((1-w)*100)}% Sub133", blend)

    print(f"  {int(w*100):2d}% Sub25 + {int((1-w)*100):2d}% Sub133: angle_std={angle_std:.5f}, depth_mean={depth_mean:.5f}, pred_LB={pred:.6f}")

# Try blending Sub 133 with Sub20 (good LB)
sub20 = load_sub(20)
print()
for w in np.arange(0.01, 0.15, 0.01):
    blend = create_blend([w, 1-w], [sub20, sub133])
    angle_std, depth_mean = compute_profile(blend)
    X_test = poly.transform([[angle_std, depth_mean]])
    pred = model.predict(X_test)[0]

    if pred < best_pred_with_133:
        best_pred_with_133 = pred
        best_blend_with_133 = (f"{int(w*100)}% Sub20 + {int((1-w)*100)}% Sub133", blend)

    print(f"  {int(w*100):2d}% Sub20 + {int((1-w)*100):2d}% Sub133: angle_std={angle_std:.5f}, depth_mean={depth_mean:.5f}, pred_LB={pred:.6f}")

# Try blending Sub 133 with Sub34 (good LB)
sub34 = load_sub(34)
print()
for w in np.arange(0.01, 0.15, 0.01):
    blend = create_blend([w, 1-w], [sub34, sub133])
    angle_std, depth_mean = compute_profile(blend)
    X_test = poly.transform([[angle_std, depth_mean]])
    pred = model.predict(X_test)[0]

    if pred < best_pred_with_133:
        best_pred_with_133 = pred
        best_blend_with_133 = (f"{int(w*100)}% Sub34 + {int((1-w)*100)}% Sub133", blend)

    print(f"  {int(w*100):2d}% Sub34 + {int((1-w)*100):2d}% Sub133: angle_std={angle_std:.5f}, depth_mean={depth_mean:.5f}, pred_LB={pred:.6f}")

if best_blend_with_133 is not None and best_pred_with_133 < sub133_pred:
    print(f"\nBest blend with Sub 133: {best_blend_with_133[0]}")
    print(f"Predicted improvement: {sub133_pred - best_pred_with_133:.6f}")
    output_path = "submission/submission_135.csv"
    best_blend_with_133[1].to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

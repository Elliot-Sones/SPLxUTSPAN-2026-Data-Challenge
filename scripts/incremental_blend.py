"""
Incremental Blend - Start from Sub 133 (best known) and make small adjustments
"""

import pandas as pd
import numpy as np

targets = ['scaled_angle', 'scaled_depth', 'scaled_left_right']

def load_sub(num):
    return pd.read_csv(f"submission/submission_{num}.csv")

def compute_profile(df):
    angle_std = df['scaled_angle'].std()
    depth_mean = df['scaled_depth'].mean()
    return angle_std, depth_mean

def create_blend(weights, subs):
    blend = pd.DataFrame()
    blend['id'] = subs[0]['id']
    for target in targets:
        blend[target] = sum(w * s[target] for w, s in zip(weights, subs))
    return blend

# Load submissions
sub9 = load_sub(9)
sub10 = load_sub(10)
sub25 = load_sub(25)
sub111 = load_sub(111)
sub133 = load_sub(133)
sub20 = load_sub(20)
sub34 = load_sub(34)

# Sub 133 is our best: 5% Sub25 + 30% Sub9 + 44% Sub10 + 21% Sub111
# LB = 0.007809

print("=== Sub 133 Profile ===")
std133, mean133 = compute_profile(sub133)
print(f"angle_std: {std133:.5f}")
print(f"depth_mean: {mean133:.5f}")
print(f"LB: 0.007809 (BEST)")

# Key observations from the data:
# - Sub 133 has the lowest angle_std (0.13773) of all known submissions
# - The winning blend included Sub25 at only 5%
# - Sub9 and Sub10 are very similar (high correlation), but blend differently

# Try variations around Sub 133
print("\n=== Variations around Sub 133 ===")
print("Format: weights (Sub25, Sub9, Sub10, Sub111) -> angle_std, depth_mean")

variations = []

# Vary each weight slightly
for d25 in [-0.05, -0.03, -0.02, 0, 0.02, 0.03, 0.05]:
    for d9 in [-0.10, -0.05, 0, 0.05, 0.10]:
        for d10 in [-0.10, -0.05, 0, 0.05, 0.10]:
            # Base weights from Sub 133
            w25 = 0.05 + d25
            w9 = 0.30 + d9
            w10 = 0.44 + d10
            w111 = 1.0 - w25 - w9 - w10

            # Constraints
            if w25 < 0 or w9 < 0.1 or w10 < 0.2 or w111 < 0:
                continue
            if w25 > 0.15 or w111 > 0.4:
                continue

            weights = [w25, w9, w10, w111]
            blend = create_blend(weights, [sub25, sub9, sub10, sub111])
            angle_std, depth_mean = compute_profile(blend)

            variations.append({
                'w25': w25, 'w9': w9, 'w10': w10, 'w111': w111,
                'angle_std': angle_std, 'depth_mean': depth_mean
            })

# Sort by angle_std (lower is better based on our observations)
variations.sort(key=lambda x: x['angle_std'])

print("\nTop 10 lowest angle_std blends:")
for i, v in enumerate(variations[:10]):
    print(f"{i+1}. ({v['w25']:.2f}, {v['w9']:.2f}, {v['w10']:.2f}, {v['w111']:.2f}) -> "
          f"angle_std={v['angle_std']:.5f}, depth_mean={v['depth_mean']:.5f}")

# The best by angle_std
best = variations[0]
print(f"\nBest by angle_std:")
print(f"Weights: Sub25={best['w25']:.2f}, Sub9={best['w9']:.2f}, Sub10={best['w10']:.2f}, Sub111={best['w111']:.2f}")
print(f"Profile: angle_std={best['angle_std']:.5f}, depth_mean={best['depth_mean']:.5f}")
print(f"vs Sub 133: angle_std delta = {best['angle_std'] - std133:.6f}")

# Create this blend
best_blend = create_blend(
    [best['w25'], best['w9'], best['w10'], best['w111']],
    [sub25, sub9, sub10, sub111]
)

# Also check: what if we include Sub20 or Sub34 (which have good LB scores)?
print("\n=== Including Sub20 (LB 0.008619) ===")
for w20 in [0.02, 0.05, 0.08, 0.10]:
    # Take from the largest weights proportionally
    scale = 1 - w20
    blend = create_blend(
        [w20, 0.05*scale, 0.30*scale, 0.44*scale, 0.21*scale],
        [sub20, sub25, sub9, sub10, sub111]
    )
    angle_std, depth_mean = compute_profile(blend)
    print(f"{int(w20*100)}% Sub20 + {int((1-w20)*100)}% Sub133 equiv: "
          f"angle_std={angle_std:.5f}, depth_mean={depth_mean:.5f}")

print("\n=== Including Sub34 (LB 0.008377) ===")
for w34 in [0.02, 0.05, 0.08, 0.10]:
    scale = 1 - w34
    blend = create_blend(
        [w34, 0.05*scale, 0.30*scale, 0.44*scale, 0.21*scale],
        [sub34, sub25, sub9, sub10, sub111]
    )
    angle_std, depth_mean = compute_profile(blend)
    print(f"{int(w34*100)}% Sub34 + {int((1-w34)*100)}% Sub133 equiv: "
          f"angle_std={angle_std:.5f}, depth_mean={depth_mean:.5f}")

# Save the best blend found
if best['angle_std'] < std133:
    print(f"\n=== Saving Sub 136 ===")
    best_blend.to_csv("submission/submission_136.csv", index=False)
    print(f"Saved submission_136.csv")
    print(f"Weights: Sub25={best['w25']:.2f}, Sub9={best['w9']:.2f}, Sub10={best['w10']:.2f}, Sub111={best['w111']:.2f}")
    print(f"angle_std improvement: {std133 - best['angle_std']:.6f}")
else:
    print("\nNo improvement found over Sub 133")

# Also save the Sub 134 and 135 that were created earlier for comparison
print("\n=== Summary of candidates ===")
print("Sub 133 (BEST KNOWN LB 0.007809): 5% Sub25 + 30% Sub9 + 44% Sub10 + 21% Sub111")
print(f"  angle_std={std133:.5f}, depth_mean={mean133:.5f}")

std134, mean134 = compute_profile(pd.read_csv("submission/submission_134.csv"))
print(f"Sub 134: angle_std={std134:.5f}, depth_mean={mean134:.5f}")

std135, mean135 = compute_profile(pd.read_csv("submission/submission_135.csv"))
print(f"Sub 135: angle_std={std135:.5f}, depth_mean={mean135:.5f}")

if best['angle_std'] < std133:
    print(f"Sub 136: angle_std={best['angle_std']:.5f}, depth_mean={best['depth_mean']:.5f}")

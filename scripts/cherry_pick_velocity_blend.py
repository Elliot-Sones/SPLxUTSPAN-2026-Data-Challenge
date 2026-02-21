"""
Cherry-pick the best velocity model per target for blending with Sub 3190.

Based on diagnostic findings:
- ANGLE: velocity signal is weak (per-player r < 0.18). Don't touch angle.
- DEPTH: validated velocity has r=0.69 diversity, per-player signal r=0.46. Use it.
- LR: pure velocity has r=0.60 diversity (BEST EVER), per-player signal r=0.37. Use it.

Strategy: Per-target blend weights optimized for diversity vs accuracy tradeoff.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import fcntl

SUBMISSION_DIR = Path(__file__).parent.parent / "submission"


def get_next_sub_num():
    lock = SUBMISSION_DIR / ".submission_lock"
    lock.touch(exist_ok=True)
    with open(lock, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            nums = [int(p.stem.split("_")[1]) for p in SUBMISSION_DIR.glob("submission_*.csv")
                    if p.stem.split("_")[1].isdigit()]
            n = max(nums) + 1 if nums else 1
            (SUBMISSION_DIR / f"submission_{n}.csv").touch()
            return n
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def main():
    # Load base
    base = pd.read_csv(SUBMISSION_DIR / "submission_3190.csv")

    # Load validated velocity (Sub 3204) - best for depth
    val_vel = pd.read_csv(SUBMISSION_DIR / "submission_3204.csv")

    # Load pure velocity (Sub 3209) - best for LR
    pure_vel = pd.read_csv(SUBMISSION_DIR / "submission_3209.csv")

    print("=" * 60)
    print("CHERRY-PICK VELOCITY BLEND")
    print("=" * 60)

    # Diversity verification
    print("\nDiversity check:")
    for t in ["angle", "depth", "left_right"]:
        col = f"scaled_{t}"
        r_val = np.corrcoef(base[col], val_vel[col])[0, 1]
        r_pure = np.corrcoef(base[col], pure_vel[col])[0, 1]
        r_vp = np.corrcoef(val_vel[col], pure_vel[col])[0, 1]
        print(f"  {t:10s}: base-val={r_val:.4f}, base-pure={r_pure:.4f}, val-pure={r_vp:.4f}")

    # Generate multiple weight configs
    configs = [
        # (name, angle_source, angle_w, depth_source, depth_w, lr_source, lr_w)
        ("depth5_lr5", None, 0, "val", 0.05, "pure", 0.05),
        ("depth7_lr5", None, 0, "val", 0.07, "pure", 0.05),
        ("depth5_lr7", None, 0, "val", 0.05, "pure", 0.07),
        ("depth3_lr3", None, 0, "val", 0.03, "pure", 0.03),
        ("depth5_lr3", None, 0, "val", 0.05, "pure", 0.03),
        ("depth3_lr5", None, 0, "val", 0.03, "pure", 0.05),
        ("a2_depth5_lr5", "val", 0.02, "val", 0.05, "pure", 0.05),
        ("a2_depth7_lr7", "val", 0.02, "val", 0.07, "pure", 0.07),
        # Also try pure velocity for depth (to test if removing positions helps depth too)
        ("pure_d5_lr5", None, 0, "pure", 0.05, "pure", 0.05),
        ("pure_d5_lr7", None, 0, "pure", 0.05, "pure", 0.07),
    ]

    for cname, a_src, a_w, d_src, d_w, lr_src, lr_w in configs:
        bn = get_next_sub_num()
        blend = pd.DataFrame({"id": base["id"]})

        src_map = {"val": val_vel, "pure": pure_vel}

        # Angle
        if a_src and a_w > 0:
            blend["scaled_angle"] = (1 - a_w) * base["scaled_angle"] + a_w * src_map[a_src]["scaled_angle"]
        else:
            blend["scaled_angle"] = base["scaled_angle"]

        # Depth
        if d_src and d_w > 0:
            blend["scaled_depth"] = (1 - d_w) * base["scaled_depth"] + d_w * src_map[d_src]["scaled_depth"]
        else:
            blend["scaled_depth"] = base["scaled_depth"]

        # LR
        if lr_src and lr_w > 0:
            blend["scaled_left_right"] = (1 - lr_w) * base["scaled_left_right"] + lr_w * src_map[lr_src]["scaled_left_right"]
        else:
            blend["scaled_left_right"] = base["scaled_left_right"]

        blend.to_csv(SUBMISSION_DIR / f"submission_{bn}.csv", index=False)
        print(f"  Sub {bn}: {cname} (angle={a_w*100:.0f}% depth={d_w*100:.0f}% lr={lr_w*100:.0f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()

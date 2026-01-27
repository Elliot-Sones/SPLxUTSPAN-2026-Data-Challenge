#!/usr/bin/env python3
"""
Precision testing script.
Demonstrates the precision loss problem and validates fixes.
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from decimal import Decimal, getcontext

DATA_DIR = Path(__file__).parent.parent / "data"
getcontext().prec = 50  # High precision decimals


def test_raw_data_precision():
    """Test precision loss at data loading stage."""
    print("=" * 60)
    print("TEST 1: Raw Data Precision Loss")
    print("=" * 60)

    # Read raw CSV to get string value
    df = pd.read_csv(DATA_DIR / "train.csv", nrows=1)
    raw_str = df.iloc[0]['nose_x']

    # Extract first value from array string
    raw_str = raw_str.strip('[]').split(',')[0].strip()

    print(f"\nRaw string value: {raw_str}")
    print(f"String length: {len(raw_str)} characters")

    # Parse with different types
    as_decimal = Decimal(raw_str)
    as_float64 = np.float64(float(raw_str))
    as_float32 = np.float32(float(raw_str))

    print(f"\nAs Decimal (ground truth): {as_decimal}")
    print(f"As float64: {as_float64:.18f}")
    print(f"As float32: {float(as_float32):.18f}")

    f64_error = abs(float(as_decimal) - as_float64)
    f32_error = abs(float(as_decimal) - float(as_float32))

    print(f"\nfloat64 absolute error: {f64_error:.2e}")
    print(f"float32 absolute error: {f32_error:.2e}")
    print(f"float32 loses {int(np.log10(f32_error / (f64_error + 1e-20)))} additional decimal places")

    return f32_error, f64_error


def test_derivative_error_amplification():
    """Test how precision loss amplifies through derivatives."""
    print("\n" + "=" * 60)
    print("TEST 2: Derivative Error Amplification")
    print("=" * 60)

    dt = 1.0 / 60  # 60 FPS

    # Two consecutive position values (realistic motion capture data)
    # The difference represents actual motion over 1 frame
    p1 = 19.015902261279695
    p2 = 19.015912261279695  # +10 micrometers motion

    true_velocity = (p2 - p1) / dt

    # Float32 computation
    p1_f32 = np.float32(p1)
    p2_f32 = np.float32(p2)
    vel_f32 = (p2_f32 - p1_f32) / np.float32(dt)

    # Float64 computation
    p1_f64 = np.float64(p1)
    p2_f64 = np.float64(p2)
    vel_f64 = (p2_f64 - p1_f64) / np.float64(dt)

    print(f"\nPosition 1: {p1}")
    print(f"Position 2: {p2}")
    print(f"True difference: {p2 - p1:.15e}")

    print(f"\nTrue velocity: {true_velocity:.10e} m/s")
    print(f"Float32 velocity: {float(vel_f32):.10e} m/s")
    print(f"Float64 velocity: {vel_f64:.10e} m/s")

    f32_vel_error = abs(float(vel_f32) - true_velocity) / abs(true_velocity) * 100
    f64_vel_error = abs(vel_f64 - true_velocity) / abs(true_velocity) * 100

    print(f"\nFloat32 velocity error: {f32_vel_error:.2f}%")
    print(f"Float64 velocity error: {f64_vel_error:.6f}%")

    # Now compute acceleration (second derivative)
    p0 = 19.015892261279695
    p3 = 19.015922261279695

    true_acc = (p3 - 2*p2 + p1) / (dt * dt)

    # Float32
    p0_f32, p3_f32 = np.float32(p0), np.float32(p3)
    acc_f32 = (p3_f32 - 2*p2_f32 + p1_f32) / (np.float32(dt) ** 2)

    # Float64
    p0_f64, p3_f64 = np.float64(p0), np.float64(p3)
    acc_f64 = (p3_f64 - 2*p2_f64 + p1_f64) / (np.float64(dt) ** 2)

    print(f"\nTrue acceleration: {true_acc:.6e} m/s^2")
    print(f"Float32 acceleration: {float(acc_f32):.6e} m/s^2")
    print(f"Float64 acceleration: {acc_f64:.6e} m/s^2")

    f32_acc_error = abs(float(acc_f32) - true_acc) / (abs(true_acc) + 1e-10) * 100
    f64_acc_error = abs(acc_f64 - true_acc) / (abs(true_acc) + 1e-10) * 100

    print(f"\nFloat32 acceleration error: {f32_acc_error:.2f}%")
    print(f"Float64 acceleration error: {f64_acc_error:.6f}%")

    return f32_vel_error, f64_vel_error


def test_arccos_instability():
    """Test arccos instability near boundaries."""
    print("\n" + "=" * 60)
    print("TEST 3: Arccos Instability Near Boundaries")
    print("=" * 60)

    # Values very close to 1 (nearly parallel vectors)
    values = [0.9999999, 0.99999999, 0.999999999, 1.0 - 1e-10]

    print("\nComputing angles for cos(angle) values near 1:")
    for v in values:
        v_f32 = np.float32(v)
        v_f64 = np.float64(v)

        # Float32 might round to exactly 1.0
        angle_f32 = np.arccos(np.clip(v_f32, -1, 1)) * 180 / np.pi
        angle_f64 = np.arccos(np.clip(v_f64, -1, 1)) * 180 / np.pi

        print(f"  cos = {v}: float32 angle = {float(angle_f32):.6f} deg, "
              f"float64 angle = {angle_f64:.6f} deg")

    # The atan2 alternative
    print("\nUsing atan2 (stable) instead:")
    for v in values:
        # atan2 approach: angle = atan2(sqrt(1 - cos^2), cos)
        sin_val = np.sqrt(1 - v**2)
        angle_atan2 = np.arctan2(sin_val, v) * 180 / np.pi
        print(f"  cos = {v}: atan2 angle = {angle_atan2:.10f} deg")


def test_summation_precision():
    """Test precision loss in summation (energy computation)."""
    print("\n" + "=" * 60)
    print("TEST 4: Summation Precision (Energy Computation)")
    print("=" * 60)

    # Simulate 240 frames of position data
    np.random.seed(42)
    positions = 19.0 + np.random.randn(240) * 0.01  # Small variations around 19

    # Compute energy (sum of squares)
    # Float32
    pos_f32 = positions.astype(np.float32)
    energy_f32 = np.sum(pos_f32 ** 2)

    # Float64
    pos_f64 = positions.astype(np.float64)
    energy_f64 = np.sum(pos_f64 ** 2)

    # Ground truth using math.fsum (exact summation)
    import math
    energy_exact = math.fsum(positions ** 2)

    print(f"\nEnergy (sum of 240 squared values):")
    print(f"Exact (math.fsum): {energy_exact:.15f}")
    print(f"Float64: {energy_f64:.15f}")
    print(f"Float32: {float(energy_f32):.15f}")

    print(f"\nFloat64 error: {abs(energy_f64 - energy_exact):.2e}")
    print(f"Float32 error: {abs(float(energy_f32) - energy_exact):.2e}")


def test_full_pipeline_comparison():
    """Compare feature values between float32 and float64 pipelines."""
    print("\n" + "=" * 60)
    print("TEST 5: Full Pipeline Feature Comparison")
    print("=" * 60)

    # Load one shot's raw data
    df = pd.read_csv(DATA_DIR / "train.csv", nrows=1)

    # Get keypoint columns
    meta_cols = ["id", "shot_id", "participant_id"]
    target_cols = ["angle", "depth", "left_right"]
    kp_cols = [c for c in df.columns if c not in meta_cols + target_cols]

    # Parse as float32 (current)
    def parse_f32(s):
        s = s.replace("nan", "null")
        return np.array(json.loads(s), dtype=np.float32)

    # Parse as float64 (proposed)
    def parse_f64(s):
        s = s.replace("nan", "null")
        return np.array(json.loads(s), dtype=np.float64)

    # Build timeseries
    ts_f32 = np.zeros((240, len(kp_cols)), dtype=np.float32)
    ts_f64 = np.zeros((240, len(kp_cols)), dtype=np.float64)

    for i, col in enumerate(kp_cols):
        ts_f32[:, i] = parse_f32(df.iloc[0][col])
        ts_f64[:, i] = parse_f64(df.iloc[0][col])

    # Compare basic statistics
    print("\nComparing basic statistics for first keypoint (nose_x):")
    col_idx = 0

    stats = ['mean', 'std', 'min', 'max', 'sum']
    for stat in stats:
        f32_val = getattr(np, stat)(ts_f32[:, col_idx])
        f64_val = getattr(np, stat)(ts_f64[:, col_idx])
        diff = abs(float(f32_val) - f64_val)
        rel_diff = diff / (abs(f64_val) + 1e-10) * 100
        print(f"  {stat:6s}: f32={float(f32_val):.10f}, f64={f64_val:.10f}, "
              f"diff={diff:.2e} ({rel_diff:.4f}%)")

    # Compare velocity computation
    print("\nComparing velocity for first keypoint:")
    vel_f32 = np.gradient(ts_f32[:, col_idx], 1/60)
    vel_f64 = np.gradient(ts_f64[:, col_idx], 1/60)

    vel_diffs = np.abs(vel_f32.astype(np.float64) - vel_f64)
    print(f"  Max velocity difference: {np.max(vel_diffs):.2e}")
    print(f"  Mean velocity difference: {np.mean(vel_diffs):.2e}")
    print(f"  Max relative difference: {np.max(vel_diffs / (np.abs(vel_f64) + 1e-10)) * 100:.4f}%")


def main():
    """Run all precision tests."""
    print("\n" + "#" * 60)
    print("# NUMERICAL PRECISION TEST SUITE")
    print("#" * 60)

    test_raw_data_precision()
    test_derivative_error_amplification()
    test_arccos_instability()
    test_summation_precision()
    test_full_pipeline_comparison()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
1. Float32 loses ~8 decimal places at data loading time
2. This error is amplified ~60x in velocity computation (1/dt factor)
3. And ~3600x in acceleration computation (1/dt^2 factor)
4. Arccos is unstable near boundaries - use atan2 instead
5. Summation errors accumulate over 240 frames

RECOMMENDATION: Use float64 for all computation, float32 only for model input.
""")


if __name__ == "__main__":
    main()

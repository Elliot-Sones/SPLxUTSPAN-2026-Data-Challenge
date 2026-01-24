#!/usr/bin/env python3
"""
SkillMimic File Inspector

Analyzes individual .pt files from the SkillMimic dataset.
Provides comprehensive statistics, exports to multiple formats, and generates visualizations.

Usage:
    uv run python skillmimic_inspector.py <path_to_file.pt> [--output-dir OUTPUT]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_skillmimic_file(filepath: str) -> torch.Tensor:
    """
    Load a SkillMimic .pt file and validate it.

    Args:
        filepath: Path to the .pt file

    Returns:
        torch.Tensor: Loaded tensor data

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If file is not a valid PyTorch tensor
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        data = torch.load(filepath, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to load .pt file: {e}")

    if not isinstance(data, torch.Tensor):
        raise RuntimeError(f"Expected torch.Tensor, got {type(data)}")

    return data


def print_summary(data: torch.Tensor, filepath: str) -> None:
    """Print comprehensive summary of the tensor data."""
    print(f"\n{'='*80}")
    print(f"SkillMimic File Analysis: {Path(filepath).name}")
    print(f"{'='*80}\n")

    # Basic structure
    print("STRUCTURE:")
    print(f"  Shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    print(f"  Device: {data.device}")
    print(f"  Memory: {data.element_size() * data.nelement() / 1024:.2f} KB")
    print(f"  Num frames: {data.shape[0]}")
    print(f"  Num features: {data.shape[1]}")

    # Global statistics
    print("\nGLOBAL STATISTICS:")
    print(f"  Min: {data.min().item():.6f}")
    print(f"  Max: {data.max().item():.6f}")
    print(f"  Mean: {data.mean().item():.6f}")
    print(f"  Std: {data.std().item():.6f}")
    print(f"  Median: {data.median().item():.6f}")

    # Check for NaN/Inf
    num_nan = torch.isnan(data).sum().item()
    num_inf = torch.isinf(data).sum().item()
    print(f"  NaN values: {num_nan}")
    print(f"  Inf values: {num_inf}")

    # Per-feature statistics (summary)
    print("\nPER-FEATURE STATISTICS (first 10 features):")
    print(f"  {'Feature':<10} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
    print(f"  {'-'*58}")
    for i in range(min(10, data.shape[1])):
        feature_data = data[:, i]
        print(f"  {i:<10} {feature_data.min().item():<12.6f} {feature_data.max().item():<12.6f} "
              f"{feature_data.mean().item():<12.6f} {feature_data.std().item():<12.6f}")
    if data.shape[1] > 10:
        print(f"  ... ({data.shape[1] - 10} more features)")

    # Temporal statistics
    print("\nTEMPORAL STATISTICS:")
    frame_diffs = torch.diff(data, dim=0)
    print(f"  Frame-to-frame change (mean L2 norm): {torch.norm(frame_diffs, dim=1).mean().item():.6f}")
    print(f"  Frame-to-frame change (max L2 norm): {torch.norm(frame_diffs, dim=1).max().item():.6f}")
    print(f"  Frame-to-frame change (min L2 norm): {torch.norm(frame_diffs, dim=1).min().item():.6f}")


def export_csv(data: torch.Tensor, output_path: Path) -> None:
    """Export tensor to CSV format (frames as rows, features as columns)."""
    df = pd.DataFrame(
        data.numpy(),
        columns=[f"feature_{i:03d}" for i in range(data.shape[1])]
    )
    df.insert(0, 'frame', range(len(df)))
    df.to_csv(output_path, index=False)
    print(f"\nExported CSV: {output_path}")


def export_numpy(data: torch.Tensor, output_path: Path) -> None:
    """Export tensor to NumPy .npy format."""
    np.save(output_path, data.numpy())
    print(f"Exported NumPy: {output_path}")


def export_metadata(data: torch.Tensor, filepath: str, output_path: Path) -> None:
    """Export metadata and statistics to JSON."""
    metadata = {
        "source_file": str(filepath),
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "num_frames": data.shape[0],
        "num_features": data.shape[1],
        "memory_kb": float(data.element_size() * data.nelement() / 1024),
        "global_stats": {
            "min": float(data.min().item()),
            "max": float(data.max().item()),
            "mean": float(data.mean().item()),
            "std": float(data.std().item()),
            "median": float(data.median().item()),
        },
        "temporal_stats": {
            "frame_to_frame_l2_mean": float(torch.norm(torch.diff(data, dim=0), dim=1).mean().item()),
            "frame_to_frame_l2_max": float(torch.norm(torch.diff(data, dim=0), dim=1).max().item()),
            "frame_to_frame_l2_min": float(torch.norm(torch.diff(data, dim=0), dim=1).min().item()),
        },
        "nan_count": int(torch.isnan(data).sum().item()),
        "inf_count": int(torch.isinf(data).sum().item()),
    }

    with open(output_path, 'w') as f:
        json.dump(metadata, indent=2, fp=f)
    print(f"Exported metadata: {output_path}")


def plot_timeseries(data: torch.Tensor, num_features: int, output_path: Path) -> None:
    """Plot time series for selected features."""
    num_features = min(num_features, data.shape[1])

    fig = make_subplots(
        rows=num_features, cols=1,
        subplot_titles=[f"Feature {i}" for i in range(num_features)],
        shared_xaxes=True,
        vertical_spacing=0.02
    )

    for i in range(num_features):
        fig.add_trace(
            go.Scatter(
                x=list(range(data.shape[0])),
                y=data[:, i].numpy(),
                mode='lines',
                name=f'Feature {i}',
                showlegend=False
            ),
            row=i+1, col=1
        )

    fig.update_layout(
        height=200 * num_features,
        title_text="Feature Time Series",
        showlegend=False
    )
    fig.update_xaxes(title_text="Frame", row=num_features, col=1)

    fig.write_html(output_path)
    print(f"Exported time series plot: {output_path}")


def plot_distributions(data: torch.Tensor, output_path: Path, max_features: int = 20) -> None:
    """Plot distribution histograms for features."""
    num_features = min(max_features, data.shape[1])
    cols = 5
    rows = (num_features + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"F{i}" for i in range(num_features)]
    )

    for i in range(num_features):
        row = i // cols + 1
        col = i % cols + 1

        fig.add_trace(
            go.Histogram(
                x=data[:, i].numpy(),
                name=f'Feature {i}',
                showlegend=False,
                nbinsx=30
            ),
            row=row, col=col
        )

    fig.update_layout(
        height=250 * rows,
        title_text="Feature Value Distributions",
        showlegend=False
    )

    fig.write_html(output_path)
    print(f"Exported distribution plot: {output_path}")


def plot_correlation_matrix(data: torch.Tensor, output_path: Path, sample_features: int = 50) -> None:
    """Plot correlation heatmap (sampled for performance)."""
    # Sample features if too many
    num_features = min(sample_features, data.shape[1])
    step = data.shape[1] // num_features
    sampled_indices = list(range(0, data.shape[1], step))[:num_features]

    sampled_data = data[:, sampled_indices]
    corr_matrix = np.corrcoef(sampled_data.numpy().T)

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=[f"F{i}" for i in sampled_indices],
        y=[f"F{i}" for i in sampled_indices],
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=f"Feature Correlation Matrix (sampled {num_features} features)",
        xaxis_title="Feature",
        yaxis_title="Feature",
        height=800,
        width=900
    )

    fig.write_html(output_path)
    print(f"Exported correlation heatmap: {output_path}")


def infer_feature_types(data: torch.Tensor) -> Dict[str, Any]:
    """
    Infer feature types based on temporal derivatives.
    Position features should be smooth, velocity features are first derivatives.
    """
    # Calculate first and second derivatives
    first_deriv = torch.diff(data, dim=0)
    second_deriv = torch.diff(first_deriv, dim=0)

    # Statistics for each feature
    feature_types = {
        "likely_positions": [],
        "likely_velocities": [],
        "likely_accelerations": [],
        "likely_other": []
    }

    for i in range(data.shape[1]):
        # Analyze smoothness (low second derivative = position-like)
        smoothness = second_deriv[:, i].abs().mean().item()
        # Analyze change rate (high first derivative = velocity-like)
        change_rate = first_deriv[:, i].abs().mean().item()
        # Analyze value range
        value_range = data[:, i].max().item() - data[:, i].min().item()

        # Simple heuristics
        if smoothness < 0.01 and value_range > 0.5:
            feature_types["likely_positions"].append(i)
        elif change_rate > 0.1 and smoothness > 0.01:
            feature_types["likely_velocities"].append(i)
        elif smoothness > 0.05:
            feature_types["likely_accelerations"].append(i)
        else:
            feature_types["likely_other"].append(i)

    return feature_types


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SkillMimic .pt files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("filepath", help="Path to .pt file")
    parser.add_argument("--output-dir", default=".", help="Output directory for exports")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualizations")

    args = parser.parse_args()

    # Load data
    print(f"Loading {args.filepath}...")
    data = load_skillmimic_file(args.filepath)

    # Print summary
    print_summary(data, args.filepath)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate base filename
    base_name = Path(args.filepath).stem

    # Export formats
    print("\nEXPORTING DATA:")
    export_csv(data, output_dir / f"{base_name}.csv")
    export_numpy(data, output_dir / f"{base_name}.npy")
    export_metadata(data, args.filepath, output_dir / f"{base_name}_metadata.json")

    # Infer feature types
    print("\nFEATURE TYPE INFERENCE:")
    feature_types = infer_feature_types(data)
    for ftype, indices in feature_types.items():
        print(f"  {ftype}: {len(indices)} features")

    # Save feature types
    with open(output_dir / f"{base_name}_feature_types.json", 'w') as f:
        json.dump(feature_types, indent=2, fp=f)
    print(f"Exported feature types: {output_dir / f'{base_name}_feature_types.json'}")

    # Generate visualizations
    if not args.no_viz:
        print("\nGENERATING VISUALIZATIONS:")
        plot_timeseries(data, num_features=20, output_path=output_dir / "timeseries.html")
        plot_distributions(data, output_path=output_dir / "distributions.html", max_features=20)
        plot_correlation_matrix(data, output_path=output_dir / "correlation_heatmap.html", sample_features=50)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

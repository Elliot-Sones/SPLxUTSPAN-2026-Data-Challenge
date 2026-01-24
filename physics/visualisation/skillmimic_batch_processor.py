#!/usr/bin/env python3
"""
SkillMimic Batch Processor

Processes multiple .pt files from a directory and generates comparative analysis.

Usage:
    uv run python skillmimic_batch_processor.py <directory> [--pattern "*.pt"] [--output OUTPUT]
"""

import argparse
from pathlib import Path
from typing import Dict, List
import time

import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def find_pt_files(directory: str, pattern: str = "*.pt") -> List[Path]:
    """Find all .pt files in directory matching pattern."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pt_files = list(directory.glob(pattern))
    if not pt_files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {directory}")

    return sorted(pt_files)


def load_batch(filepaths: List[Path]) -> Dict[str, torch.Tensor]:
    """Load multiple .pt files and return as dictionary."""
    data_dict = {}
    print(f"Loading {len(filepaths)} files...")

    for filepath in filepaths:
        try:
            start_time = time.time()
            data = torch.load(filepath, map_location='cpu')
            load_time = time.time() - start_time

            if not isinstance(data, torch.Tensor):
                print(f"  Warning: {filepath.name} is not a tensor (type: {type(data)}), skipping")
                continue

            data_dict[filepath.name] = {
                'data': data,
                'filepath': filepath,
                'load_time': load_time
            }
            print(f"  Loaded: {filepath.name} - Shape: {data.shape}")

        except Exception as e:
            print(f"  Error loading {filepath.name}: {e}")

    print(f"\nSuccessfully loaded {len(data_dict)} files\n")
    return data_dict


def validate_consistency(data_dict: Dict[str, Dict]) -> Dict[str, any]:
    """Validate consistency across files and generate report."""
    if not data_dict:
        return {"error": "No data loaded"}

    shapes = [entry['data'].shape for entry in data_dict.values()]
    dtypes = [entry['data'].dtype for entry in data_dict.values()]

    report = {
        "num_files": len(data_dict),
        "shapes": shapes,
        "unique_shapes": list(set(shapes)),
        "dtypes": list(set(dtypes)),
        "consistent_shape": len(set(shapes)) == 1,
        "consistent_dtype": len(set(dtypes)) == 1,
    }

    # Check feature count consistency
    feature_counts = [shape[1] if len(shape) > 1 else 0 for shape in shapes]
    report["feature_counts"] = feature_counts
    report["consistent_features"] = len(set(feature_counts)) == 1

    return report


def generate_summary_table(data_dict: Dict[str, Dict]) -> pd.DataFrame:
    """Generate summary statistics table for all files."""
    rows = []

    for filename, entry in data_dict.items():
        data = entry['data']

        # Calculate statistics
        row = {
            'filename': filename,
            'num_frames': data.shape[0],
            'num_features': data.shape[1] if len(data.shape) > 1 else 1,
            'min': data.min().item(),
            'max': data.max().item(),
            'mean': data.mean().item(),
            'std': data.std().item(),
            'median': data.median().item(),
            'nan_count': torch.isnan(data).sum().item(),
            'inf_count': torch.isinf(data).sum().item(),
            'memory_kb': data.element_size() * data.nelement() / 1024,
            'load_time_ms': entry['load_time'] * 1000,
        }

        # Temporal stats
        if data.shape[0] > 1:
            frame_diffs = torch.diff(data, dim=0)
            row['frame_change_mean_l2'] = torch.norm(frame_diffs, dim=1).mean().item()
            row['frame_change_max_l2'] = torch.norm(frame_diffs, dim=1).max().item()
        else:
            row['frame_change_mean_l2'] = 0.0
            row['frame_change_max_l2'] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def analyze_cross_file_statistics(data_dict: Dict[str, Dict]) -> Dict[str, any]:
    """Analyze statistics across all files."""
    all_data = [entry['data'] for entry in data_dict.values()]

    # Check if all have same number of features
    feature_counts = [d.shape[1] if len(d.shape) > 1 else 1 for d in all_data]
    if len(set(feature_counts)) != 1:
        return {
            "error": "Inconsistent feature counts across files",
            "feature_counts": feature_counts
        }

    num_features = feature_counts[0]
    num_files = len(all_data)

    # Per-feature statistics across files
    per_feature_stats = []
    for feature_idx in range(num_features):
        feature_values = torch.cat([d[:, feature_idx] for d in all_data])

        stats = {
            'feature_idx': feature_idx,
            'min': feature_values.min().item(),
            'max': feature_values.max().item(),
            'mean': feature_values.mean().item(),
            'std': feature_values.std().item(),
            'median': feature_values.median().item(),
        }

        # Calculate variance across files (how much does this feature vary between motions)
        file_means = [d[:, feature_idx].mean().item() for d in all_data]
        stats['cross_file_mean_std'] = np.std(file_means)

        per_feature_stats.append(stats)

    return {
        'num_files': num_files,
        'num_features': num_features,
        'per_feature_stats': per_feature_stats,
    }


def plot_comparative_timeseries(
    data_dict: Dict[str, Dict],
    feature_idx: int,
    output_path: Path,
    max_files: int = 10
) -> None:
    """Plot time series of a specific feature across multiple files."""
    fig = go.Figure()

    filenames = list(data_dict.keys())[:max_files]

    for filename in filenames:
        data = data_dict[filename]['data']
        if feature_idx < data.shape[1]:
            fig.add_trace(go.Scatter(
                x=list(range(data.shape[0])),
                y=data[:, feature_idx].numpy(),
                mode='lines',
                name=filename,
                opacity=0.7
            ))

    fig.update_layout(
        title=f"Feature {feature_idx} Across Files",
        xaxis_title="Frame",
        yaxis_title="Value",
        height=600,
        hovermode='x unified'
    )

    fig.write_html(output_path)


def plot_duration_comparison(data_dict: Dict[str, Dict], output_path: Path) -> None:
    """Plot comparison of motion durations (number of frames)."""
    filenames = list(data_dict.keys())
    frame_counts = [data_dict[fn]['data'].shape[0] for fn in filenames]

    fig = go.Figure(data=[
        go.Bar(x=filenames, y=frame_counts)
    ])

    fig.update_layout(
        title="Motion Duration Comparison",
        xaxis_title="File",
        yaxis_title="Number of Frames",
        height=500,
        xaxis_tickangle=-45
    )

    fig.write_html(output_path)


def plot_feature_variance_comparison(cross_file_stats: Dict, output_path: Path) -> None:
    """Plot which features vary most across different motions."""
    per_feature_stats = cross_file_stats['per_feature_stats']

    feature_indices = [s['feature_idx'] for s in per_feature_stats]
    cross_file_stds = [s['cross_file_mean_std'] for s in per_feature_stats]

    # Sort by variance
    sorted_pairs = sorted(zip(feature_indices, cross_file_stds), key=lambda x: x[1], reverse=True)
    top_n = 50
    top_features = [p[0] for p in sorted_pairs[:top_n]]
    top_variances = [p[1] for p in sorted_pairs[:top_n]]

    fig = go.Figure(data=[
        go.Bar(x=top_features, y=top_variances)
    ])

    fig.update_layout(
        title=f"Top {top_n} Most Variable Features Across Files",
        xaxis_title="Feature Index",
        yaxis_title="Cross-File Std Dev of Mean",
        height=600
    )

    fig.write_html(output_path)


def generate_batch_report(
    data_dict: Dict[str, Dict],
    consistency_report: Dict,
    cross_file_stats: Dict,
    output_path: Path
) -> None:
    """Generate HTML report with all batch analysis results."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SkillMimic Batch Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; }}
            .good {{ color: green; }}
            .warning {{ color: orange; }}
            .error {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>SkillMimic Batch Analysis Report</h1>

        <h2>Summary</h2>
        <table>
            <tr><td class="metric">Total Files</td><td>{consistency_report['num_files']}</td></tr>
            <tr><td class="metric">Consistent Shape</td><td class="{'good' if consistency_report['consistent_shape'] else 'error'}">{consistency_report['consistent_shape']}</td></tr>
            <tr><td class="metric">Consistent Features</td><td class="{'good' if consistency_report['consistent_features'] else 'error'}">{consistency_report['consistent_features']}</td></tr>
            <tr><td class="metric">Unique Shapes</td><td>{consistency_report['unique_shapes']}</td></tr>
            <tr><td class="metric">Data Types</td><td>{consistency_report['dtypes']}</td></tr>
        </table>
    """

    if not cross_file_stats.get('error'):
        html_content += f"""
        <h2>Cross-File Statistics</h2>
        <table>
            <tr><td class="metric">Number of Features</td><td>{cross_file_stats['num_features']}</td></tr>
            <tr><td class="metric">Number of Files</td><td>{cross_file_stats['num_files']}</td></tr>
        </table>
        """

    html_content += """
    </body>
    </html>
    """

    with open(output_path, 'w') as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(
        description="Batch process SkillMimic .pt files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", help="Directory containing .pt files")
    parser.add_argument("--pattern", default="*.pt", help="File pattern to match (default: *.pt)")
    parser.add_argument("--output", default="batch_analysis", help="Output directory")

    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"SkillMimic Batch Processor")
    print(f"{'='*80}\n")

    # Find files
    pt_files = find_pt_files(args.directory, args.pattern)
    print(f"Found {len(pt_files)} files matching '{args.pattern}' in {args.directory}\n")

    # Load all files
    data_dict = load_batch(pt_files)

    if not data_dict:
        print("Error: No files successfully loaded")
        return

    # Validate consistency
    print("CONSISTENCY CHECK:")
    consistency_report = validate_consistency(data_dict)
    for key, value in consistency_report.items():
        print(f"  {key}: {value}")
    print()

    # Generate summary table
    print("Generating summary table...")
    summary_df = generate_summary_table(data_dict)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary table
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary table: {summary_path}")

    # Print summary table
    print("\nSUMMARY TABLE:")
    print(summary_df.to_string(index=False))
    print()

    # Cross-file analysis
    print("Analyzing cross-file statistics...")
    cross_file_stats = analyze_cross_file_statistics(data_dict)

    if not cross_file_stats.get('error'):
        # Save per-feature stats
        per_feature_df = pd.DataFrame(cross_file_stats['per_feature_stats'])
        per_feature_path = output_dir / "per_feature_cross_file_stats.csv"
        per_feature_df.to_csv(per_feature_path, index=False)
        print(f"Saved per-feature statistics: {per_feature_path}")

        # Generate visualizations
        print("\nGenerating comparative visualizations...")
        viz_dir = output_dir / "comparative_plots"
        viz_dir.mkdir(exist_ok=True)

        # Plot first few features as examples
        for feature_idx in range(min(5, cross_file_stats['num_features'])):
            plot_comparative_timeseries(
                data_dict,
                feature_idx,
                viz_dir / f"feature_{feature_idx:03d}_comparison.html"
            )
        print(f"  Created feature comparison plots in {viz_dir}")

        # Duration comparison
        plot_duration_comparison(data_dict, viz_dir / "duration_comparison.html")
        print(f"  Created duration comparison plot")

        # Feature variance comparison
        plot_feature_variance_comparison(cross_file_stats, viz_dir / "feature_variance.html")
        print(f"  Created feature variance plot")

    # Generate HTML report
    print("\nGenerating batch report...")
    generate_batch_report(
        data_dict,
        consistency_report,
        cross_file_stats,
        output_dir / "report.html"
    )
    print(f"Saved batch report: {output_dir / 'report.html'}")

    print(f"\n{'='*80}")
    print(f"Batch analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

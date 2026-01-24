"""
Batch visualization tool for SPLxUTSPAN Data Challenge 2026
Generate visualizations for multiple shots at once
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path

# Import the main viewer
from frame_by_frame_viewer import (
    parse_array_json,
    load_shot_data,
    create_interactive_visualization,
    create_data_table
)


def visualize_batch(csv_path, shot_indices, output_dir='batch_output'):
    """Generate visualizations for multiple shots"""

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    print("="*80)
    print(f"Batch Visualization - {len(shot_indices)} shots")
    print("="*80)

    results = []

    for i, shot_index in enumerate(shot_indices):
        print(f"\nProcessing shot {i+1}/{len(shot_indices)} (index: {shot_index})...")

        try:
            shot, keypoint_names = load_shot_data(csv_path, shot_index=shot_index)
            num_frames = len(shot[keypoint_names[0]])

            print(f"  Shot ID: {shot['shot_id']}")
            print(f"  Participant: {shot['participant_id']}")
            print(f"  Frames: {num_frames} ({num_frames/60:.2f}s)")
            print(f"  Angle: {shot['angle']:.2f}° | Depth: {shot['depth']:.2f}\" | L/R: {shot['left_right']:.2f}\"")

            # Create output filenames
            viewer_file = f'{output_dir}/shot_{shot_index}_viewer.html'
            table_file = f'{output_dir}/shot_{shot_index}_table.html'

            # Create visualizations
            create_interactive_visualization(shot, keypoint_names, viewer_file)
            create_data_table(shot, keypoint_names, table_file)

            results.append({
                'index': shot_index,
                'shot_id': shot['shot_id'],
                'participant_id': shot['participant_id'],
                'num_frames': num_frames,
                'duration': num_frames/60,
                'angle': shot['angle'],
                'depth': shot['depth'],
                'left_right': shot['left_right'],
                'viewer_file': viewer_file,
                'table_file': table_file,
                'status': 'Success'
            })

            print(f"  ✓ Generated: {viewer_file}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'index': shot_index,
                'status': f'Failed: {e}'
            })

    # Create summary HTML
    create_summary_html(results, f'{output_dir}/index.html')

    print("\n" + "="*80)
    print("Batch processing complete!")
    print(f"Results saved to: {output_dir}/")
    print(f"Open {output_dir}/index.html to see all visualizations")
    print("="*80)

    return results


def create_summary_html(results, output_path):
    """Create an index page with links to all visualizations"""

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Shot Visualizations - Summary</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        a {
            color: #2196F3;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .success {
            color: green;
            font-weight: bold;
        }
        .failed {
            color: red;
            font-weight: bold;
        }
        .stats {
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <h1>Shot Visualizations Summary</h1>

    <div class="stats">
        <h2>Statistics</h2>
"""

    # Calculate statistics
    successful = [r for r in results if r['status'] == 'Success']
    failed = [r for r in results if r['status'] != 'Success']

    html += f"""
        <p><strong>Total Shots Processed:</strong> {len(results)}</p>
        <p><strong>Successful:</strong> <span class="success">{len(successful)}</span></p>
        <p><strong>Failed:</strong> <span class="failed">{len(failed)}</span></p>
    </div>

    <table>
        <tr>
            <th>Index</th>
            <th>Shot ID</th>
            <th>Participant</th>
            <th>Frames</th>
            <th>Duration (s)</th>
            <th>Angle (°)</th>
            <th>Depth (")</th>
            <th>L/R (")</th>
            <th>3D Viewer</th>
            <th>Data Table</th>
            <th>Status</th>
        </tr>
"""

    for r in results:
        if r['status'] == 'Success':
            html += f"""
        <tr>
            <td>{r['index']}</td>
            <td>{r['shot_id']}</td>
            <td>{r['participant_id']}</td>
            <td>{r['num_frames']}</td>
            <td>{r['duration']:.2f}</td>
            <td>{r['angle']:.2f}</td>
            <td>{r['depth']:.2f}</td>
            <td>{r['left_right']:.2f}</td>
            <td><a href="{Path(r['viewer_file']).name}" target="_blank">View</a></td>
            <td><a href="{Path(r['table_file']).name}" target="_blank">View</a></td>
            <td class="success">{r['status']}</td>
        </tr>
"""
        else:
            html += f"""
        <tr>
            <td>{r['index']}</td>
            <td colspan="9" class="failed">{r['status']}</td>
        </tr>
"""

    html += """
    </table>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Summary page created: {output_path}")


def main():
    csv_path = '../data/train.csv'

    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        sys.exit(1)

    print("Batch Visualization Tool")
    print("="*80)
    print("Examples:")
    print("  - Visualize shots 0, 1, 2: python batch_viewer.py 0 1 2")
    print("  - Visualize first 5 shots: python batch_viewer.py 0-4")
    print("  - Visualize specific shots: python batch_viewer.py 0 5 10 15 20")
    print("="*80)

    if len(sys.argv) < 2:
        print("\nUsage: python batch_viewer.py <shot_indices>")
        print("  shot_indices: space-separated list of shot indices or ranges (e.g., 0 1 2 or 0-4)")
        sys.exit(1)

    # Parse command line arguments
    shot_indices = []
    for arg in sys.argv[1:]:
        if '-' in arg:
            # Range specification
            start, end = map(int, arg.split('-'))
            shot_indices.extend(range(start, end + 1))
        else:
            # Single index
            shot_indices.append(int(arg))

    print(f"\nWill generate visualizations for {len(shot_indices)} shots: {shot_indices}")

    # Confirm
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Generate visualizations
    visualize_batch(csv_path, shot_indices)


if __name__ == '__main__':
    main()

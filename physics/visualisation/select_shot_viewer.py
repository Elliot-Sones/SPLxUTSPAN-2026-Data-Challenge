"""
Interactive shot selector for frame-by-frame visualization
Allows you to choose any shot from the dataset to visualize
"""

import pandas as pd
import numpy as np
import json
import sys
import os

# Import the main viewer
from frame_by_frame_viewer import (
    parse_array_json,
    load_shot_data,
    create_interactive_visualization,
    create_data_table
)


def list_shots(csv_path):
    """List all available shots with their metadata"""
    df = pd.read_csv(csv_path)

    print("="*80)
    print("Available Shots:")
    print("="*80)
    print(f"{'Index':<6} {'Shot ID':<12} {'Participant':<12} {'Angle':<10} {'Depth':<10} {'L/R':<10}")
    print("-"*80)

    for idx, row in df.iterrows():
        print(f"{idx:<6} {row['shot_id']:<12} {row['participant_id']:<12} "
              f"{row['angle']:<10.2f} {row['depth']:<10.2f} {row['left_right']:<10.2f}")

    print("-"*80)
    print(f"Total shots: {len(df)}")
    print("="*80)
    return len(df)


def visualize_shot(csv_path, shot_index):
    """Load and visualize a specific shot"""
    shot, keypoint_names = load_shot_data(csv_path, shot_index=shot_index)

    num_frames = len(shot[keypoint_names[0]])

    print("\n" + "="*60)
    print("Generating Visualization")
    print("="*60)
    print(f"Shot ID: {shot['shot_id']}")
    print(f"Participant ID: {shot['participant_id']}")
    print(f"Target Angle: {shot['angle']:.2f}°")
    print(f"Target Depth: {shot['depth']:.2f}\"")
    print(f"Target Left/Right: {shot['left_right']:.2f}\"")
    print(f"Total Frames: {num_frames}")
    print(f"Duration: {num_frames/60:.2f} seconds")
    print(f"Total Keypoints: {len(keypoint_names)//3}")
    print("="*60)

    # Create output filenames
    viewer_file = f'shot_{shot_index}_frame_viewer.html'
    table_file = f'shot_{shot_index}_data_table.html'

    # Create visualizations
    create_interactive_visualization(shot, keypoint_names, viewer_file)
    create_data_table(shot, keypoint_names, table_file)

    print("\nVisualization complete!")
    print(f"3D Viewer: {viewer_file}")
    print(f"Data Table: {table_file}")
    print("\nOpen these files in a web browser to explore the shot data.")

    return viewer_file, table_file


def main():
    csv_path = '../data/train.csv'

    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        print("Make sure you're running this script from the visualisation directory.")
        sys.exit(1)

    # List all available shots
    total_shots = list_shots(csv_path)

    # Get user input
    print("\nEnter the shot index to visualize (or 'q' to quit):")

    while True:
        try:
            user_input = input("> ").strip()

            if user_input.lower() == 'q':
                print("Exiting...")
                break

            shot_index = int(user_input)

            if shot_index < 0 or shot_index >= total_shots:
                print(f"Error: Please enter a number between 0 and {total_shots-1}")
                continue

            # Visualize the selected shot
            visualize_shot(csv_path, shot_index)

            # Ask if user wants to visualize another
            print("\nVisualize another shot? Enter index or 'q' to quit:")

        except ValueError:
            print("Error: Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == '__main__':
    main()

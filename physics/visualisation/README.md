# Frame-by-Frame Visualization Tool

High-precision frame-by-frame visualization for SPLxUTSPAN Data Challenge 2026 dataset.

## Features

- Interactive 3D visualization of all 72 keypoints
- Frame-by-frame navigation with slider control
- Play/Pause animation at 60 FPS
- Hover over any keypoint to see exact coordinates (6 decimal places)
- Complete skeleton rendering with anatomically correct connections
- Separate data table with all coordinate values (10 decimal places)
- Interactive shot selection
- Batch processing for multiple shots

## Quick Start

### Option 1: Visualize First Shot (Default)

```bash
cd visualisation
uv run python frame_by_frame_viewer.py
```

Generates:
- `frame_viewer.html` - Interactive 3D visualization
- `frame_data_table.html` - Complete coordinate data table

### Option 2: Interactive Shot Selection

```bash
cd visualisation
uv run python select_shot_viewer.py
```

This will:
1. Display a list of all available shots with metadata
2. Let you select which shot to visualize
3. Generate files named `shot_N_frame_viewer.html` and `shot_N_data_table.html`

### Option 3: Batch Process Multiple Shots

```bash
cd visualisation
uv run python batch_viewer.py 0 1 2 3 4
```

Or use ranges:
```bash
uv run python batch_viewer.py 0-9
```

This creates a `batch_output/` directory with:
- Individual viewer and table files for each shot
- `index.html` - Summary page with links to all visualizations

## Output Files

### frame_viewer.html
- 3D interactive visualization
- Use slider to navigate frame-by-frame
- Click "Play" to animate at 60 FPS
- Click "Pause" to stop animation
- Drag to rotate camera
- Scroll to zoom
- Hover over keypoints for exact coordinates

### frame_data_table.html
- Searchable/sortable table with all frame data
- Columns: Frame, Time (s), Keypoint, X, Y, Z
- 10 decimal place precision for all coordinates
- Filter by frame or keypoint name

## Customization

To visualize a different shot, modify the `shot_index` parameter in `main()`:

```python
shot, keypoint_names = load_shot_data(csv_path, shot_index=5)  # Load 6th shot
```

## Data Structure

Each shot contains:
- 72 keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles, toes, heels, all finger joints)
- Each keypoint has X, Y, Z coordinates
- Frame rate: 60 FPS
- Coordinate precision: float32

## Skeleton Connections

The visualization includes anatomically correct connections:
- Head (nose, eyes, ears)
- Torso (shoulders, neck, hips)
- Arms (shoulders to elbows to wrists)
- Legs (hips to knees to ankles to feet)
- Hands (all finger joints for both hands)

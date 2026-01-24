# Visualization Summary

## Generated Files

The visualization tool has been successfully created and run. The following files are ready to use:

### Interactive Visualizations

1. **frame_viewer.html** (9.8 MB)
   - 3D interactive visualization of Shot ID: qxapevM (Participant 1)
   - 240 frames (4.00 seconds at 60 FPS)
   - All 72 keypoints displayed with skeleton connections
   - Features:
     - Slider to navigate frame-by-frame
     - Play/Pause buttons for animation
     - 3D rotation and zoom with mouse
     - Hover over keypoints to see exact coordinates
     - Shows: Angle 46.31°, Depth 6.93", Left/Right 6.73"

2. **frame_data_table.html** (5.9 MB)
   - Complete data table with all coordinate values
   - Every keypoint for every frame
   - 10 decimal place precision
   - Searchable and sortable
   - Columns: Frame, Time (s), Keypoint, X, Y, Z

### Tools

Three Python scripts are available:

1. **frame_by_frame_viewer.py** (12 KB)
   - Main visualization tool
   - Generates visualizations for first shot by default
   - Can be modified to visualize any shot

2. **select_shot_viewer.py** (3.4 KB)
   - Interactive shot selector
   - Lists all 345 shots from training data
   - Allows selecting specific shots to visualize

3. **batch_viewer.py** (6.0 KB estimated)
   - Batch processing tool
   - Generate visualizations for multiple shots at once
   - Creates summary index page

## Data Specifications

### Shot Details (Default Visualization)
- **Shot ID**: qxapevM
- **Participant**: 1
- **Total Frames**: 240
- **Duration**: 4.00 seconds
- **Frame Rate**: 60 FPS
- **Target Angle**: 46.31°
- **Target Depth**: 6.93"
- **Target Left/Right**: 6.73"

### Keypoint Information
- **Total Keypoints**: 72 (69 unique named keypoints)
- **Coordinate System**: 3D (X, Y, Z)
- **Units**: Feet
- **Precision**: float32

### Keypoint Categories

1. **Head** (5): nose, left_eye, right_eye, left_ear, right_ear
2. **Torso** (7): neck, left_shoulder, right_shoulder, left_hip, right_hip, mid_hip
3. **Arms** (6): left_elbow, right_elbow, left_wrist, right_wrist, left_wrist_2, right_wrist_2
4. **Legs** (10): left_knee, right_knee, left_ankle, right_ankle, left_heel, right_heel, left_big_toe, right_big_toe, left_small_toe, right_small_toe
5. **Left Hand** (21): All finger joints including CMC, MCP, PIP, DIP, IP, and distal joints for all 5 fingers
6. **Right Hand** (21): Mirror of left hand
7. **Hand Extremities** (2): left_pinky, right_pinky, left_thumb, right_thumb (additional markers)

## Usage Instructions

### To View Current Visualization
1. Open `frame_viewer.html` in any web browser
2. Use the slider at the bottom to navigate frames
3. Click "Play" to animate the shot
4. Drag to rotate, scroll to zoom
5. Hover over red dots (keypoints) to see coordinates

### To View Data Table
1. Open `frame_data_table.html` in any web browser
2. Scroll through all frame data
3. Click column headers to sort
4. All coordinates shown with 10 decimal places

### To Visualize Different Shots

#### Method 1: Edit the Default Script
```python
# In frame_by_frame_viewer.py, line 392
shot, keypoint_names = load_shot_data(csv_path, shot_index=5)  # Change 0 to desired index
```

#### Method 2: Use Interactive Selector
```bash
cd visualisation
uv run python select_shot_viewer.py
```
Enter the shot index when prompted.

#### Method 3: Batch Process
```bash
cd visualisation
uv run python batch_viewer.py 0 1 2 3 4  # Or any shot indices
```

## Technical Details

### File Sizes
- Interactive 3D viewer: ~10 MB per shot (includes all frame data)
- Data table: ~6 MB per shot (complete coordinate list)
- Combined: ~16 MB per shot

### Performance
- Visualization generation: ~5-10 seconds per shot
- Browser rendering: Instant for frame navigation
- Animation: Smooth at 60 FPS on modern browsers

### Requirements
- Python packages: pandas, numpy, plotly, json
- All included in project environment
- No additional installation needed with `uv run`

## Visualization Features

### 3D Viewer Controls
- **Mouse Drag**: Rotate 3D view
- **Mouse Scroll**: Zoom in/out
- **Slider**: Navigate to specific frame
- **Play Button**: Animate at 60 FPS
- **Pause Button**: Stop animation
- **Hover**: Show keypoint name and coordinates

### Data Precision
- **Display Precision**: 6 decimal places (hover tooltips)
- **Table Precision**: 10 decimal places
- **Storage Precision**: float32 (6-9 significant digits)
- **Time Precision**: 6 decimal places (microsecond accuracy)

## Next Steps

1. **Explore the Default Visualization**
   - Open `frame_viewer.html` to see the interactive 3D view
   - Open `frame_data_table.html` to examine precise coordinates

2. **Compare Different Shots**
   - Use `select_shot_viewer.py` to visualize different participants
   - Compare angles, depths, and motion patterns

3. **Batch Analysis**
   - Use `batch_viewer.py` to create visualizations for multiple shots
   - Useful for comparing multiple attempts or participants

4. **Custom Analysis**
   - Modify the scripts to add custom annotations
   - Extract specific keypoint trajectories
   - Calculate velocities, accelerations, or angles

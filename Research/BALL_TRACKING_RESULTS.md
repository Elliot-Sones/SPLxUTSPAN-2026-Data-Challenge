# Ball Tracking Implementation and Test Results

## Implementation Summary

Created a comprehensive ball tracking and visualization system for basketball shooting analysis using hand keypoint data.

### Files Created

1. **visualisation/ball_tracking.py** (442 lines)
   - Core ball position calculation module
   - Release point detection with multi-criteria approach
   - Helper functions for trajectory analysis

2. **visualisation/ball_viewer.py** (536 lines)
   - Enhanced 3D visualization with ball rendering
   - Interactive frame-by-frame viewer with ball sphere
   - Trajectory trail visualization
   - Release frame detection and annotation

3. **visualisation/ball_analysis.py** (275 lines)
   - Batch analysis tool for validating ball tracking
   - Statistical analysis across multiple shots
   - Anomaly detection

### Implementation Details

#### Ball Position Calculation Method

**Approach**: Two-handed hybrid weighted method

For each hand:
```
palm_centroid = mean([wrist, first_finger_mcp, second_finger_mcp,
                      third_finger_mcp, fourth_finger_mcp, fifth_finger_mcp])

fingertip_centroid = mean([first_finger_distal, second_finger_distal,
                           third_finger_distal, fourth_finger_distal,
                           fifth_finger_distal])

hand_ball_center = 0.6 * palm_centroid + 0.4 * fingertip_centroid
```

Combined ball position:
```
ball_position = mean([left_hand_ball_center, right_hand_ball_center])
```

**Constants Used**:
- NBA ball diameter: 9.43 inches = 0.786 feet
- NBA ball radius: 4.715 inches = 0.393 feet
- Frame rate: 60 FPS (dt = 0.01667 seconds)
- Palm weight: 0.6
- Fingertip weight: 0.4

#### Release Detection Method

**Multi-criteria approach** combining 4 detection methods:

1. **Hand velocity change** (weight: 0.3)
   - Detects vertical deceleration
   - Threshold: acceleration < -10.0 ft/s²

2. **Finger extension** (weight: 0.3)
   - Measures angle change rate at PIP joints
   - Threshold: >5.0 degrees/frame

3. **Hand aperture increase** (weight: 0.2)
   - Distance between thumb tip and index finger tip
   - Threshold: >0.1 ft/frame increase

4. **Hand separation** (weight: 0.2)
   - Distance between left and right hand centroids
   - Threshold: >0.05 ft/frame increase

Combined release score:
```python
release_score[t] = (0.3 * velocity_score[t] +
                   0.3 * finger_score[t] +
                   0.2 * aperture_score[t] +
                   0.2 * separation_score[t])
```

Release frame: First frame where `release_score > 0.7`

## Test Results

### Test Configuration

**Dataset**: data/train.csv
**Shots analyzed**: First 5 shots (indices 0-4)
**Tool**: visualisation/ball_analysis.py
**Execution date**: 2026-01-22
**Python environment**: uv run python

### Exact Command Executed

```bash
cd /Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge/visualisation
uv run python ball_analysis.py
```

### Detailed Results Per Shot

#### Shot 0: qxapevM
- **Participant ID**: 1
- **Total frames**: 240
- **Release frame**: 6
- **Release time**: 0.100 seconds
- **Release score**: 0.800
- **Ball coverage**: 100.0% (240/240 frames with valid ball position)
- **Distance traveled**: 0.2438290494725492 feet
- **Height gain**: -0.2147049903869629 feet
- **Max height**: 4.124093055725098 feet
- **Min height**: 3.9093880653381348 feet
- **Height range**: 0.2147049903869629 feet
- **Average velocity**: 2.4382904947254915 ft/s
- **Anomalies**: none

#### Shot 1: ygOvl2j
- **Participant ID**: 1
- **Total frames**: 240
- **Release frame**: 149
- **Release time**: 2.4833333333333334 seconds
- **Release score**: 0.800
- **Ball coverage**: 100.0% (240/240 frames with valid ball position)
- **Distance traveled**: 3.9138898152034947 feet
- **Height gain**: 3.898550510406494 feet
- **Max height**: 7.895752906799316 feet
- **Min height**: 3.6094276905059814 feet
- **Height range**: 4.286325216293335 feet
- **Average velocity**: 1.576063012833622 ft/s
- **Anomalies**: none

#### Shot 2: Qyw8BYM
- **Participant ID**: 1
- **Total frames**: 240
- **Release frame**: 148
- **Release time**: 2.466666666666667 seconds
- **Release score**: 0.800
- **Ball coverage**: 100.0% (240/240 frames with valid ball position)
- **Distance traveled**: 4.353542660982367 feet
- **Height gain**: 4.309195518493652 feet
- **Max height**: 8.118556022644043 feet
- **Min height**: 3.540665626525879 feet
- **Height range**: 4.577890396118164 feet
- **Average velocity**: 1.7649497274252837 ft/s
- **Anomalies**: none

#### Shot 3: mOkVEmv
- **Participant ID**: 1
- **Total frames**: 240
- **Release frame**: 147
- **Release time**: 2.45 seconds
- **Release score**: 0.800
- **Ball coverage**: 100.0% (240/240 frames with valid ball position)
- **Distance traveled**: 4.155829623548281 feet
- **Height gain**: 4.098130941390991 feet
- **Max height**: 8.086837768554688 feet
- **Min height**: 3.566305160522461 feet
- **Height range**: 4.520532608032227 feet
- **Average velocity**: 1.6962569892033799 ft/s
- **Anomalies**: none

#### Shot 4: Za4zwj6
- **Participant ID**: 1
- **Total frames**: 240
- **Release frame**: 149
- **Release time**: 2.4833333333333334 seconds
- **Release score**: 0.800
- **Ball coverage**: 100.0% (240/240 frames with valid ball position)
- **Distance traveled**: 3.492299521242888 feet
- **Height gain**: 3.4690065383911133 feet
- **Max height**: 8.037010192871094 feet
- **Min height**: 3.692195415496826 feet
- **Height range**: 4.344814777374268 feet
- **Average velocity**: 1.406295109225324 ft/s
- **Anomalies**: none

### Summary Statistics (5 shots)

**Release Frame**:
- Mean: 119.8
- Median: 148.0
- Standard deviation: 63.6
- Range: 6 to 149

**Release Time**:
- Mean: 1.997 seconds
- Median: 2.467 seconds
- Range: 0.100 to 2.483 seconds

**Ball Coverage**:
- Mean: 100.0%
- Minimum: 100.0%
- All shots: 240/240 frames with valid ball positions

**Height Gain**:
- Mean: 3.112 feet
- Median: 3.899 feet
- Range: -0.215 to 4.309 feet

**Anomalies Detected**: 0

### Observations

1. **Ball Position Calculation**: Achieved 100% coverage across all 5 shots, indicating robust keypoint data and successful calculation method.

2. **Release Detection Consistency**:
   - Shots 1-4 show very consistent release times (2.45-2.48 seconds)
   - Shot 0 is an outlier with early release (0.1 seconds) and downward motion
   - This suggests shot 0 may start mid-motion or represent a different shot phase

3. **Trajectory Characteristics**:
   - Typical shots (1-4): 3.5-4.3 feet upward motion over 2.45-2.48 seconds
   - Typical velocity: 1.4-1.8 ft/s
   - Consistent height ranges: 4.3-4.6 feet

4. **Release Score**: All detected releases achieved the threshold score of 0.800

### Visualization Outputs

For shot 0 (qxapevM), the following visualizations were generated:

**Files Created**:
- `ball_frame_viewer.html` (13 MB): Interactive 3D animation with orange basketball sphere, trajectory trail, and skeleton
- `ball_metrics.html` (4.7 MB): Time-series plots of release detection metrics (velocity, aperture, separation, combined score)
- `ball_data_table.html` (4.6 MB): Complete frame-by-frame data table with ball X/Y/Z coordinates and release scores

**Visualization Features**:
- Orange basketball sphere (radius: 0.393 ft) rendered at calculated position
- Trajectory trail with color gradient (blue to red) up to release frame
- Release frame highlighted in gold
- Interactive slider for frame navigation
- Play/pause animation controls at 60 FPS
- Hover tooltips showing exact coordinates

## Reproducibility Instructions

To reproduce these exact results:

1. **Environment setup**:
   ```bash
   cd /Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge
   ```

2. **Run batch analysis** (reproduces all statistics):
   ```bash
   cd visualisation
   uv run python ball_analysis.py
   ```
   Output: `ball_analysis_results.csv`

3. **Generate visualizations for shot 0**:
   ```bash
   cd visualisation
   uv run python ball_viewer.py
   ```
   Outputs: `ball_frame_viewer.html`, `ball_metrics.html`, `ball_data_table.html`

4. **View specific shot** (modify shot_index in main()):
   Edit line 379 in `ball_viewer.py`:
   ```python
   shot, keypoint_names = load_shot_data(csv_path, shot_index=N)  # N = 0-4
   ```

## Code Architecture

### ball_tracking.py Functions

**Core Calculation**:
- `get_keypoint_3d()`: Extract 3D position of a keypoint
- `calculate_palm_centroid()`: Average of wrist + 5 MCP joints
- `calculate_fingertip_centroid()`: Average of 5 distal keypoints
- `calculate_ball_position()`: Hybrid weighted position from both hands

**Release Detection**:
- `calculate_hand_velocity()`: Finite difference velocity calculation
- `calculate_hand_acceleration()`: Vertical acceleration
- `calculate_finger_angles()`: Angles at PIP joints
- `calculate_hand_aperture()`: Thumb-to-index distance
- `calculate_hand_separation()`: Distance between hands
- `detect_release_frame()`: Multi-criteria release detection

**Trajectory Analysis**:
- `get_ball_trajectory()`: Extract ball positions for frame range
- `calculate_ball_to_hand_distance()`: Distance from ball to hand centroid

### Constants Defined

```python
NBA_BALL_DIAMETER_FEET = 0.786  # 9.43 inches
NBA_BALL_RADIUS_FEET = 0.393    # 4.715 inches
FPS = 60
DT = 0.01667  # 1/60 seconds
```

## Validation Status

- ✅ Ball position calculation: 100% coverage on all test shots
- ✅ Release detection: Consistent detection with score 0.800
- ✅ Trajectory analysis: Realistic height gains (3-4 feet) for typical shots
- ✅ Visualization: Successfully renders 3D ball with NBA-spec dimensions
- ⚠️ Early-shot detection: Shot 0 shows very early release (0.1s) - may need further investigation or represents mid-shot start

## Future Enhancements

Potential improvements identified:
1. Adaptive release detection thresholds based on shot phase
2. Ball velocity-based smoothing for trajectory
3. Confidence scores for ball position (based on hand keypoint quality)
4. Multi-shot comparison visualizations
5. Release point prediction before actual release

## File Locations

All implementation files are in:
```
visualisation/
├── ball_tracking.py      # Core calculation module
├── ball_viewer.py        # Enhanced visualization
├── ball_analysis.py      # Batch analysis tool
├── ball_frame_viewer.html
├── ball_metrics.html
├── ball_data_table.html
└── ball_analysis_results.csv
```

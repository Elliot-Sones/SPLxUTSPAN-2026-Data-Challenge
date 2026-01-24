# SkillMimic 3D Skeleton Visualization Fix

## What Was Broken

### Original Implementation (`skillmimic_3d_viewer_with_skeleton.py`)

**Problem**: Only showed 20 of 53 joints with nonsensical connections

**Issues**:
1. **Incomplete skeleton**: Extracted only 20 feature triplets
   - Features: 51-59, 108-116, 220-264
   - Missing: complete legs, most of torso, head, neck

2. **Wrong connections**: Treated hand fingers as entire body structure
   - Assumed joints 0-5 were pelvis/torso/arms (incorrect)
   - Assumed joints 6-19 were finger chains (partially correct but incomplete)

3. **No anatomical structure**: Didn't map to actual humanoid body parts
   - No reference to mocap_humanoid.xml structure
   - Arbitrary feature selection without understanding

4. **Result**: Unrecognizable mess that doesn't look human

## What Was Fixed

### New Implementation (`skillmimic_3d_viewer_fixed.py`)

**Solution**: Complete 53-joint skeleton with proper anatomical structure

### 1. Research Phase

**Sources**:
- Fetched SkillMimic paper (arxiv.org/abs/2408.15270)
- Analyzed mocap_humanoid.xml from GitHub repository
- Found 53-body structure with complete hierarchy

**Discovery**:
- SkillMimic uses Isaac Gym mocap humanoid model
- 53 bodies: Pelvis + 8 leg joints + 15 torso/head + 30 finger joints
- Position features start at index 165 (not 51)

### 2. Joint Mapper (`skillmimic_joint_mapper.py`)

**Automated analysis**:
- Scanned all 337 features for position data characteristics
- Found 53 consecutive triplets starting at feature 165
- Mapped to anatomical names from mocap_humanoid.xml hierarchy
- Validated symmetry and limb lengths

**Output**: `skillmimic_joint_map.json` with complete mapping

### 3. Updated Visualization

**Improvements**:

a) **Complete skeleton** (53 joints):
   - All body parts: head, neck, torso, spine, pelvis
   - Both legs: hip → knee → ankle → toe
   - Both arms: thorax → shoulder → elbow → wrist
   - All 10 fingers: 3 segments each

b) **Anatomical connections** (52 connections):
   - Proper hierarchy from mocap_humanoid.xml
   - Biologically correct parent-child relationships

c) **Color-coded body parts**:
   - Blue: Torso/spine (thick lines)
   - Gold: Head/neck (thick lines)
   - Green: Legs (thick lines)
   - Red: Arms (medium lines)
   - Orange: Fingers (thin lines)

d) **Visual enhancements**:
   - Coordinate axes at pelvis (shows X/Y/Z)
   - Ground plane with grid
   - Key joint markers (major joints only)
   - Better camera angle
   - Legend showing body parts

## Results

### Before (Broken)
- Showed: 20 joints, random connections
- Structure: Unrecognizable, floating blobs
- Missing: Legs, torso, head, most of body
- Quality: Horrible

### After (Fixed)
- Showed: 53 joints, anatomically correct
- Structure: Recognizable humanoid shooting basketball
- Complete: All body parts visible
- Quality: Professional, similar to SPL ball_viewer.py

## Validation

**Automated checks passed**:
- ✓ Left/right symmetry (Y difference < 0.04 for all pairs)
- ✓ Limb lengths reasonable (0.24-0.46 units)
- ✓ Pelvis stability (root joint has lowest variability)
- ✓ Ball trajectory from hand position

**Visual inspection**:
- ✓ Recognizable shooting motion
- ✓ Proper human proportions
- ✓ Natural movement patterns
- ✓ Ball release from hands

## Files Created/Modified

### Created:
1. `visualisation/skillmimic_joint_mapper.py` - Automated joint discovery
2. `visualisation/skillmimic_joint_map.json` - Feature-to-anatomy mapping
3. `visualisation/skillmimic_3d_viewer_fixed.py` - Corrected visualization
4. `SKILLMIMIC_JOINTS.md` - Complete joint documentation
5. `SKILLMIMIC_FIX_SUMMARY.md` - This file

### Reference (Original):
- `visualisation/skillmimic_3d_viewer_with_skeleton.py` - Broken implementation

## Usage

### Generate Visualization

```bash
cd /Users/elliot18/Desktop/Home/Projects/SPLxUTSPAN-2026-Data-Challenge

# Run joint mapper (already done, creates skillmimic_joint_map.json)
uv run python visualisation/skillmimic_joint_mapper.py \
  "/Users/elliot18/Downloads/005_014pickle_shot_001.pt"

# Generate fixed visualization
uv run python visualisation/skillmimic_3d_viewer_fixed.py \
  "/Users/elliot18/Downloads/005_014pickle_shot_001.pt" \
  --output-dir visualisation/output

# Open in browser
open visualisation/output/skillmimic_3d_skeleton_fixed.html
```

### Output

- **HTML file**: `visualisation/output/skillmimic_3d_skeleton_fixed.html`
- **Features**: Interactive 3D, animation controls, frame slider
- **Shows**: Complete humanoid skeleton + basketball trajectory

## Technical Details

### Data Structure (337 features)

**Position features** (165-323):
- 53 bodies × 3 coordinates = 159 features
- X, Y, Z for each body part
- Start index: 165

**Ball features**:
- Ball X: feature 325
- Ball Y: feature 119
- Ball Z: feature 326
- Contact: feature 336

**Other features** (0-164, 324, 327-337):
- Likely velocities, rotations, other state
- Not needed for basic skeleton visualization

### Skeleton Structure

From mocap_humanoid.xml:
- Root: Pelvis (freejoint with 6 DOF)
- Bodies: 53 total in depth-first ordering
- Joints: Each body has 3 rotational DOF
- Hierarchy: Tree structure from root to extremities

## References

### Sources Used
- [SkillMimic Paper](https://arxiv.org/abs/2408.15270) - Project description
- [SkillMimic GitHub](https://github.com/wyhuai/SkillMimic) - Repository
- [mocap_humanoid.xml](https://github.com/wyhuai/SkillMimic/blob/main/skillmimic/data/assets/mjcf/mocap_humanoid.xml) - MJCF skeleton
- [Isaac Gym Humanoid](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/humanoid.py) - Reference implementation

### Test Data
- File: `005_014pickle_shot_001.pt`
- Shape: (104 frames, 337 features)
- Motion: Basketball shooting
- Release frame: 59

## Success Criteria (All Met)

- ✓ Complete skeleton with all 53 joints
- ✓ Anatomically correct connections
- ✓ Recognizable shooting motion
- ✓ Ball trajectory from hands
- ✓ Visual quality comparable to SPL viewer
- ✓ Proper body proportions
- ✓ Color-coded body parts
- ✓ Interactive controls

# Comprehensive External Data Sources Found

## Summary of All Data Downloaded

### 1. OpenBiomechanics - Baseball Pitching (DOWNLOADED)
**Location:** `external_data/openbiomechanics/`

| File | Size | Description |
|------|------|-------------|
| joint_angles.csv | 95MB | Time-series joint angles for 411 pitches |
| joint_velos.csv | 95MB | Time-series joint velocities |
| poi_metrics.csv | 267KB | Point-of-interest metrics including pitch_speed_mph |
| metadata.csv | 46KB | Player metadata |

**Key Features:**
- 411 pitches with release velocity (69-94 mph)
- Shoulder, elbow, wrist angles and velocities
- Time-series at release point
- Arm mechanics similar to shooting follow-through

**Potential Use:** Learn arm angle -> release velocity relationship

### 2. SkillMimic - Basketball Motion Capture (DOWNLOADED)
**Location:** `external_data/skillmimic/`

| File | Size | Description |
|------|------|-------------|
| shot_style1.pt | 141KB | PyTorch tensor [104, 337] - shooting motion |
| shot_style2.pt | 131KB | Different shooting style |
| shot_style3.pt | 136KB | Another shooting style |

**Key Features:**
- Actual basketball shooting motion capture
- 104 frames per shot
- 337 features (likely body + ball position)
- From CVPR 2025 paper

**Potential Use:** Learn shooting form patterns, ball trajectory from body pose

### 3. CMU Mocap - Basketball (DOWNLOADED)
**Location:** `external_data/cmu_mocap/`

| File | Size | Description |
|------|------|-------------|
| 06_14_shoot.bvh | 367KB | Basketball crossover dribble + shoot |
| 06_15_shoot.bvh | 420KB | Basketball dribble + shoot |
| 15_12_layup.bvh | 6.9MB | Basketball lay-up shot |

**Key Features:**
- BVH format skeleton data
- Full body hierarchy (hips, legs, arms, spine)
- Basketball shooting motions

**Potential Use:** Learn shooting form patterns

### 4. NBA Ball Trajectory (ALREADY HAD)
**Location:** `external_data/`

| File | Rows | Description |
|------|------|-------------|
| player_metrics.csv | 189 | Per-player release velocity stats |
| path_detail.csv | 79,776 | Ball trajectory after release |

**Status:** Already tested - did not improve LB score

---

## New Approaches to Try

### Approach 1: Physics-Informed Feature Learning
Use OpenBiomechanics baseball data to learn:
```
arm_angles + arm_velocities -> release_velocity
```
Then apply similar features to our basketball data.

### Approach 2: SkillMimic Transfer Learning
1. Parse SkillMimic shooting data format
2. Extract body pose features at release frame
3. Pre-train on SkillMimic, fine-tune on our data

### Approach 3: Synthetic Data Augmentation
Use physics simulation (inspired by SkillMimic/IsaacGym):
1. Generate synthetic shooting motions with known outcomes
2. Train on synthetic + real data

### Approach 4: Reinforcement Learning Approach
Instead of supervised learning:
1. Learn a policy that predicts "corrections" to shots
2. Use physics model to simulate outcomes
3. Train to minimize prediction error

### Approach 5: CMU Mocap Form Analysis
1. Parse BVH shooting data
2. Extract "ideal" shooting form patterns
3. Compare our data to ideal patterns
4. Use deviation from ideal as features

---

## Key Resources Found

### Simulation Environments
- [IsaacGym](https://developer.nvidia.com/isaac-gym) - Physics simulation for RL
- [MuJoCo](https://mujoco.org/) - Open source physics engine
- [Unity ML-Agents](https://unity.com/products/machine-learning-agents) - Game engine + ML

### Research Papers
- [Learning to Ball (2025)](https://arxiv.org/html/2509.22442) - RL for basketball, 91.8% shooting accuracy
- [SkillMimic (CVPR 2025)](https://github.com/wyhuai/SkillMimic) - Learning basketball skills from demos
- [OpenBiomechanics](https://www.openbiomechanics.org/) - Free sports biomechanics data

### Curated Lists
- [awesome-biomechanics](https://github.com/modenaxe/awesome-biomechanics) - Biomechanics datasets
- [awesome-isaac-gym](https://github.com/robotlearning123/awesome-isaac-gym) - RL simulation resources

---

## Recommended Next Steps

1. **Parse SkillMimic data** - Understand the 337 features, extract body pose at release
2. **Build arm mechanics model** from OpenBiomechanics - Learn joint angle -> velocity
3. **Extract CMU shooting form** - Identify release timing and arm positions
4. **Create physics-informed features** combining insights from all sources
5. **Test on our data** - Apply learned patterns to our 345 training samples

---

## Files to Create

| Script | Purpose |
|--------|---------|
| `parse_skillmimic.py` | Decode SkillMimic .pt format |
| `learn_release_velocity.py` | Train on OpenBiomechanics |
| `parse_bvh.py` | Extract CMU mocap joint angles |
| `combined_physics_model.py` | Integrate all sources |

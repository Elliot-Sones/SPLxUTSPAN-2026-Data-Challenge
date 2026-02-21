# Extended Physics Features Results

Date: 2026-02-15
Script: scripts/extended_physics_features.py
Anchor: Sub 2475 (LB 0.006485)

## Concept

Expand the kinetic chain approach (KC-PLS, proven in Sub 2475) to include 46 previously
unused hand/finger keypoints. Three new sub-chains:
1. Right hand (shooting hand): 53 features - fingertip positions/velocities, spread, wrist flexion, finger curl
2. Left hand (guide hand): 21 features - separation timing/distance, relative position, aim alignment
3. Torso/spine: 17 features - shoulder rotation, trunk lean, body alignment

All sub-chains compressed via PLS (3 components each) before integration.

## Baseline

KC-PLS only (no new features): mean LOO = 0.005302
(Note: this uses single-frame, not our best multi-frame pipeline)

## Ablation Results

| Config | Mean LOO | Delta vs Baseline | Angle | Depth | LR |
|--------|----------|-------------------|-------|-------|----|
| KC-PLS only (baseline) | 0.005302 | - | 0.005028 | 0.005135 | 0.005741 |
| + right_hand | 0.005202 | -1.88% | 0.004727 (-6.0%) | 0.005097 (-0.7%) | 0.005782 (+0.7%) |
| + left_hand | 0.005254 | -0.90% | 0.004773 (-5.1%) | 0.005135 (0.0%) | 0.005853 (+1.9%) |
| + torso | 0.005364 | +1.18% | 0.005010 (-0.4%) | 0.005270 (+2.6%) | 0.005813 (+1.3%) |
| + ALL | 0.005215 | -1.63% | 0.004415 (-12.2%) | 0.005298 (+3.2%) | 0.005932 (+3.3%) |

## Key Findings

1. **Right hand features are the clear winner** (-1.88% mean, best individual)
   - Angle benefits most (-6.0%) - fingertip kinematics predict shot arc
   - Depth slightly better (-0.7%)
   - LR marginally worse (+0.7%)

2. **Left hand adds modest signal** (-0.90%) concentrated in angle
   - Guide hand position/separation does contain information
   - But the PLS compression may mix useful/noisy features

3. **Torso features HURT** (+1.18%) - adds noise, no useful signal
   - Shoulder rotation and trunk lean don't predict outcomes well
   - These features are already partially captured by existing shoulder/hip features

4. **Combined "all" is worse than right_hand alone** (-1.63% vs -1.88%)
   - Torso noise offsets left_hand gain
   - Best strategy: right_hand only

5. **Angle is the primary beneficiary across all sub-chains**
   - This makes physical sense: hand/finger features directly control ball release

## Submissions Generated

| Sub | Description |
|-----|-------------|
| 2492 | Extended physics ALL standalone (LOO 0.005215) |
| 2493 | 10% ALL physics + 90% Sub 2475 |
| 2494 | 20% ALL physics + 80% Sub 2475 |
| 2495 | 30% ALL physics + 70% Sub 2475 |
| 2496 | 50% ALL physics + 50% Sub 2475 |
| 2497 | 10% ALL physics ANGLE ONLY + Sub 2475 |
| 2498 | 20% ALL physics ANGLE ONLY + Sub 2475 |
| 2499 | 30% ALL physics ANGLE ONLY + Sub 2475 |
| 2500 | 10% ALL physics ANGLE+DEPTH + Sub 2475 |
| 2501 | 20% ALL physics ANGLE+DEPTH + Sub 2475 |
| 2502 | right_hand only standalone (LOO 0.005202) |
| 2503 | 10% right_hand + 90% Sub 2475 |
| 2504 | 20% right_hand + 80% Sub 2475 |
| 2505 | 30% right_hand + 70% Sub 2475 |
| 2506 | 10% right_hand ANGLE ONLY + Sub 2475 |
| 2507 | 20% right_hand ANGLE ONLY + Sub 2475 |

## Top Candidates for LB Testing

1. **Sub 2503**: 10% right_hand all targets + Sub 2475 - safest bet, uses best sub-chain
2. **Sub 2506**: 10% right_hand ANGLE ONLY + Sub 2475 - targets only where signal is strongest
3. **Sub 2493**: 10% all physics + Sub 2475 - uses all sub-chains, moderate risk
4. **Sub 2497**: 10% all physics ANGLE ONLY + Sub 2475 - strongest LOO angle signal with minimal risk

## Assessment

MODERATE SIGNAL expected. The right_hand features capture genuine physics (fingertip
kinematics at release directly determine ball trajectory) and the improvement is concentrated
in angle - our biggest overfit target (2.57x ratio). However:
- LOO improvement is modest (-1.88%)
- At 10% blend weight, the actual perturbation is small
- The angle overfit problem means LOO improvements may not fully transfer to LB

Still better signal than the ElasticNet experiment (which was mostly noise).

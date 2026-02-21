# Physics-Based Ball Release Research

## Research Summary

This document summarizes research on physics-based ball release simulation for basketball free throws.

## Key Finding: The Backforce Mechanism (Hore & Watts)

The most important finding comes from biomechanics research by Hore & Watts (2011):

**"Skilled throwers use physics to time ball release to the nearest millisecond"**

Key insights:
1. Ball release timing is NOT actively controlled by the CNS at the millisecond level
2. The CNS controls finger stiffness/force, physics handles the release timing
3. Release occurs when hand acceleration produces a "backforce" that overcomes finger grip

### Two-Phase Model:
- **Phase 1 (CNS-controlled)**: Brain progressively increases finger flexor torque to counteract increasing backforce from ball acceleration
- **Phase 2 (Physics-driven)**: When hand deceleration begins, backforce overcomes finger stiffness, ball rolls off fingers

## Our Data Analysis

From shot 0 analysis:
- Peak hand speed: 3.38 m/s (occurs 39 frames before release)
- Hand speed at release: 1.10 m/s (hand is decelerating)
- Required ball speed: ~7 m/s to reach hoop

**Key Insight**: Hand motion alone cannot propel the ball to the hoop. The "finger push" (finger extension during release) adds approximately 3-4 m/s.

## Implementation Approaches

### 1. PHYSICS Mode (Pure Hand Velocity)
- Ball velocity = hand velocity at release moment
- Result: ~1.1 m/s (insufficient to reach hoop)
- Use case: Analysis of actual hand kinematics

### 2. FINGERPUSH Mode (Hand + Finger Extension)
- Ball velocity = hand velocity + finger push velocity
- Finger push direction: wrist-to-ball vector (natural finger extension)
- Finger push speed: estimated from trajectory requirements
- Result: ~5-6 m/s
- Use case: More physically realistic model

### 3. HOLD Mode (Hybrid - Direction + Computed Speed)
- Direction from hand motion
- Speed computed from projectile equations
- Result: ~7 m/s (reaches hoop accurately)
- Use case: Accurate trajectory prediction

## Sources

### Primary Research
- [Hore & Watts (2011) - Skilled throwers use physics](https://www.researchgate.net/publication/51508273_Skilled_throwers_use_physics_to_time_ball_release_to_the_nearest_millisecond)
- [Timing of finger opening in overarm throws](https://link.springer.com/article/10.1007/BF00231714)

### Physics Simulation
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MuJoCo Contact Model](https://mujoco.readthedocs.io/en/stable/computation/index.html)
- [MyoSuite Hand Models](https://github.com/myohub/myosuite)

### Related Work
- [OpenSim Pitching Analysis](https://www.drivelinebaseball.com/2017/03/computed-muscle-control-analysis-pitching-mechanics/)
- [TossingBot](https://arxiv.org/pdf/1903.11239) - Residual physics for throwing
- [OpenAI Dexterous Manipulation](https://journals.sagepub.com/doi/full/10.1177/0278364919887447)

## Limitations

1. **Mocap bodies are kinematic**: MuJoCo mocap bodies cannot exert true contact forces
2. **No finger actuation model**: Would need muscle-driven finger model for true finger push simulation
3. **Simplified finger extension**: Current model estimates finger push rather than simulating it

## Recommendation

For feature extraction (our ML use case):
- Use **HOLD mode** (hybrid) for accurate trajectory prediction
- Extract release angle from hand motion (30-50 degrees typical)
- Extract hand speed profile for biomechanical features
- The physics-derived features (release velocity, angle, position) are valid without true contact simulation

For true physics simulation (future work):
- Would require full muscle-actuated hand model (like MyoSuite)
- Or dynamic finger bodies with joint actuators
- Contact force tuning with solref/solimp parameters

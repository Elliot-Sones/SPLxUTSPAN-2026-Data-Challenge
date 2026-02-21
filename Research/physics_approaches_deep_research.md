# Physics-Based Approaches for Ball Release Parameter Estimation - Deep Research

Date: 2026-02-07

## Context

We have 345 training basketball free throw shots with:
- 69 body keypoints (including detailed finger joints) at 60 Hz, 240 frames per shot
- NO ball tracking data
- Known outcomes: angle, depth, left_right at the hoop plane
- Need to predict outcomes for 113 test shots
- Current best LB: 0.007556 using per-player tree ensembles on raw keypoint features
- Previous MuJoCo approach failed: mocap bodies cannot exert contact forces, hand velocity (3-4 m/s) is much lower than required ball velocity (7-8 m/s), simulation diverges

---

## 1. Differentiable Physics Engines

### 1.1 Overview of Available Engines

**MuJoCo MJX (MuJoCo XLA)**
- JAX-native reimplementation of MuJoCo with automatic differentiation
- Supports GPU-parallelized gradient computation through contact dynamics
- Recent work (DiffMJX) addresses gradient oscillation from stiff contacts using adaptive ODE integrators
- Available via `pip install mujoco-mjx`
- Reference: [MuJoCo MJX docs](https://mujoco.readthedocs.io/en/stable/mjx.html)

**Nimble (nimblephysics)**
- Stanford-developed differentiable fork of DART physics engine
- Analytical gradients through LCP contact formulation
- Built specifically for biomechanics: supports OpenSim models, inverse dynamics, body scaling
- Python API via `pip install nimblephysics`
- Reference: [Nimble Physics](https://nimblephysics.org/), [GitHub](https://github.com/keenon/nimblephysics)

**Brax**
- Google's differentiable physics in JAX for RL and robotics
- Fast parallel simulation but focused on robotics, not biomechanics
- Reference: [Brax GitHub](https://github.com/google/brax), [Paper](https://arxiv.org/abs/2106.13281)

**DiffTaichi**
- Source-code-transform differentiable simulator supporting fluid and solid mechanics
- More general-purpose, less biomechanics-focused
- Reference: [DiffTaichi GitHub](https://github.com/taichi-dev/difftaichi)

**Drake**
- MIT/TRI toolbox for robot dynamics and trajectory optimization
- Strong trajectory optimization and contact-implicit capabilities
- Python bindings, but steep learning curve
- Reference: [Drake](https://drake.mit.edu/)

**JaxSim**
- Differentiable physics in JAX for robotics
- Full support for forward/reverse mode AD of rigid body dynamics algorithms
- Reference: [JaxSim GitHub](https://github.com/ami-iit/jaxsim)

### 1.2 Can They Solve Our Inverse Problem?

The inverse problem: Given observed skeleton motion -> estimate ball velocity at release.

**Theoretical feasibility**: YES, differentiable physics can solve this by:
1. Define a differentiable forward model: skeleton pose -> joint torques -> ball velocity
2. Optimize parameters (contact stiffness, friction, finger force) to minimize error between predicted and observed outcomes

**Practical feasibility for our problem**: LOW-MEDIUM
- We have no ball data to constrain the ball state during simulation
- The "ball" must be inferred from hand geometry alone
- 345 samples is very small for optimization-heavy approaches
- The finger-ball interaction is the hardest part to model and we only have keypoint positions

**Best candidate**: Nimble, because it was built for biomechanics and has analytical gradients through contact. MJX is second choice for GPU acceleration.

### 1.3 Realistic Assessment

The differentiable physics approach could theoretically solve for ball release velocity by:
1. Parameterizing the ball-hand contact model
2. Running differentiable simulation forward
3. Backpropagating from trajectory error to release parameters

However, the fundamental challenge remains: **we have no ball observation during the shot**. We only see where it ends up (angle, depth, left_right). This means we would need to:
- Forward simulate: skeleton -> contact forces -> ball release -> projectile trajectory -> hoop crossing
- Then differentiate through this entire chain

This is a very long computational graph with many local minima. With 345 samples and 60 Hz data, convergence is not guaranteed.

**Verdict**: Theoretically sound but practically risky. High engineering effort for uncertain payoff.

---

## 2. MyoSuite / MyoHand

### 2.1 Overview

MyoSuite is a MuJoCo-based suite for musculoskeletal motor control. The MyoHand model has:
- 23 joints, 39 muscles
- Detailed tendon routing and muscle force optimization
- Contact-rich manipulation tasks (grasping, reorienting objects)
- Reference: [MyoSuite GitHub](https://github.com/MyoHub/myosuite), [Paper](https://arxiv.org/abs/2205.13600)

### 2.2 Can It Estimate Forces from Finger Positions?

**In principle**: Yes. MyoHand can run inverse dynamics to compute the muscle activations needed to produce observed finger motions. From these activations, contact forces on a held ball can be estimated.

**In practice**: This is extremely hard for our case because:
1. MyoHand expects EMG or activation signals, not just kinematic positions
2. Our keypoint data has only 5 finger joints per hand (thumb CMC/MCP/IP/distal, index-pinky MCP/PIP/DIP/distal) - MyoHand has 23 DOF
3. The mapping from our sparse keypoints to MyoHand's 23 DOF is underdetermined
4. Contact force estimation requires accurate contact geometry - we do not know the ball's position relative to the fingers

### 2.3 Realistic Assessment

MyoHand is designed for the forward problem (given muscle activations, compute motion) and RL-based control (learn policies to achieve goals). Using it for the inverse problem (observed positions -> forces) requires:
- Inverse kinematics fitting of 23 DOF from ~16 finger keypoints (underdetermined)
- Muscle redundancy resolution (39 muscles for 23 DOF - many solutions)
- Contact geometry estimation without ball tracking

**Verdict**: Too complex for our use case. The engineering effort would be enormous and the hand model would need to be custom-fitted to each player. Not feasible for 345 samples.

---

## 3. Physics-Informed Machine Learning

### 3.1 Relevant Approaches

**Physics-Informed Neural Networks (PINNs)**
- Add physics loss terms (ODE/PDE constraints) to neural network training
- Loss = lambda_data * L_data + lambda_ode * L_ode + lambda_ic * L_ic
- For projectile motion: constrain that predicted trajectory follows parabolic path under gravity
- Reference: [PINN for projectile](https://github.com/okada39/pinn_projectile)

**Universal Differential Equations (UDEs)**
- Replace unknown terms in differential equations with neural networks
- "A Universal Model Combining Differential Equations and Neural Networks for Ball Trajectory Prediction" (arxiv 2503.18584)
- Key claim: works with "minimal training data of only a few dozen samples"
- Reference: [Paper](https://arxiv.org/abs/2503.18584)

**Gray-Box / Hybrid Models**
- Combine physical model output with data-driven residual model
- Physical model provides interpretable baseline, NN captures residual
- Shown to improve sample efficiency compared to pure data-driven approaches
- Reference: [Deep Grey-Box Modeling](https://proceedings.mlr.press/v206/takeishi23a/takeishi23a.pdf)

### 3.2 Application to Our Problem

The most promising PIML approach for our specific problem:

**Physics-Constrained Feature Augmentation**:
1. Use physics to compute "ideal" ball release parameters from observed kinematics
2. These become features for the ML model, not direct predictions
3. The ML model learns the residual between physics estimate and actual outcome

This is actually what we already partially have with our physics_features.py module. The question is whether we can make the physics features MORE accurate.

**Projectile Physics as Soft Constraint**:
1. Train model to predict release velocity components (vx, vy, vz) and release position
2. Add physics loss: the predicted velocity + position must produce a trajectory that crosses the hoop plane at approximately the right angle/depth/left_right
3. This constrains the model to produce physically plausible predictions

### 3.3 Realistic Assessment

**Feasibility**: HIGH for feature augmentation, MEDIUM for soft constraints

The physics features already show r=0.64 correlation with angle. The gap is in depth (r=0.14) and left_right (r=0.13). Physics constraints might help with angle but are unlikely to help with depth/left_right since those depend on fine aim that is hard to extract from kinematics.

**Key insight from UDE paper**: Using physics equations with neural network parameter inference can work with very small datasets. This is directly relevant - we could formulate the problem as:

```
trajectory = physics_model(release_pos, release_vel, spin)
release_vel = f(skeleton_features)  # learned mapping
```

And train f() end-to-end with physics_model as a differentiable layer.

**Verdict**: Most promising direction. Feature augmentation is already partially implemented. The UDE/gray-box approach of learning release parameters through differentiable trajectory simulation is worth pursuing.

---

## 4. Optimal Control / Trajectory Optimization

### 4.1 Formulation

"Find the ball trajectory that best explains the observed hand motion":

```
minimize ||predicted_outcome - actual_outcome||^2
subject to:
  trajectory follows projectile equations after release
  ball is in contact with hand before release
  release frame is between frames t1 and t2
  release velocity is consistent with hand kinematics
```

This is a constrained optimization problem that can be solved with:
- Direct transcription (discretize time, optimize all variables simultaneously)
- Shooting methods (optimize initial conditions, simulate forward)
- Contact-implicit methods (optimize through contact/release events)

### 4.2 Contact-Implicit Trajectory Optimization

Recent work makes this more tractable:
- [Variational Contact-Implicit Trajectory Optimization](https://link.springer.com/chapter/10.1007/978-3-030-28619-4_66) - Manchester & Kuindersma
- [Contact-implicit MPC](https://journals.sagepub.com/doi/10.1177/02783649241273645) - discovers contact modes automatically
- [Inverse dynamics trajectory optimization for contact-implicit MPC](https://journals.sagepub.com/doi/abs/10.1177/02783649251344635) - Kurtz et al. 2025

The key advantage: no need to explicitly specify when contact breaks (release frame). The optimization discovers it.

### 4.3 Realistic Assessment

**Feasibility**: LOW for full trajectory optimization, MEDIUM for simplified version

Full trajectory optimization through contact is computationally expensive and requires:
- Accurate contact model (we do not have ball geometry)
- Good initial guess (hard without ball data)
- Per-shot optimization (345+ runs)

Simplified version: Fix release frame from kinematics, optimize only release velocity to minimize trajectory error. This is essentially what the HOLD mode in our current simulator does, but we could make it more principled.

**Verdict**: Full trajectory optimization is overkill for our problem. A simplified "find best release velocity given release frame and release position" optimization is already implemented and working. Marginal improvement possible by jointly optimizing release frame + velocity.

---

## 5. Basketball Shot Biomechanics Research

### 5.1 Key Findings on Ball Release Parameters

**Kinetic Chain and Energy Transfer**:
- Proximal-to-distal sequencing: shoulder -> elbow -> wrist -> fingers
- Shoulder rotation contributes to vertical velocity component
- Elbow and wrist actions produce horizontal velocity and backspin
- The angular velocity of the wrist is "largely derived from the angular velocities of the shoulder and the elbow"
- Reference: [Arm Joint Coordination in Basketball](https://pmc.ncbi.nlm.nih.gov/articles/PMC12121896/)

**Ball Release Velocity Components**:
- 67% of ball velocity at release explained by elbow extension + shoulder internal rotation
- Wrist flexion contributes small amount to velocity but important for accuracy
- MP joint (finger) torque contributes to stable ball release rather than velocity adjustment
- Reference: [Kinetic Analysis of Fingers in Throwing](https://journals.humankinetics.com/view/journals/mcj/26/2/article-p226.xml)

**Release Accuracy Determinants**:
- Accuracy primarily depends on ability to control release velocity deviation (not mean velocity)
- Release velocity standard deviation is the primary correlator with shooting percentage
- Higher coordination variability near release correlates with higher accuracy
- Reference: [Intra-Individual Release Variability](https://pmc.ncbi.nlm.nih.gov/articles/PMC8256521/)

**Markerless Motion Capture for Free Throws**:
- 2023 study used 9 cameras at 120 Hz for markerless free throw analysis
- Proficient shooters had lower knee/center-of-mass angular velocities
- Proficient shooters had greater release height and less forward trunk lean
- Reference: [Biomechanical characteristics of proficient free-throw shooters](https://pmc.ncbi.nlm.nih.gov/articles/PMC10436204/)

### 5.2 Estimation Without Ball Tracking

The biomechanics literature consistently uses ball tracking for release parameter estimation. There is NO established method for estimating ball release velocity purely from body kinematics.

The closest analog is baseball pitching research:
- Gradient boosting predicted pitch velocity with RMSE 0.34 mph using 16 kinematic/kinetic predictors
- Most important features: max elbow extension velocity (19.3%), max humeral rotation velocity (9.6%), trunk forward flexion (7.9%)
- Reference: [ML prediction of fastball velocity](https://www.sciencedirect.com/science/article/abs/pii/S0021929022000550)

### 5.3 Implications for Our Problem

Key biomechanical features we should extract (many already in our physics_features.py):
1. **Elbow extension angular velocity** at release (strongest predictor of ball speed)
2. **Shoulder flexion angular velocity** at release
3. **Wrist flexion angular velocity** at release
4. **Release height** relative to player height
5. **Trunk forward lean** at release
6. **Knee angle** at release (energy generation)
7. **Center of mass velocity** (whole-body contribution)
8. **Forearm angle** from vertical at release
9. **Coordination variability** (standard deviation of joint angles across the motion)
10. **Timing features**: time from peak velocity to release, temporal ordering of joint peaks

**Critical insight**: The baseball pitching research shows that ML on biomechanical features CAN predict ball velocity from body kinematics. The 0.34 mph RMSE with gradient boosting is remarkably good. Our problem is analogous but harder because basketball free throws have much less velocity variation.

---

## 6. Contact-Implicit Trajectory Optimization

### 6.1 State of the Art

**Todorov's Contact Model (MuJoCo)**:
- Convex, smooth, and invertible contact model
- Defines contact dynamics in both forward and inverse directions
- Supports efficient gradient computation through contact
- Reference: [Todorov contact model](https://www.semanticscholar.org/paper/A-convex,-smooth-and-invertible-contact-model-for-Todorov/3e677b1fa07d66f539f8086046779190694327eb)

**DiffMJX (2025)**:
- Augments MuJoCo MJX with adaptive ODE integrator for smooth gradients through contact
- Addresses gradient oscillation from stiff contacts
- Backpropagation through checkpointed tape or adjoint sensitivity
- Reference: [Hard Contacts with Soft Gradients](https://arxiv.org/html/2506.14186v1)

### 6.2 Application to Our Problem

The contact-implicit approach would avoid needing to explicitly specify the release frame. Instead:
1. Model hand-ball contact as a compliant constraint
2. As the hand decelerates, contact force drops below threshold
3. Ball separates naturally when physics dictates

This is elegant but requires knowing the ball's position relative to the hand, which we do not have.

### 6.3 Realistic Assessment

**Verdict**: The theory is beautiful but impractical for our case. Without ball position data, we cannot set up the contact problem. We know where the fingers are, but not where the ball center is relative to the fingers. And the ball radius (NBA basketball = 4.7 inches) means the contact point geometry matters significantly.

---

## 7. Markerless Mocap + Physics

### 7.1 Key Papers

**"Differentiable Biomechanics Unlocks Opportunities for Markerless Motion Capture" (Cotton, 2024)**
- Uses differentiable physics to fit inverse kinematics to markerless mocap data
- Scales body model to individual anthropomorphics end-to-end
- Implicit trajectory representation propagated through differentiable forward kinematics
- Reference: [arxiv 2402.17192](https://arxiv.org/abs/2402.17192)

**"Differentiable Biomechanics for Markerless Motion Capture in Upper Limb Stroke Rehabilitation" (2024)**
- Combines differentiable biomechanics with markerless mocap
- High agreement with optical motion capture: 2-5 degrees for joint angles, 0.04 m/s for end-effector velocity
- Reference: [arxiv 2411.14992](https://arxiv.org/abs/2411.14992)

**"Biomechanical Reconstruction with Confidence Intervals" (2025)**
- Extends differentiable biomechanics with uncertainty estimation
- Confidence intervals within 10-15 mm spatial error for virtual markers
- Reference: [arxiv 2502.06486](https://arxiv.org/html/2502.06486)

### 7.2 Application to Our Problem

The differentiable biomechanics pipeline could:
1. Fit a skeletal model to our 69 keypoints
2. Compute joint angles and angular velocities with biomechanical constraints
3. Run inverse dynamics to estimate joint torques
4. Use torques + kinematics as features for prediction

This is a more principled version of what our physics_features.py already does, but with:
- Proper skeletal model scaling per player
- Biomechanically constrained joint angles (no hyperextension, etc.)
- Smooth trajectories through physics-based filtering
- Potentially more accurate velocity estimates

### 7.3 Realistic Assessment

**Feasibility**: MEDIUM-HIGH

Using Nimble + AddBiomechanics pipeline:
1. Convert our keypoints to "virtual markers" on a skeletal model
2. Run inverse kinematics to get joint angles
3. Run inverse dynamics to get joint torques
4. Extract features from the optimized trajectories

This is the most principled approach and has been validated in recent papers. The main challenges:
- Mapping our 69 keypoints to a standard biomechanical model
- Our data is at 60 Hz (adequate for Nimble, which supports real-time)
- Engineering effort is moderate (Nimble has Python API)

**Verdict**: This is the most promising physics-based approach. It produces cleaner, more biomechanically valid features from the same keypoint data. Worth implementing.

---

## 8. Specific Tools and Libraries

### 8.1 Most Relevant Tools

**Nimble Physics** (HIGHEST PRIORITY)
- Differentiable physics engine built for biomechanics
- Supports OpenSim model format
- Inverse kinematics + inverse dynamics from marker data
- Python API: `pip install nimblephysics`
- Used by AddBiomechanics (300+ researchers, 14,000+ motion files)
- GitHub: https://github.com/keenon/nimblephysics

**Pose2Sim** (HIGH PRIORITY)
- Markerless mocap to OpenSim pipeline
- Handles 2D pose -> 3D triangulation -> OpenSim IK
- But: designed for multi-camera input, our data is already 3D
- Could use the OpenSim IK portion for biomechanical fitting
- GitHub: https://github.com/perfanalytics/pose2sim

**AddBiomechanics** (MEDIUM PRIORITY)
- Cloud-based tool for automated biomechanical analysis
- Uploads marker trajectories, returns scaled model + IK + ID
- 3-5 minutes for kinematics, 30 minutes for full dynamics
- Web: https://addbiomechanics.org/

**OpenSim** (MEDIUM PRIORITY)
- Gold standard for musculoskeletal simulation
- Can run inverse kinematics, inverse dynamics, static optimization
- Python API available but complex setup
- Could be used standalone or through Nimble

**MJINX** (LOW PRIORITY)
- JAX-based numerical inverse kinematics for MuJoCo
- Differentiable objectives for IK
- GitHub: https://github.com/based-robotics/mjinx

### 8.2 Tool Comparison for Our Use Case

| Tool | IK from keypoints | ID (torques) | Differentiable | Python API | Effort |
|------|-------------------|--------------|----------------|------------|--------|
| Nimble | Yes | Yes | Yes | Yes | Medium |
| Pose2Sim | Yes (needs adaptation) | Via OpenSim | No | Yes | Medium |
| AddBiomechanics | Yes (cloud) | Yes | No | Limited | Low |
| OpenSim | Yes | Yes | No | Yes | High |
| MuJoCo MJX | Manual | Manual | Yes | Yes | High |

---

## 9. Ball Trajectory Inference Without Ball Tracking

### 9.1 The BallRadar Approach (KDD 2023)

Most directly relevant paper: "Ball Trajectory Inference from Multi-Agent Sports Contexts" (Kim et al., KDD 2023)
- Infers ball trajectory from PLAYER trajectories (no ball tracking)
- Uses Set Transformer + Hierarchical Bi-LSTM
- Hierarchical: first predicts ball possessor, then ball trajectory
- Designed for soccer (multiple players), but the concept applies
- Reference: [Paper](https://arxiv.org/abs/2306.08206), [GitHub](https://github.com/hyunsungkim-ds/ballradar)

**Key difference from our problem**: BallRadar uses multiple players' trajectories to infer ball position. We have a single player shooting a free throw. The multi-agent context that BallRadar exploits does not exist in our setting.

### 9.2 PitcherNet (2024)

Predicts pitch statistics from kinematic data in broadcast video:
- Estimates pitch velocity, ball release point, release extension
- End-to-end from video to pitch statistics
- Reference: [PitcherNet](https://arxiv.org/html/2405.07407v1)

More relevant to our problem since it predicts ball properties from body kinematics.

---

## 10. Synthesis: What Should We Actually Do?

### 10.1 Ranking of Approaches by Expected Impact

| Rank | Approach | Expected improvement | Effort | Risk |
|------|----------|---------------------|--------|------|
| 1 | Better biomechanical features (Nimble IK/ID) | Medium | Medium | Low |
| 2 | Physics-constrained gray-box model | Medium | Medium | Medium |
| 3 | Differentiable trajectory optimization | Low-Medium | High | High |
| 4 | MyoSuite hand model | Low | Very High | Very High |
| 5 | Full contact-implicit optimization | Low | Very High | Very High |

### 10.2 Recommended Strategy

**Phase 1: Better Features via Biomechanical Pipeline (1-2 days)**

Use Nimble or direct computation to extract cleaner biomechanical features:

1. **Per-player skeletal model scaling**: Use anthropometric ratios from keypoints to scale segment lengths. This makes joint angle computation more accurate.

2. **Inverse kinematics**: Fit joint angles to keypoint positions with biomechanical constraints (no hyperextension, joint limits). This gives cleaner angular velocities than raw keypoint differencing.

3. **Joint angular velocities at key frames**: Specifically:
   - Elbow extension angular velocity (strongest predictor per baseball research)
   - Shoulder flexion angular velocity
   - Wrist flexion angular velocity
   - Hip/knee extension angular velocity (energy generation)

4. **Coordination timing features**:
   - Time between peak angular velocities of successive joints (proximal-distal delay)
   - Whether joints fire in correct sequence (shoulder before elbow before wrist)
   - Variability of coordination across the shot

5. **Body orientation features**:
   - Trunk lean angle at release
   - Shoulder-hip alignment (body rotation)
   - Center of mass position relative to feet

**Phase 2: Physics-Constrained Predictions (1 day)**

Instead of predicting angle/depth/left_right directly, predict intermediate physics quantities and constrain them:

1. Predict release_vx, release_vy, release_vz from features
2. Compute trajectory under gravity: position at hoop plane = f(release_pos, release_vel)
3. Convert to angle/depth/left_right
4. Train with physics-consistent loss: MSE on final targets + penalty for unphysical release velocities

This can be done with a simple differentiable model (even linear regression through a physics layer).

**Phase 3: Gray-Box Residual Model (1 day)**

1. Physics model predicts "base" angle/depth/left_right from biomechanical features
2. ML model predicts residual (difference between physics prediction and actual)
3. Final prediction = physics_prediction + ML_residual
4. The physics model provides a strong prior, the ML model corrects for individual differences

### 10.3 What Specifically Will NOT Work

Based on this research, these approaches should be avoided:

1. **Full MuJoCo contact simulation**: Mocap bodies cannot exert real contact forces. This was already proven to fail.

2. **MyoSuite for force estimation**: Too many unknowns (39 muscles from 16 keypoints), no EMG data, no ball position data.

3. **Neural network approaches**: 345 samples is far too few for any neural architecture (transformers, LSTMs, etc.) without massive pretraining.

4. **Ball trajectory prediction from body pose alone**: The BallRadar approach requires multi-agent context. Single-player free throw kinematics do not contain enough information to fully determine ball trajectory - there is inherent uncertainty.

5. **End-to-end differentiable physics optimization**: The computational graph from skeleton -> contact -> release -> flight -> hoop is too long and has too many local minima for 345 samples.

### 10.4 Critical Insight

The fundamental limitation is: **body kinematics alone cannot fully determine ball trajectory**. This is supported by both the research literature and our data:

- Depth correlation with physics features: r = 0.14 (essentially noise)
- Left_right correlation with physics features: r = 0.13 (essentially noise)

Even perfect physics simulation cannot overcome this. The ball's exact trajectory depends on:
- Contact point between fingers and ball (unmeasured)
- Ball spin axis and magnitude (unmeasured)
- Micro-adjustments in finger pressure (below keypoint resolution)
- Ball inflation pressure, surface moisture, etc.

Physics can help most with **angle** (r=0.64 with current features) because the release angle is strongly determined by arm geometry and gross kinematics. Depth and left_right require capturing fine motor control details that 60 Hz keypoint data cannot resolve.

The best strategy is: extract the maximum possible signal from physics-based features for angle prediction, while accepting that depth and left_right improvements must come from other approaches (per-player priors, temporal patterns, etc.).

---

## Sources

### Differentiable Physics
- [Brax - Differentiable Physics Engine](https://arxiv.org/abs/2106.13281)
- [Nimble Physics Engine](https://nimblephysics.org/)
- [DiffTaichi](https://github.com/taichi-dev/difftaichi)
- [MuJoCo MJX Documentation](https://mujoco.readthedocs.io/en/stable/mjx.html)
- [Hard Contacts with Soft Gradients (DiffMJX)](https://arxiv.org/html/2506.14186v1)
- [Differentiable Physics Simulations with Contacts](https://arxiv.org/abs/2207.05060)
- [Drake Robotics Toolbox](https://drake.mit.edu/)
- [JaxSim](https://github.com/ami-iit/jaxsim)

### Biomechanics and Motion Capture
- [Differentiable Biomechanics for Markerless Motion Capture (Cotton, 2024)](https://arxiv.org/abs/2402.17192)
- [Differentiable Biomechanics for Upper Limb Rehabilitation](https://arxiv.org/abs/2411.14992)
- [Biomechanical Reconstruction with Confidence Intervals](https://arxiv.org/html/2502.06486)
- [AddBiomechanics](https://addbiomechanics.org/)
- [Pose2Sim](https://github.com/perfanalytics/pose2sim)
- [MyoSuite](https://github.com/MyoHub/myosuite)
- [Nimble Physics GitHub](https://github.com/keenon/nimblephysics)

### Basketball and Sports Biomechanics
- [Biomechanical characteristics of proficient free-throw shooters](https://pmc.ncbi.nlm.nih.gov/articles/PMC10436204/)
- [Arm Joint Coordination in Basketball](https://pmc.ncbi.nlm.nih.gov/articles/PMC12121896/)
- [Intra-Individual Release Variability](https://pmc.ncbi.nlm.nih.gov/articles/PMC8256521/)
- [Key Kinematic Components for Basketball Free Throw](https://www.semanticscholar.org/paper/Key-Kinematic-Components-for-Optimal-Basketball-Cabarkapa-Fry/5e189104dad27f032ec0bf15c99788638b68afae)
- [Kinematics of Arm Joint Motions in Basketball Shooting](https://www.sciencedirect.com/science/article/pii/S187770581501471X)
- [Kinetic Analysis of Fingers in Throwing](https://journals.humankinetics.com/view/journals/mcj/26/2/article-p226.xml)

### Ball Trajectory and Prediction
- [Ball Trajectory Inference (BallRadar, KDD 2023)](https://arxiv.org/abs/2306.08206)
- [Universal Model for Ball Trajectory Prediction](https://arxiv.org/abs/2503.18584)
- [PitcherNet](https://arxiv.org/html/2405.07407v1)
- [ML Prediction of Fastball Velocity](https://www.sciencedirect.com/science/article/abs/pii/S0021929022000550)

### Physics-Informed ML
- [PINN for Projectile Motion](https://github.com/okada39/pinn_projectile)
- [Deep Grey-Box Modeling](https://proceedings.mlr.press/v206/takeishi23a/takeishi23a.pdf)
- [Gray-Box Hybrid Simulation](https://arxiv.org/html/2410.17103v1)
- [Enhancing Biomechanical ML with Limited Data](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2024.1350135/full)
- [Physics-Informed ML Survey](https://link.springer.com/article/10.1007/s44379-025-00016-0)

### Contact Dynamics
- [Todorov Contact Model](https://www.semanticscholar.org/paper/A-convex,-smooth-and-invertible-contact-model-for-Todorov/3e677b1fa07d66f539f8086046779190694327eb)
- [Contact-Implicit Trajectory Optimization](https://journals.sagepub.com/doi/10.1177/0278364919849235)
- [Contact-Implicit MPC for Quadrupeds](https://journals.sagepub.com/doi/10.1177/02783649241273645)
- [Hand-Ball Contact Force Modeling](https://ieeexplore.ieee.org/document/6973883/)
- [Fingertip Contact Force Simulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC8044057/)

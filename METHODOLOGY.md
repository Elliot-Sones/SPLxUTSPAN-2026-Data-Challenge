# Methodology: SPLxUTSPAN 2026 Free Throw Prediction

**Final LB Score**: 0.006148

---

## Starting Point

The obvious first move was to treat every shot the same. Feed all 207 joint positions into a Ridge regression, train it on the full dataset, and hope the patterns generalise. It worked, roughly. Early wins - joint angles and multi-frame averaging - confirmed real signal, cutting error by six and four percent. But the residuals told a clear story: the model was not making random mistakes. It was making the same mistakes for the same player, shot after shot.

![MuJoCo skeleton visualization](https://www.kaggle.com/datasets/elliotsones/spl-utspan-methodology-images/mujoco_skeleton.png)

*The 69 motion capture keypoints mapped onto a MuJoCo physics skeleton, showing the basketball, backboard, and hoop. This visualization helped me understand the spatial relationship between the player's body and the target, and motivated the hoop-relative coordinate system used throughout the pipeline.*

## Breakthrough 1 - Every Player Shoots Differently

Players do not just differ in skill. They differ in *mechanism*. Each player had a specific physical tell unique to them, and what predicted one player's shot told you nothing about anyone else's. One controlled depth through whole-body forward thrust. Another aimed left-right purely through wrist position set before the ball even left the hand. A global model averaged over all of this and captured none of it.

The fix was to build each prediction from the player's own data alone, weighted by biomechanical similarity to the shot being predicted. A Gaussian kernel over feature space meant each test shot was explained primarily by the most similar training shots. That single change dropped error by thirteen percent. It also revealed why depth is fundamentally harder to predict than angle: the motor system can precisely control where you aim your arm, but it cannot perfectly control how hard you throw - and depth is driven by force, not position.

**Outcome: -13% error reduction. Leaderboard confirmed.**

## Breakthrough 2 - Finding the Exact Moment the Shot Is Decided

The assumption had been to extract features at release. But when I traced each joint's correlation with each outcome across every frame, I found something unexpected. Depth was statistically committed 930 milliseconds before release - the shooter's forward momentum is fixed mid-jump. Angle committed around 550 milliseconds before release, locked in by elbow geometry. Left-right did not commit until after release - it lives entirely in the final wrist snap. I extracted features at the frame each outcome was actually decided: 150 for depth, 153 for angle, 170 for left-right.

**Outcome: -4 to 7% error reduction per target. Leaderboard confirmed.**

## Breakthrough 3 - Tracking Force From the Ground to the Fingertips

The wrist is the last link in a chain that starts in the legs. Energy travels from the ground up through the hips, trunk, shoulder, and arrives at the fingertips. I built features tracking this proximal-to-distal transfer - how much energy had propagated through each segment by release - and compressed them per player using PLS. These captured not where the body was, but how force was flowing through it.

At the far end of that chain, I measured exactly how the fingers and wrist behave at release: fingertip positions and velocities, finger spread, wrist flexion, and curl across all five fingers. These 53 features captured the fine motor mechanics of the final contact between hand and ball. Together, the kinetic chain and hand physics gave the model a complete picture of force from origin to release.

**Outcome: kinetic chain -2.2% LOO, hand physics -1.88% LOO. Both leaderboard confirmed.**

## Breakthrough 4 - Velocity CNN and Position CNN

With the feature-based models near their ceiling - something only discovered after fixing a validation bug inflating my scores by 3x - I turned to neural networks. A velocity CNN learned how fast every joint was moving over time: when momentum peaked, how sharply the wrist accelerated. A position CNN learned where every joint was moving over time, capturing the spatial trajectory. Despite sharing the same architecture, they captured genuinely different information - their predictions correlated at only r=0.63 for depth, meaning they were wrong on different shots.

**Outcome: velocity CNN LB 0.006306, position CNN LB 0.006265. Both leaderboard confirmed.**

## Breakthrough 5 - Fusing MiniRocket Temporal Features Into the Ridge Model

Every breakthrough so far came from me understanding the biomechanics and encoding that understanding as features. This one came from admitting that 240 frames of 69 joints contain patterns no human would think to look for. MiniRocket stamps 5,000 differently shaped rulers across the full motion sequence and counts how often each pattern fires. Most match nothing useful, but the few that survive regularization detect real timing signatures in the shooting motion. Rather than running these as a separate model - 5,000 random features on 65 training shots per player would be far too many - I fused them directly into the Ridge feature space alongside the hand-crafted biomechanics, so the useful ones act as corrections filling in timing information that single-frame snapshots miss. The fused model beat both pure Ridge and pure ROCKET on every target, helping depth and left-right most - the two targets driven by force and timing.

**Outcome: fused model beat both individual models on all three targets. Leaderboard confirmed.**

---

The final submission stacks all of these together - Ridge pipeline, MiniRocket features, velocity CNN, position CNN, kinetic chain, and hand physics - each at carefully chosen weights. Combining them only worked because each model was wrong on different shots. Models that were diverse but noisy were tested and rejected. The result was a final LB score of 0.006148, the product of five breakthroughs compounding rather than any single idea.

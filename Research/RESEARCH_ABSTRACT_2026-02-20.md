# Research Abstract: Player-Specific Biomechanical Information Channels in Free Throw Shooting

## Abstract

We analyzed 345 free throw shots from 5 basketball players to investigate whether individual players use systematically different body segments as primary information channels for shot outcome control. Using per-player Pearson correlation analysis across ~850 biomechanical features, combined with rigorous statistical validation (Fisher z-tests, bootstrap confidence intervals, permutation tests, 5-fold cross-validation, and cross-player transfer tests), we find strong evidence that each player employs a distinct biomechanical "control channel" for each shot dimension.

The strongest finding: Player 5's shot DEPTH is controlled by a single global signal - forward body velocity (z-axis) across all major joints simultaneously at the moment of release (r=0.860, 5-fold CV mean r=0.869 ± 0.100). A 4-feature ridge regression model achieves LOO R2=0.716 on this player-target pair. By contrast, Player 1's depth is controlled by right-elbow z-position (r=-0.684), and Player 4's by right-wrist y-position (r=0.687).

For lateral control (left-right), Player 2's aim is determined by wrist lateral POSITION at 10ms before release (r=-0.788, Fisher z vs. other players: z=4.1-4.6, p<0.001). Cross-player transfer of this feature to other players yields r=-0.049 - essentially zero - confirming player-specificity rather than a shared biomechanical strategy. Player 3's lateral aim is controlled by left-shoulder position (r=+0.692, cross-player transfer r=+0.004).

These channel differences are statistically confirmed: pairwise Fisher z-tests between players on the same feature yield z=4-5 for key comparisons (p<0.001), permutation tests show all key correlations exceed the 99th percentile of the null distribution, and 5-fold cross-validation confirms temporal stability.

**Physical interpretation**: We identify three distinct depth-control strategies:
1. Isolated arm mechanics (Players 1, 2, 3): depth encoded in elbow/wrist kinematics
2. Wrist positioning (Player 4): depth pre-encoded in wrist height before release
3. Whole-body momentum (Player 5): depth controlled by total body forward thrust

**Practical implications**: Player-adaptive predictive models that weight features by within-player correlation achieve 40-70% reduction in prediction error for the strongest channels. A simple oracle model covering all players achieves LOO R2=0.582 for Player 2's lateral control and R2=0.716 for Player 5's depth using only 3-4 features per player-target pair.

## Key Statistics

| Finding | Player | Target | Feature | r | Cross-Player Transfer |
|---------|--------|--------|---------|---|----------------------|
| Whole-body thrust | P5 | Depth | vel_left_shoulder_z_f153 | 0.860 | - |
| Wrist positioning | P2 | LR | hr_right_wrist_y_f150 | -0.788 | **-0.049** |
| Shoulder aim | P3 | LR | hr_ls_y_f170 | 0.692 | **+0.004** |
| Hip sweep | P1 | LR | vel_rh_y_f175 | 0.669 | Fisher z=4.09 vs P2 |
| Neck position | P4 | Angle | hr_neck_y_f165 | -0.522 | Fisher z=4.25 vs P2 |
| Elbow position | P1 | Depth | hr_right_elbow_z_f150 | -0.684 | - |

## Methodological Contribution
- Player-adaptive diagonal Mahalanobis metric: w_i(player) = |r(feature_i, target)| + ε
- Honest LOO evaluation with per-fold weight recomputation
- Evidence hierarchy: raw r → permutation control → bootstrap CI → CV stability → transfer test

## Limitations
- 5 players: insufficient for population-level generalization
- Correlations may reflect shooting style consistency rather than causal mechanisms
- Dataset-specific: free throws at fixed distance, controlled conditions

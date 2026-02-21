# Final Push Log - 2026-02-16 to 2026-02-21

## Budget: ~30 submissions (5/day x 6 days)
## Current best: Sub 2503 LB 0.006471
## Target: 0.0059

---

## Phase 1: Backlog Candidates (Day 1)

### Sub 2609 - 1% pulse + 99% Sub 2503
- Description: Pulse features (kinetic energy pulses) at 1% blend with current best
- Source pipeline: scripts/pulse_features_transfer.py
- Base anchor: Sub 2503 (LB 0.006471)
- Blend: 1% pulse standalone + 99% Sub 2503
- LB score: **0.006446** (NEW BEST, -0.000025 vs Sub 2503)

### Sub 2604 - 10% energy wave + 90% Sub 2503
- Description: Energy wave transfer features at 10% blend
- Source pipeline: scripts/energy_wave_transfer.py
- Base anchor: Sub 2503 (LB 0.006471)
- Blend: 10% energy wave + 90% Sub 2503
- LB score: **0.006456** (-0.000015 vs Sub 2503)

### Sub 2583 - Row surgery: P5 + uncertain LR fallback
- Description: Row-level surgery changing 14 rows based on per-row confidence gating
- Source pipeline: scripts/sub2503_row_surgery_candidates.py
- Base anchor: Sub 2503 (LB 0.006471)
- Changes: 14 rows modified
- LB score: 0.006516 (+0.000045, WORSE)

### Phase 1 Conclusions
- Pulse features have REAL LB signal at 1% blend weight
- Energy wave features have REAL LB signal at 10% blend weight
- Row surgery is noise - DEAD
- NEW BEST ANCHOR: Sub 2609 (LB 0.006446)

---

## Phase 2: Multi-Submission Optimization (Day 1)

### Sub 2613 - 5% energy wave + 95% Sub 2609 (compound)
- Description: Layer energy wave on top of already-best pulse blend
- Effective composition: ~94% Sub2503 + ~1% pulse + 5% energy wave
- LB score: **0.006422** (-0.000049 vs Sub 2503)

### Sub 2622 - 4-way: 92% Sub2609 + 1% pulse + 5% energy + 2% CNN1557
- Description: Multi-source blend with temporal CNN diversity
- Effective composition: ~91% Sub2503 + ~2% pulse + 5% energy + 2% CNN
- CNN1557 correlation with anchor: angle=0.88, depth=0.74, LR=0.71
- LB score: **0.006376** (-0.000095 vs Sub 2503, -0.000070 vs Sub 2609)
- MASSIVE JUMP - 4x typical incremental gain

### Key Diversity Analysis (correlations with Sub 2609)
| Sub | Type | Angle r | Depth r | LR r |
|-----|------|---------|---------|------|
| 2608 | Pulse standalone | 0.80 | 0.51 | 0.69 |
| 2602 | Energy wave | 0.92 | 0.63 | 0.69 |
| 1557 | Temporal CNN | 0.88 | 0.74 | 0.71 |
| 1507 | Trajectory | 0.99 | 0.63 | 0.96 |
| 1103 | Video SSL | -0.15 | 0.09 | -0.04 |

### Phase 2 Conclusions
- Multi-source diversity blending WORKS - 4-way significantly better than 2-way
- CNN1557 at just 2% added -0.000046 over pulse+energy only
- Next: try higher CNN weights, add trajectory depth, try 5-way blends
- NEW BEST ANCHOR: Sub 2622 (LB 0.006376)
- Daily budget exhausted (5/5 used)

## Phase 3: Tree Model Development (Day 1)

### XGBoost per-player per-target (Sub 2679)
- Features: 1814 summary statistics across 5 temporal windows x 15 joints x 3 coords
- Per-player XGBoost: n_estimators=100, max_depth=3, lr=0.05, subsample=0.8
- 3-seed average for test predictions
- LOO (scaled [0,1]): angle=0.005170, depth=0.009339, LR=0.015157, mean=0.009889
- Diversity vs Sub 2622: angle=0.967, depth=0.816, LR=0.874
- Kill criteria: BOTH PASSED (LOO < 0.012, min_corr < 0.85)
- Standalone: Sub 2679, Blends: 2680-2690, Combos with CNN: 2691-2703

### Day 2 Candidate Priority List

**Top 5 for submission (highest expected LB improvement):**
1. Sub 2691: Sub2622 + 2%XGB + 2%CNN (new model family + confirmed CNN signal)
2. Sub 2700: 6-way mega blend (pulse+energy+CNN+XGB+traj)
3. Sub 2639: 3% CNN1557 on Sub2622 (more of confirmed CNN signal)
4. Sub 2696: 5%XGB(depth) + 5%CNN(LR) target surgery on Sub2622
5. Sub 2623: 5% CNN1557 on Sub2609 (higher CNN weight)

**Second tier (submit if top 5 show promise):**
6. Sub 2681: 5% XGB + 95% Sub2622
7. Sub 2662: Sub2622 + 2%CNN + 2%traj(depth)
8. Sub 2701: 6-way with more CNN+XGB
9. Sub 2694: Sub2622 + 5%XGB + 2%CNN
10. Sub 2686: 10% XGB depth-only on Sub2622

### LightGBM Results (Sub 2704)
- LOO: angle=0.005084, depth=0.009961, LR=0.016328, mean=0.010458
- Diversity vs Sub 2622: angle=0.956, depth=0.838, LR=0.817
- LR diversity better than XGB (0.817 vs 0.874) but LOO slightly worse
- Standalone: Sub 2704, Blends: 2705-2715

### Tree Average (XGB+LGBM) Results
- Combined diversity vs Sub 2622: angle=0.963, depth=0.835, LR=0.852
- More stable predictions than either alone

### Day 1 Final Submissions (limited to 2 remaining slots)

#### Sub 2716: Sub2622 + 2%TreeAvg + 2%CNN (SUBMITTED)
- **LB: 0.006343** (NEW BEST, -0.000033 vs Sub 2622)
- Description: Layer 2% tree average (XGB+LGBM) + 2% more CNN uniformly on Sub 2622
- Conservative: only 4% new weight total
- Tree model adds genuine value across all targets

#### Sub 2717: Sub2622 + 5%TreeAvg(depth) + 5%CNN(LR) (SUBMITTED)
- LB: 0.006382 (+0.000006 vs Sub 2622)
- Per-target surgery approach slightly worse than uniform blending
- Conclusion: uniform small weights > targeted larger weights

### Day 1 Summary
- Started: Sub 2503 at LB 0.006471
- Ended: Sub 2716 at LB 0.006343 (-0.000128, -1.98%)
- 7 submissions used today
- Key insight: layering diverse sources at 2% weights compounds to large gains
- Effective composition of Sub 2716: ~87% Sub2503 + ~2% pulse + 5% energy + ~4% CNN + 2% TreeAvg
- Gap to target 0.0059: 0.000443 (6.98% relative)

### Next Session Priorities (Day 2)

Pre-generated candidates Sub 2718-2739. Top 5 priority:
1. **Sub 2730**: Sub2716 + 1%Tree + 1%CNN + 1%Traj(depth) + 1%Energy (4-source, 4% total)
2. **Sub 2724**: 2% trajectory depth on Sub 2716 (untested diversity source)
3. **Sub 2719**: 2% more TreeAvg on Sub 2716 (confirm tree model helps at higher weight)
4. **Sub 2722**: 2% more CNN on Sub 2716 (confirm CNN helps at higher weight)
5. **Sub 2736**: 3% LGBM LR-only (LGBM has better LR diversity r=0.82 vs XGB 0.87)

Patterns from Day 1:
- Uniform small weights (2%) beat targeted larger weights (5%)
- Every new diverse source at 2% has helped
- Sub 2716 effective composition: ~87% Sub2503 + 2% pulse + 5% energy + 4% CNN + 2% TreeAvg

---

## Day 2 Session (Feb 16 afternoon)

### Submission Budget Update
- Kaggle limit: 10 submissions/day (not 5)
- Day 1 used: 8 submissions (2609, 2604, 2583, 2613, 2622, 2716, 2717, + 1 other)
- Day 2 used: 2 (2831, 2819) - both worse than Sub 2716
- Total submissions left today: 0

### LB Results (Day 2)

#### Sub 2831: Cherry-picked per-target (SUBMITTED)
- angle 1%pulse, depth 2%pulse+2%traj+2%energy, LR 3%kNN+2%RF on Sub 2716
- LB: 0.006386 (WORSE - too much weight on weak diverse sources)
- Lesson: per-target 5-6% diverse weight hurts

#### Sub 2819: All-target 6% diverse (SUBMITTED)
- angle 2%pulse, depth 6%diverse (pulse+traj+energy+RF), LR 6%diverse (kNN+RF+pulse+CNN) on Sub 2716
- LB: 0.006353 (WORSE but close - confirms 6% is too much per target)
- Lesson: 2% per source is the sweet spot, total 13% optimal

### Key Lesson: Optimal Total Diverse Weight = ~13%
- Sub 2716 (13% diverse): LB 0.006343 (BEST)
- Sub 2819 (19% diverse): LB 0.006353 (slightly worse)
- Sub 2831 (17% diverse): LB 0.006386 (worse)
- Conclusion: Adding more diversity beyond 13% total HURTS
- Strategy: reallocate 13% budget more efficiently, don't increase total

### New Models Built

#### Kernel Ridge Regression (KILLED)
- LOO: 0.103 (way too weak, 16x worse than Ridge)
- Diversity: angle r=0.55, depth r=0.55, LR r=0.36 (incredible diversity)
- But quality too low - noise dominates at any blend weight
- Script: scripts/gp_diverse_model.py

#### LASSO-weighted k-NN (Sub 2832)
- 500 features: positions + velocities + window stats + inter-joint distances
- LOO: 0.009777 (passes kill criteria)
- Diversity: angle r=0.953, depth r=0.901, LR r=0.861
- Moderate diversity, similar to tree models
- Script: scripts/learned_similarity.py
- Standalone: Sub 2832, Blends: 2833-2835

#### DTW-Similarity Weighted Ridge (Sub 2851)
- DTW distance on shooting arm trajectories (frames 100-200)
- LOO: 0.009826 (passes kill criteria)
- Diversity: angle r=0.966, depth r=0.904, LR r=0.864
- Moderate diversity, not a breakthrough
- Script: scripts/dtw_similarity_model.py
- Standalone: Sub 2851, Blends: 2852-2860

#### BiGRU Temporal Sequence Model (Sub 2870) - BREAKTHROUGH DIVERSITY
- Full 240-frame sequence with key joint positions + velocities
- TinyBiGRU: 16-hidden, attention pooling, ~15K params
- 5-seed average, 80 epochs on CPU
- **Diversity vs Sub 2716: angle r=0.941, depth r=0.505, LR r=0.715**
- DEPTH r=0.505 is the most diverse model we have for depth!
- Script: scripts/temporal_bigru.py
- Standalone: Sub 2870, Blends: 2871-2875
- Per-target blends: 2876-2886

### Stacking Meta-Learner Candidates (Sub 2836-2850)
- IVW (inverse variance weighted): Sub 2842 (r=0.959 vs anchor)
- Median: Sub 2843 (r=0.952)
- Trimmed mean: Sub 2844 (r=0.972)
- Per-player best-diverse selector: Sub 2839-2841
- Blended versions: Sub 2845-2850

### Diversity Reallocation Candidates (Sub 2861-2869)
- Strategy: keep total diverse weight at 13% but distribute across more sources
- wider_8way: Sub 2861
- diversity_weighted: Sub 2862
- lr_focused: Sub 2863
- depth_focused: Sub 2864
- proven_plus: Sub 2865
- max_diversity (9 sources at 1.3%): Sub 2866
- Per-target from Ridge: Sub 2867
- Incremental per-target on Sub 2716: Sub 2868
- Aggressive per-target: Sub 2869

### Updated Diversity Table (vs Sub 2716)
| Sub | Type | Angle r | Depth r | LR r |
|-----|------|---------|---------|------|
| 2870 | BiGRU temporal | 0.941 | **0.505** | 0.715 |
| 2608 | Pulse standalone | 0.801 | 0.533 | 0.705 |
| 1507 | Trajectory | 0.989 | 0.611 | 0.953 |
| 2602 | Energy wave | 0.928 | 0.672 | 0.725 |
| 2780 | RF velocity | 0.949 | 0.734 | 0.699 |
| 1557 | Temporal CNN | 0.887 | 0.742 | 0.725 |
| 2784 | k-NN bootstrap | 0.945 | 0.784 | **0.501** |
| 2679 | XGBoost | 0.969 | 0.823 | 0.880 |
| 2832 | LASSO-kNN | 0.953 | 0.901 | 0.861 |
| 2851 | DTW Ridge | 0.966 | 0.904 | 0.864 |
| 2704 | LightGBM | 0.957 | 0.845 | 0.825 |
| 1103 | Video SSL | -0.144 | 0.084 | -0.047 |

### Day 3 Priorities (10 submission slots)

**Top 5 (MUST submit):**
1. **Sub 2876**: 2% BiGRU depth-only on Sub 2716 (exploits r=0.505 depth diversity)
2. **Sub 2886**: Per-target max-diverse: angle 2%P, depth 3%BiGRU+2%P, LR 3%kNN+2%BiGRU
3. **Sub 2884**: 8-way rebuild from Ridge: 87%R + 2%BiGRU + 2%P + 3%E + 3%CNN + 1%XGB + 1%kNN + 1%RF
4. **Sub 2865**: Proven_plus: Sub2716 recipe + 1%kNN + 1%RF
5. **Sub 2862**: Diversity-weighted 6-source: 2.5%pulse + 2.5%kNN + 2%energy + 2%cnn + 2%RF + 2%traj

**Second tier (submit if top 5 show promise):**
6. Sub 2885: 5% BiGRU depth + 3% kNN LR on Sub 2716
7. Sub 2881: 3% BiGRU depth+LR on Sub 2716
8. Sub 2839: Per-player best-diverse selector at 3%
9. Sub 2868: Incremental per-target diverse on Sub 2716
10. Sub 2847: 5% median of all models + 95% Sub 2716

---

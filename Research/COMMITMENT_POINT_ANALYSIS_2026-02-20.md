# Commitment Point Analysis: When Is a Free Throw Outcome Determined?

**Date**: 2026-02-20
**Script**: scripts/commitment_point_analysis.py

## Method

For each player x target, compute max |Pearson r| between any single keypoint position (hip-centered, 33 keypoints x 3 coords = 99 features) and the shot outcome at every frame from 60 to 220 (3-frame steps = 50ms resolution at 60fps). The 'commitment point' is when this best-r first reaches 80% of its session peak.

Release frame: 153 (frame 153 = t=0)

## Results

### Summary Table

| Target | Mean Peak r | Mean Peak (ms before release) | Mean 80% Commit (ms before release) |
|--------|------------|-------------------------------|--------------------------------------|
| angle | 0.500 | 200ms | 550ms |
| depth | 0.731 | 460ms | 930ms |
| left_right | 0.727 | -410ms | -60ms |

### Per-Player Detail

#### ANGLE

| Player | Peak r | Peak frame | ms before release | 80% commit ms |
|--------|--------|------------|-------------------|---------------|
| P1 | 0.461 | 177 | -400ms | 650ms |
| P2 | 0.515 | 96 | 950ms | 1000ms |
| P3 | 0.418 | 120 | 550ms | 600ms |
| P4 | 0.555 | 159 | -100ms | 450ms |
| P5 | 0.552 | 153 | 0ms | 50ms |

#### DEPTH

| Player | Peak r | Peak frame | ms before release | 80% commit ms |
|--------|--------|------------|-------------------|---------------|
| P1 | 0.664 | 150 | 50ms | 800ms |
| P2 | 0.753 | 99 | 900ms | 1550ms |
| P3 | 0.656 | 111 | 700ms | 1250ms |
| P4 | 0.755 | 120 | 550ms | 750ms |
| P5 | 0.827 | 147 | 100ms | 300ms |

#### LEFT_RIGHT

| Player | Peak r | Peak frame | ms before release | 80% commit ms |
|--------|--------|------------|-------------------|---------------|
| P1 | 0.748 | 174 | -350ms | -250ms |
| P2 | 0.848 | 159 | -100ms | 600ms |
| P3 | 0.741 | 162 | -150ms | 100ms |
| P4 | 0.587 | 174 | -350ms | -200ms |
| P5 | 0.710 | 219 | -1100ms | -550ms |


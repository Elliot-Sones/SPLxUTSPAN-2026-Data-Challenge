# Submission Priority List - Feb 17, 2026

## Current State
- **LB Best**: Sub 2716, LB 0.006343 (1st place, leads #2 by 0.000252)
- **Daily slots**: 10 (reset at midnight UTC)
- **Deadline**: Feb 21, 2026

## Key Diversity Sources
| Source | depth r | LR r | angle r | Status |
|--------|---------|------|---------|--------|
| Velocity BiGRU (Sub 3002) | 0.43 | 0.42 | 0.92 | Complete |
| RFF (Sub 2988) | 0.45 | 0.16 | 0.94 | Complete |
| Trajectory (Sub 1507) | 0.61 | 0.95 | 0.99 | Existing |
| CNN 1557 | 0.74 | 0.73 | 0.89 | Existing |
| Phase features (Sub 3010) | 0.79 | 0.86 | 0.94 | Complete |

## Priority Order

### SLOT 1-3: Ultra-Conservative Diverse Injection (Highest EV)
Based on LB-proven pattern: 1-2% weights give -0.000015 to -0.000033 per cascade step.

| Priority | Sub | Description | Rationale |
|----------|-----|-------------|-----------|
| 1 | 3134 | 2% vel_bgru depth + 1% RFF LR | Most diverse depth (r=0.43) + most diverse LR (r=0.16) |
| 2 | 3133 | 1% vel_bgru depth + 1% vel_bgru LR | Conservative both targets from single diverse source |
| 3 | 3131 | 1% vel_bgru all targets | Ultra-conservative, tests vel_bgru signal on all targets |

### SLOT 4-6: Multi-Source Diverse
| Priority | Sub | Description | Rationale |
|----------|-----|-------------|-----------|
| 4 | 3136 | 1% vel + 1% traj depth + 1% RFF LR | Triple diverse source, decorrelated (vel-traj r=0.15) |
| 5 | 3066 | 5% avg(vel+traj) depth + 2% avg(vel+rff) LR | Averaged diverse ensemble reduces noise |
| 6 | 3046 | 8% vel_bgru depth + 2% RFF LR | More aggressive, tests higher weights |

### SLOT 7-8: Known-Quality Blends (Insurance)
| Priority | Sub | Description | Rationale |
|----------|-----|-------------|-----------|
| 7 | 3120 | 89.7% Sub 2716 + 10.3% Sub 784 LR only | Optimizer's best from known subs |
| 8 | 3113 | 95% Sub 2716 + 5% known best per target | Safe per-target known blend |

### SLOT 9-10: Reserve for Day's Findings
- If slots 1-3 improve: test higher weights of the winning source
- If slots 1-3 don't improve: try Tier 2 from optimizer (3084, 3092)
- If BiGRU v2 or pseudo-labeling produce good results: test those

## Strategy Notes
- Sub 2819 (6% diverse blend) scored 0.006353 (0.000010 worse than Sub 2716)
- This confirms aggressive diverse weights hurt; 1-2% is optimal
- The velocity BiGRU is our strongest diversity source ever (depth r=0.43)
- If even 1% vel_bgru helps, we can cascade further
- The cascade pattern (Sub2503 -> Sub2609 -> Sub2622 -> Sub2716) proves each step helps

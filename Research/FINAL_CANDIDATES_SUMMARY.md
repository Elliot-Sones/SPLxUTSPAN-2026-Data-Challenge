# Final Candidates Summary

## Best Performing Submissions (Current)
| Submission | LB Score | angle_std | Description |
|------------|----------|-----------|-------------|
| Sub 219 | **0.007682** | 0.137162 | Selective amplification on Sub 133 |
| Sub 133 | 0.007809 | 0.137728 | Blend: 5% Sub25 + 30% Sub9 + 44% Sub10 + 21% Sub111 |

## Top Candidates by Profile Match (angle_std close to 0.137162)

### Tier 1: Exact Profile Match (angle_std = 0.1371-0.1372)
| Submission | angle_std | Description |
|------------|-----------|-------------|
| **Sub 446** | 0.137163 | 0.60*Sub149 + 0.40*Sub219 |
| **Sub 450** | 0.137160 | 0.70*Sub133 + 0.30*Sub25 |
| **Sub 449** | 0.137171 | 0.65*Sub133 + 0.35*Sub25 |
| **Sub 451** | 0.137150 | 0.75*Sub133 + 0.25*Sub25 |
| **Sub 481** | 0.137151 | Combined best techniques |

### Tier 2: Slightly Different Profile (may work better)
| Submission | angle_std | Description |
|------------|-----------|-------------|
| **Sub 413** | 0.135890 | contrast=147, pctl=88, alpha=0.8 |
| **Sub 414** | 0.135905 | contrast=147, pctl=91, alpha=0.8 |
| **Sub 469** | 0.135792 | amp147_pctl87_alpha0.7 |
| **Sub 470** | 0.135786 | amp147_pctl87_alpha0.8 |

### Tier 3: Novel Approaches
| Submission | angle_std | Description |
|------------|-----------|-------------|
| Sub 441 | 0.137163 | 0.6*Sub149 + 0.4*Sub219 (optimized pair) |
| Sub 443 | 0.137162 | Optimized blend (63% Sub133) |
| Sub 456 | 0.137038 | 0.5*Sub133 + 0.25*Sub219 + 0.25*Sub149 |

## Recommended Testing Order

### Priority 1 (Most likely to beat Sub 219)
1. **Sub 446**: Exact profile match with different base
2. **Sub 450**: Simple blend matching profile
3. **Sub 413**: Lower angle_std with Sub 147 contrast

### Priority 2 (Different approaches)
4. **Sub 469**: Lowest angle_std candidate
5. **Sub 456**: Three-way blend
6. **Sub 443**: Optimized 5-way blend

### Priority 3 (If above don't work)
7. Sub 481: Combined techniques
8. Sub 449: Alternate blend ratio
9. Sub 451: Higher Sub 133 weight

## Key Findings

1. **Sub 147 as contrast** produces lower angle_std (0.1358) than Sub 151 (0.1372)
2. **0.6*Sub149 + 0.4*Sub219** exactly matches Sub 219's profile
3. **0.70*Sub133 + 0.30*Sub25** is simpler and matches profile
4. **Three-way blends** provide more flexibility but don't clearly beat pairs

## External Data Attempts (All Failed)
- NBA trajectory data: Modality mismatch
- OpenBiomechanics baseball: Different mechanics
- Physics simulation: Doesn't generalize
- SkillMimic: No outcome labels

## Approaches Tried
- [x] External datasets (6+ sources)
- [x] Physics simulation (MuJoCo)
- [x] Transfer learning (baseball, basketball)
- [x] Pseudo-labeling / semi-supervised
- [x] Stacking / meta-learning
- [x] Selective amplification variations
- [x] Exhaustive blend optimization
- [x] Per-target optimization
- [x] Feature engineering (100+ features)

## Conclusion

The best strategy appears to be **optimizing blend weights** between existing good submissions. Sub 219's technique (selective amplification) works, and using Sub 147 as contrast instead of Sub 151 produces slightly better profiles.

The most promising candidates to test:
1. Sub 446 (0.60*Sub149 + 0.40*Sub219)
2. Sub 450 (0.70*Sub133 + 0.30*Sub25)
3. Sub 413 (Sub 147 contrast amplification)

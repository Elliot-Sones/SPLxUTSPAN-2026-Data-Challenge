# Latent Release Physics Priority 1 - Benchmark Results

Date: 2026-02-15
Script: `scripts/latent_release_physics_benchmark.py`
Goal: Test Priority 1 idea as additive latent release-state features in the per-example local-weighted pipeline.

## What We Ran - Exact Commands

Pilot run (small-scale validation):

```bash
uv run python scripts/latent_release_physics_benchmark.py --scale 1 --seed 20260215 --run-tag latent_release_priority1_pilot_20260215
```

Full run (same pipeline, only `--scale` changed):

```bash
uv run python scripts/latent_release_physics_benchmark.py --scale 8 --seed 20260215 --run-tag latent_release_priority1_full_20260215
```

## Data Used

- Train source: `data/train.csv` parsed through `scripts/per_example_pipeline.py` loader
- Test source: `data/test.csv` parsed through same loader
- Targets: `angle`, `depth`, `left_right`
- Target scalers:
  - `data/scaler_angle.pkl`
  - `data/scaler_depth.pkl`
  - `data/scaler_left_right.pkl`

Shots used per run:
- Pilot (`--scale 1`): `55 / 345` train shots, `113` test shots
- Full (`--scale 8`): `345 / 345` train shots, `113` test shots

## Model and Config

Both variants used identical modeling:
- Base model: per-player locally weighted `Ridge(alpha=10.0)` from `scripts/per_example_pipeline.py`
- PLS augmentation: same `augment_with_pls(...)` path used by existing per-example pipeline
- Validation metric: scaled OOF MSE

Compared variants:
1. `baseline`
   - Existing compact handcrafted features (`198`) + PLS (`15`) -> `213` augmented features
2. `latent`
   - Baseline + latent release-state block (`27`) -> `225` handcrafted, `240` augmented features

Bandwidth grids:
- Pilot (`scale=1`): `[0.30]`
- Full (`scale=8`): `[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]`

## Exact Results

## Pilot (`latent_release_priority1_pilot_20260215`)

Baseline:
- angle MSE: `0.014595107552`
- depth MSE: `0.009518159665`
- left_right MSE: `0.015692653834`
- mean MSE: `0.013268640350`

Latent:
- angle MSE: `0.014656687326`
- depth MSE: `0.010383383941`
- left_right MSE: `0.016521215685`
- mean MSE: `0.013853762317`

Delta (latent vs baseline):
- angle: `+0.421921%`
- depth: `+9.090248%`
- left_right: `+5.279935%`
- mean: `+4.409811%`

Artifact:
- `output/latent_release_physics_benchmark_latent_release_priority1_pilot_20260215.json`

## Full (`latent_release_priority1_full_20260215`)

Best baseline bandwidths:
- angle `bw=0.50`
- depth `bw=0.50`
- left_right `bw=0.50`

Baseline:
- angle MSE: `0.002510553050`
- depth MSE: `0.004510395048`
- left_right MSE: `0.004208871868`
- mean MSE: `0.003743273322`

Best latent bandwidths:
- angle `bw=0.50`
- depth `bw=0.40`
- left_right `bw=0.50`

Latent:
- angle MSE: `0.002736052342`
- depth MSE: `0.004657125522`
- left_right MSE: `0.004598180369`
- mean MSE: `0.003997119411`

Delta (latent vs baseline):
- angle: `+8.982056%`
- depth: `+3.253162%`
- left_right: `+9.249711%`
- mean: `+6.781393%`

Artifact:
- `output/latent_release_physics_benchmark_latent_release_priority1_full_20260215.json`

## Conclusion

This Priority 1 implementation as currently formulated did not improve signal:
- Latent release-state additive block is worse than baseline on all three targets in full-scale CV.
- Full-scale mean MSE regressed from `0.003743273322` to `0.003997119411`.

Decision from this run:
- Do not promote this exact latent feature variant to submission blending.
- If continued, next revision should add explicit reliability gating to suppress latent features on low-confidence shots before re-testing.

## Failure Diagnostics (Exact)

To explain the regression, we ran follow-up diagnostics.

### Diagnostic Command 1: reliability + raw correlation scan

```bash
uv run python -c "import sys, numpy as np, joblib; sys.path.insert(0,'scripts'); import per_example_pipeline as pep; import latent_release_physics_benchmark as l; train,_=pep.load_data(); y=train['y']; tmap={'angle':0,'depth':1,'left_right':2};\
for target in pep.TARGETS:\
    X,rf=pep.extract_all_features(train,target)\
    lat=l.extract_latent_matrix(train,rf)\
    rel=lat[:,22]\
    print(f'[{target}] latent_shape={lat.shape} reliability_mean={rel.mean():.6f} median={np.median(rel):.6f} p10={np.quantile(rel,0.1):.6f} p90={np.quantile(rel,0.9):.6f}')\
    yt=joblib.load(pep.DATA_DIR / f'scaler_{target}.pkl').transform(y[:,tmap[target]].reshape(-1,1)).ravel()\
    cors=[]\
    for j in range(lat.shape[1]):\
        a=lat[:,j]\
        if np.std(a)<1e-10:\
            cors.append(0.0)\
        else:\
            cors.append(abs(np.corrcoef(a,yt)[0,1]))\
    top=np.sort(cors)[-5:][::-1]\
    print(f'[{target}] top5_abs_corr=' + ','.join(f'{v:.6f}' for v in top))\
"
```

Exact output summary:
- angle: `reliability_mean=0.871104`, `median=0.873329`, `p10=0.836615`, `p90=0.903939`, `top5_abs_corr=0.718939,0.665516,0.656648,0.542856,0.542482`
- depth: `reliability_mean=0.871104`, `median=0.873329`, `p10=0.836615`, `p90=0.903939`, `top5_abs_corr=0.241258,0.133901,0.132326,0.120861,0.113811`
- left_right: `reliability_mean=0.871104`, `median=0.873329`, `p10=0.836615`, `p90=0.903939`, `top5_abs_corr=0.139993,0.112533,0.108977,0.076128,0.075841`

Interpretation:
- Reliability was tightly clustered and high, so this version did not effectively separate high-trust vs low-trust cases.

### Diagnostic Command 2: global-vs-within-player correlation check (angle top features)

```bash
uv run python -c "import sys, numpy as np, joblib; sys.path.insert(0,'scripts'); import per_example_pipeline as pep; import latent_release_physics_benchmark as l; train,_=pep.load_data(); y=train['y']; p=train['pids']; X,rf=pep.extract_all_features(train,'angle'); lat=l.extract_latent_matrix(train,rf); yt=joblib.load(pep.DATA_DIR / 'scaler_angle.pkl').transform(y[:,0].reshape(-1,1)).ravel();\
for j in np.argsort([abs(np.corrcoef(lat[:,k],yt)[0,1]) if np.std(lat[:,k])>1e-10 else 0 for k in range(lat.shape[1])])[-5:][::-1]:\
    g=np.corrcoef(lat[:,j],yt)[0,1]\
    print(f'feat{j} global_r={g:.6f}')\
    vals=[]\
    for pid in sorted(np.unique(p)):\
        m=p==pid\
        if np.std(lat[m,j])<1e-10 or np.std(yt[m])<1e-10:\
            r=0.0\
        else:\
            r=np.corrcoef(lat[m,j],yt[m])[0,1]\
        vals.append(r)\
    print('  per_player_r=' + ','.join(f'{v:.6f}' for v in vals))\
"
```

Exact output:
- `feat7 global_r=0.718939` with per-player `-0.040192,0.104862,0.089984,0.068824,-0.111758`
- `feat6 global_r=0.665516` with per-player `-0.090691,0.116503,0.097395,0.146490,-0.088232`
- `feat25 global_r=0.656648` with per-player `-0.087174,0.106754,0.066147,0.166119,-0.097365`
- `feat15 global_r=0.542856` with per-player `-0.118906,0.108054,-0.025619,0.000883,-0.081965`
- `feat12 global_r=0.542482` with per-player `-0.087931,0.045244,0.042480,-0.026996,-0.224572`

Interpretation:
- High global correlations were not stable within players (sign flips and weak magnitudes), so signal was not robust for per-player local prediction.

### Diagnostic Command 3: neighbor-set stability in local-weighted model (angle)

```bash
uv run python -c "import sys, numpy as np; sys.path.insert(0,'scripts'); import per_example_pipeline as pep; import latent_release_physics_benchmark as l; from sklearn.preprocessing import StandardScaler; from sklearn.neighbors import NearestNeighbors; train,_=pep.load_data(); y=train['y']; p=train['pids']; Xb,rf=pep.extract_all_features(train,'angle'); Xl=np.hstack([Xb,l.extract_latent_matrix(train,rf)]); Xb_aug,_=pep.augment_with_pls(Xb,y[:,0],p,Xb,p,train['X_raw'],train['X_raw']); Xl_aug,_=pep.augment_with_pls(Xl,y[:,0],p,Xl,p,train['X_raw'],train['X_raw']);\
for pid in sorted(np.unique(p)):\
    m=np.where(p==pid)[0]\
    xb=Xb_aug[m]; xl=Xl_aug[m]\
    sb=StandardScaler().fit_transform(xb)\
    sl=StandardScaler().fit_transform(xl)\
    k=min(10,len(m)-1)\
    nnb=NearestNeighbors(n_neighbors=k+1).fit(sb)\
    nnl=NearestNeighbors(n_neighbors=k+1).fit(sl)\
    ov=[]\
    for i in range(len(m)):\
        ib=nnb.kneighbors(sb[i:i+1],return_distance=False)[0]\
        il=nnl.kneighbors(sl[i:i+1],return_distance=False)[0]\
        ib=ib[ib!=i][:k]; il=il[il!=i][:k]\
        ov.append(len(set(ib.tolist()) & set(il.tolist()))/k)\
    print(f'pid={pid} k={k} mean_neighbor_overlap={np.mean(ov):.6f} p10={np.quantile(ov,0.1):.6f}')\
"
```

Exact output:
- `pid=1 k=10 mean_neighbor_overlap=0.647143 p10=0.500000`
- `pid=2 k=10 mean_neighbor_overlap=0.846970 p10=0.700000`
- `pid=3 k=10 mean_neighbor_overlap=0.777941 p10=0.600000`
- `pid=4 k=10 mean_neighbor_overlap=0.828358 p10=0.700000`
- `pid=5 k=10 mean_neighbor_overlap=0.917568 p10=0.800000`

Interpretation:
- The extra latent block materially changes nearest-neighbor sets for some players (especially player 1), which degrades local weighting quality in `locally_weighted_prediction`.

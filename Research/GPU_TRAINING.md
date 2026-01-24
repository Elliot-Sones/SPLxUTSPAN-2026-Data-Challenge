# GPU Training Guide

## Quick Start on vast.ai

### 1. Rent a GPU Instance

Go to [vast.ai](https://vast.ai) and rent an instance with:
- GPU: RTX 3090 or better (for speed)
- Image: PyTorch 2.0+ with CUDA
- Storage: 10GB+ (for data and models)

### 2. Upload Code

```bash
# From your local machine
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude 'output/*.pkl' \
    . root@<INSTANCE_IP>:/workspace/splxutspan/
```

### 3. Setup Environment

```bash
ssh root@<INSTANCE_IP>
cd /workspace/splxutspan
bash vast_setup.sh
```

### 4. Run Training

```bash
# Quick test - XGBoost with GPU
python src/train_gpu.py --model xgboost_gpu

# With hyperparameter tuning (recommended)
python src/train_gpu.py --model xgboost_gpu --tune --trials 100

# Deep learning
python src/train_gpu.py --model cnn_lstm
python src/train_gpu.py --model transformer

# All models
python src/train_gpu.py --model all
```

### 5. Run Full Experiments

```bash
# Quick experiments (XGBoost + LightGBM)
python src/run_experiments.py --quick

# Full suite with tuning and deep learning
python src/run_experiments.py --full
```

## Available Models

| Model | Type | Notes |
|-------|------|-------|
| `xgboost_gpu` | Gradient Boosting | Fast, uses extracted features |
| `lightgbm_gpu` | Gradient Boosting | Fast, uses extracted features |
| `cnn_lstm` | Deep Learning | Uses raw time series |
| `transformer` | Deep Learning | Uses raw time series |

## Expected Results

Leader score: **0.008381 scaled MSE**

| Model | Expected MSE | Scaled MSE | Gap to Leader |
|-------|-------------|------------|---------------|
| XGBoost (default) | ~40 | ~0.035 | ~4x |
| XGBoost (tuned) | ~30 | ~0.025 | ~3x |
| CNN-LSTM | TBD | TBD | TBD |
| Transformer | TBD | TBD | TBD |

## Hyperparameter Tuning

The `--tune` flag uses Optuna to search for optimal hyperparameters:

```bash
# 50 trials (default)
python src/train_gpu.py --model xgboost_gpu --tune

# More trials for better results
python src/train_gpu.py --model xgboost_gpu --tune --trials 200
```

## Output Files

All outputs are saved to `output/`:
- `features_train.pkl` - Cached features
- `raw_timeseries_train.pkl` - Cached raw data
- `gpu_training_results.json` - Training results
- `experiments_*.json` - Experiment logs

## Downloading Results

```bash
# From local machine
rsync -avz root@<INSTANCE_IP>:/workspace/splxutspan/output/ ./output/
```

## Tips

1. **Start with XGBoost tuning** - it's fast and gives good results
2. **Use 3-fold CV for tuning** - faster iteration (default)
3. **Use 5-fold CV for final evaluation** - more robust estimate
4. **Deep learning needs more data** - may not outperform gradient boosting on 345 samples
5. **Monitor GPU memory** - reduce batch size if OOM errors

## Cost Estimate

On vast.ai with RTX 3090 (~$0.30/hr):
- Quick experiments: ~$0.50 (1-2 hours)
- Full experiments with tuning: ~$2-5 (6-15 hours)

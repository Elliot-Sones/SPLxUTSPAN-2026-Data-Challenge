#!/bin/bash
# Setup script for vast.ai GPU training
# Run this after SSH-ing into the instance

set -e

echo "=========================================="
echo "SPLxUTSPAN 2026 - vast.ai GPU Setup"
echo "=========================================="

# Check GPU
echo "Checking GPU..."
nvidia-smi

# Install dependencies
echo "Installing Python dependencies..."
pip install -q -r requirements-gpu.txt

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Verify XGBoost GPU
python -c "import xgboost as xgb; print(f'XGBoost: {xgb.__version__}')"

# Verify LightGBM GPU
python -c "import lightgbm as lgb; print(f'LightGBM: {lgb.__version__}')"

echo "=========================================="
echo "Setup complete!"
echo ""
echo "Available training commands:"
echo "  python src/train_gpu.py --model xgboost_gpu"
echo "  python src/train_gpu.py --model lightgbm_gpu"
echo "  python src/train_gpu.py --model cnn_lstm"
echo "  python src/train_gpu.py --model transformer"
echo "  python src/train_gpu.py --model all"
echo ""
echo "With hyperparameter tuning:"
echo "  python src/train_gpu.py --model xgboost_gpu --tune --trials 100"
echo "=========================================="

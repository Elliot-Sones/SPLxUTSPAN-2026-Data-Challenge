# Models for SPLxUTSPAN 2026 Data Challenge
from .baseline import XGBoostBaseline, LightGBMBaseline, SklearnBaseline

# Deep learning models (require PyTorch)
try:
    from .deep_learning import CNNLSTM, TransformerModel, DeepLearningTrainer
except ImportError:
    pass  # PyTorch not installed

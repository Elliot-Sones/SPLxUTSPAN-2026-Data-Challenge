import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Tuple, List, Optional

# Constants
FRAME_RATE = 60
DT = 1.0 / FRAME_RATE
NUM_FRAMES = 240
PRE_RELEASE_START = 150
PRE_RELEASE_END = 200

def parse_array_json(s):
    """Parse string array to numpy array, handling NaNs."""
    if pd.isna(s):
        return np.full(NUM_FRAMES, np.nan, dtype=np.float32)
    s = str(s).replace('nan', 'null')
    return np.array(json.loads(s), dtype=np.float32)

def load_data(
    data_dir: str = '../../data', 
    train: bool = True, 
    max_shots: Optional[int] = None
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Load data and convert JSON strings to numpy arrays.
    
    Args:
        data_dir: Directory containing train.csv/test.csv
        train: Load train.csv if True, else test.csv
        max_shots: Limit number of shots for testing
        
    Returns:
        df: DataFrame with metadata/targets
        X: Numpy array of shape (n_shots, n_frames, n_features)
        keypoint_cols: List of feature names
    """
    filename = 'train.csv' if train else 'test.csv'
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        # Fallback to current directory or search
        if os.path.exists(filename):
            filepath = filename
        else:
            raise FileNotFoundError(f"Could not find {filename} in {data_dir} or current directory")
            
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, nrows=max_shots)
    
    # Identify keypoint columns
    meta_cols = ['id', 'shot_id', 'participant_id']
    target_cols = ['angle', 'depth', 'left_right']
    keypoint_cols = [c for c in df.columns if c not in meta_cols + target_cols]
    
    # Pre-allocate X
    n_shots = len(df)
    n_features = len(keypoint_cols)
    X = np.zeros((n_shots, NUM_FRAMES, n_features), dtype=np.float32)
    
    # Parse JSON arrays
    print("Parsing time-series data...")
    for i, row in df.iterrows():
        for j, col in enumerate(keypoint_cols):
            X[i, :, j] = parse_array_json(row[col])
            
    print(f"Loaded X shape: {X.shape}")
    return df, X, keypoint_cols

def get_keypoint_index(keypoint_cols: List[str]) -> dict:
    """Map keypoint names (e.g. 'nose') to their starting feature index."""
    idx_map = {}
    for i, col in enumerate(keypoint_cols):
        # Assumes format "name_x", "name_y", "name_z"
        if col.endswith('_x'):
            name = col[:-2]
            idx_map[name] = i // 3 # Assuming 3 axes per keypoint and contiguous
            # Better: just map name -> col index
    return idx_map

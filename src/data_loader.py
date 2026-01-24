"""
Memory-efficient data loader for SPLxUTSPAN 2026 Data Challenge.

The train.csv is ~324MB with time series stored as string arrays.
This module provides chunked loading and lazy evaluation to avoid OOM.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterator, Tuple, List, Optional, Dict
import pickle
import joblib


# Constants
FRAME_RATE = 60
NUM_FRAMES = 240
NUM_KEYPOINTS = 69
NUM_COORDS = 3  # x, y, z
NUM_FEATURES = NUM_KEYPOINTS * NUM_COORDS  # 207

DATA_DIR = Path(__file__).parent.parent / "data"

# Column names
META_COLS = ["id", "shot_id", "participant_id"]
TARGET_COLS = ["angle", "depth", "left_right"]


def get_keypoint_columns() -> List[str]:
    """Get list of all keypoint column names."""
    # Read just the header to get column names
    df = pd.read_csv(DATA_DIR / "train.csv", nrows=0)
    keypoint_cols = [c for c in df.columns if c not in META_COLS + TARGET_COLS]
    return keypoint_cols


def parse_array_string(s: str) -> np.ndarray:
    """Parse a string representation of array to numpy array."""
    if pd.isna(s):
        return np.full(NUM_FRAMES, np.nan, dtype=np.float32)
    # Replace nan with null for JSON parsing
    s = s.replace("nan", "null")
    arr = np.array(json.loads(s), dtype=np.float32)
    # Replace None/null with np.nan
    return arr


def load_metadata(train: bool = True) -> pd.DataFrame:
    """Load only metadata columns (fast, low memory)."""
    filepath = DATA_DIR / ("train.csv" if train else "test.csv")
    cols_to_load = META_COLS + (TARGET_COLS if train else [])
    df = pd.read_csv(filepath, usecols=cols_to_load)
    return df


def load_single_shot(row_idx: int, train: bool = True) -> Tuple[Dict, np.ndarray]:
    """
    Load a single shot's time series data.

    Returns:
        metadata: dict with id, shot_id, participant_id, and targets (if train)
        timeseries: np.ndarray of shape (240, 207) - frames x features
    """
    filepath = DATA_DIR / ("train.csv" if train else "test.csv")

    # Read single row
    df = pd.read_csv(filepath, skiprows=range(1, row_idx + 1), nrows=1)
    row = df.iloc[0]

    # Extract metadata
    metadata = {
        "id": row["id"],
        "shot_id": row["shot_id"],
        "participant_id": row["participant_id"],
    }
    if train:
        metadata["angle"] = row["angle"]
        metadata["depth"] = row["depth"]
        metadata["left_right"] = row["left_right"]

    # Extract time series
    keypoint_cols = [c for c in df.columns if c not in META_COLS + TARGET_COLS]
    timeseries = np.zeros((NUM_FRAMES, len(keypoint_cols)), dtype=np.float32)

    for i, col in enumerate(keypoint_cols):
        timeseries[:, i] = parse_array_string(row[col])

    return metadata, timeseries


def iterate_shots(train: bool = True, chunk_size: int = 10) -> Iterator[Tuple[Dict, np.ndarray]]:
    """
    Iterate through all shots, yielding (metadata, timeseries) tuples.

    Uses chunked reading to manage memory.
    """
    filepath = DATA_DIR / ("train.csv" if train else "test.csv")

    # Get total rows
    meta_df = load_metadata(train)
    total_rows = len(meta_df)
    del meta_df

    # Get keypoint columns from header
    header_df = pd.read_csv(filepath, nrows=0)
    keypoint_cols = [c for c in header_df.columns if c not in META_COLS + TARGET_COLS]
    del header_df

    # Read in chunks
    for chunk_start in range(0, total_rows, chunk_size):
        chunk_df = pd.read_csv(
            filepath,
            skiprows=range(1, chunk_start + 1),
            nrows=min(chunk_size, total_rows - chunk_start)
        )

        for _, row in chunk_df.iterrows():
            # Extract metadata
            metadata = {
                "id": row["id"],
                "shot_id": row["shot_id"],
                "participant_id": row["participant_id"],
            }
            if train:
                metadata["angle"] = row["angle"]
                metadata["depth"] = row["depth"]
                metadata["left_right"] = row["left_right"]

            # Extract time series
            timeseries = np.zeros((NUM_FRAMES, len(keypoint_cols)), dtype=np.float32)
            for i, col in enumerate(keypoint_cols):
                timeseries[:, i] = parse_array_string(row[col])

            yield metadata, timeseries

        del chunk_df


def load_all_as_arrays(train: bool = True, max_shots: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load all data into numpy arrays.

    WARNING: This loads everything into memory. Use only if you have enough RAM.
    For 344 training shots: ~344 * 240 * 207 * 4 bytes = ~68MB for timeseries alone.

    Returns:
        X: np.ndarray of shape (n_shots, 240, 207)
        y: np.ndarray of shape (n_shots, 3) - targets (None for test)
        meta: pd.DataFrame with metadata
    """
    meta_df = load_metadata(train)
    n_shots = len(meta_df) if max_shots is None else min(max_shots, len(meta_df))

    X = np.zeros((n_shots, NUM_FRAMES, NUM_FEATURES), dtype=np.float32)
    if train:
        y = np.zeros((n_shots, 3), dtype=np.float32)
    else:
        y = None

    for i, (metadata, timeseries) in enumerate(iterate_shots(train)):
        if i >= n_shots:
            break
        X[i] = timeseries
        if train:
            y[i] = [metadata["angle"], metadata["depth"], metadata["left_right"]]

    return X, y, meta_df.iloc[:n_shots]


def load_scalers() -> Dict[str, object]:
    """Load the target scalers (saved with joblib)."""
    import warnings
    scalers = {}
    for target in TARGET_COLS:
        scaler_path = DATA_DIR / f"scaler_{target}.pkl"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore sklearn version warnings
            scalers[target] = joblib.load(scaler_path)
    return scalers


def scale_targets(y: np.ndarray, scalers: Dict) -> np.ndarray:
    """Scale targets using provided scalers."""
    y_scaled = np.zeros_like(y)
    for i, target in enumerate(TARGET_COLS):
        y_scaled[:, i] = scalers[target].transform(y[:, i].reshape(-1, 1)).ravel()
    return y_scaled


def inverse_scale_targets(y_scaled: np.ndarray, scalers: Dict) -> np.ndarray:
    """Inverse scale targets."""
    y = np.zeros_like(y_scaled)
    for i, target in enumerate(TARGET_COLS):
        y[:, i] = scalers[target].inverse_transform(y_scaled[:, i].reshape(-1, 1)).ravel()
    return y


if __name__ == "__main__":
    # Quick test
    print("Testing data loader...")

    # Test metadata loading
    meta = load_metadata(train=True)
    print(f"Training shots: {len(meta)}")
    print(f"Participants: {meta['participant_id'].unique()}")

    # Test single shot loading
    print("\nLoading first shot...")
    metadata, ts = load_single_shot(0, train=True)
    print(f"Metadata: {metadata}")
    print(f"Timeseries shape: {ts.shape}")
    print(f"Timeseries dtype: {ts.dtype}")
    print(f"NaN count: {np.isnan(ts).sum()}")

    # Test iteration (just 3 shots)
    print("\nIterating first 3 shots...")
    for i, (meta, ts) in enumerate(iterate_shots(train=True, chunk_size=3)):
        print(f"  Shot {i}: participant={meta['participant_id']}, shape={ts.shape}")
        if i >= 2:
            break

    print("\nData loader test complete!")

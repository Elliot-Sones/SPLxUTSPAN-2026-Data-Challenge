"""
Core physics simulation components.
"""

from .simulator import BasketballSimulator, create_simulator
from .scale_calibration import calibrate_scale_factor, get_keypoint_indices, validate_scale
from .release_extraction import (
    detect_release_frame,
    extract_release_position,
    extract_release_velocity,
    estimate_backspin,
    extract_all_release_params,
    PLAYER_RELEASE_WINDOWS
)
from .target_mapping import TargetMapper, calibrate_mean_correction, apply_corrections

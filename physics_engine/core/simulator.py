"""
MuJoCo Basketball Ball Trajectory Simulator

Core simulation engine for predicting shot outcomes from release parameters.
"""

import numpy as np
import mujoco
from pathlib import Path
from typing import Tuple, Optional, List


class BasketballSimulator:
    """
    Simulates basketball trajectories using MuJoCo physics.

    Uses 1ms timestep with RK4 integrator for accurate projectile motion.
    8-segment capsule rim for realistic collision detection.
    """

    # Court constants (meters)
    HOOP_X = 0.0           # Hoop at origin
    HOOP_Y = 0.15          # 15cm forward of backboard
    HOOP_Z = 3.05          # 10 feet = 3.05 meters
    HOOP_RADIUS = 0.2286   # 9 inches
    FREE_THROW_DIST = 4.57 # 15 feet
    BALL_RADIUS = 0.12     # Standard basketball

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize simulator with MuJoCo model.

        Args:
            model_path: Path to MuJoCo XML. If None, uses default basketball_court.xml
        """
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "basketball_court.xml"

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Verify model loaded correctly
        assert self.model.nq == 7, f"Expected 7 DOF (ball free joint), got {self.model.nq}"

    def reset(self):
        """Reset simulation state."""
        mujoco.mj_resetData(self.model, self.data)

    def simulate_shot(
        self,
        release_pos: np.ndarray,
        release_vel: np.ndarray,
        backspin: float = 0.0,
        max_time: float = 3.0
    ) -> Tuple[Optional[np.ndarray], Optional[float], List[np.ndarray]]:
        """
        Simulate ball trajectory from release to hoop crossing.

        Args:
            release_pos: [x, y, z] in meters (x negative = before hoop)
            release_vel: [vx, vy, vz] in m/s (vx positive = toward hoop)
            backspin: Angular velocity in rad/s (positive = backspin)
            max_time: Maximum simulation time in seconds

        Returns:
            landing_pos: [y, z] at hoop plane crossing, or None if miss
            entry_angle: Angle of ball velocity at entry (degrees), or None
            trajectory: List of ball positions [x, y, z]
        """
        self.reset()

        # Set initial ball position (qpos[0:3])
        self.data.qpos[0:3] = release_pos
        # Set quaternion to identity (no initial rotation)
        self.data.qpos[3:7] = [1, 0, 0, 0]

        # Set initial velocity (qvel[0:3] = linear, qvel[3:6] = angular)
        self.data.qvel[0:3] = release_vel
        # Backspin is rotation around Y axis (lateral axis)
        self.data.qvel[3:6] = [0, backspin, 0]

        trajectory = []
        prev_x = release_pos[0]
        prev_pos = release_pos.copy()

        while self.data.time < max_time:
            mujoco.mj_step(self.model, self.data)

            pos = self.data.qpos[0:3].copy()
            vel = self.data.qvel[0:3].copy()
            trajectory.append(pos)

            # Check if ball crossed hoop plane (x goes from negative to positive)
            # We use x = 0.1 to account for ball radius
            if prev_x < 0.1 and pos[0] >= 0.1:
                # Linear interpolation to exact crossing point
                alpha = (0.1 - prev_x) / (pos[0] - prev_x + 1e-8)
                crossing_y = prev_pos[1] + alpha * (pos[1] - prev_pos[1])
                crossing_z = prev_pos[2] + alpha * (pos[2] - prev_pos[2])

                landing_pos = np.array([crossing_y, crossing_z])

                # Entry angle: angle of velocity vector from horizontal
                # Negative because ball is coming down
                entry_angle = np.degrees(np.arctan2(-vel[2], vel[0]))

                return landing_pos, entry_angle, trajectory

            # Ball hit ground (too low)
            if pos[2] < self.BALL_RADIUS:
                return None, None, trajectory

            prev_x = pos[0]
            prev_pos = pos.copy()

        # Timeout - ball never reached hoop
        return None, None, trajectory

    def validate_physics(self) -> bool:
        """
        Verify simulation matches analytical projectile motion.

        Returns True if physics validation passes.
        """
        # Known release parameters
        release_pos = np.array([-4.57, 0.0, 2.0])
        release_vel = np.array([6.5, 0.0, 4.5])  # ~7.7 m/s, ~35 degrees

        # Analytical solution (no drag, no spin effects)
        def analytical_pos(t):
            x = release_pos[0] + release_vel[0] * t
            y = release_pos[1] + release_vel[1] * t
            z = release_pos[2] + release_vel[2] * t - 0.5 * 9.81 * t**2
            return np.array([x, y, z])

        # Run simulation
        landing, _, trajectory = self.simulate_shot(release_pos, release_vel, backspin=0)

        # Compare at multiple time points
        max_error = 0.0
        for i in range(0, min(500, len(trajectory)), 50):
            t = i * 0.001  # 1ms timestep
            mujoco_pos = trajectory[i]
            expected_pos = analytical_pos(t)
            error = np.linalg.norm(mujoco_pos - expected_pos)
            max_error = max(max_error, error)

        # Allow 1cm tolerance (air resistance, numerical precision)
        passed = max_error < 0.01

        if not passed:
            print(f"Physics validation FAILED: max error = {max_error:.4f}m")

        return passed


def create_simulator() -> BasketballSimulator:
    """Factory function to create simulator with default settings."""
    return BasketballSimulator()


if __name__ == "__main__":
    # Quick test
    print("Testing basketball simulator...")

    sim = create_simulator()

    # Test physics validation
    print("\nPhysics validation...")
    if sim.validate_physics():
        print("  PASSED")
    else:
        print("  FAILED")

    # Test a typical free throw
    print("\nSimulating free throw...")
    release_pos = np.array([-4.57, 0.0, 2.1])  # Free throw line, 2.1m release height
    release_vel = np.array([7.0, 0.0, 4.0])    # ~8 m/s, ~30 degrees up

    landing, entry_angle, trajectory = sim.simulate_shot(release_pos, release_vel, backspin=15)

    if landing is not None:
        print(f"  Landing position: y={landing[0]:.3f}m, z={landing[1]:.3f}m")
        print(f"  Entry angle: {entry_angle:.1f} degrees")
        print(f"  Trajectory points: {len(trajectory)}")

        # Check if make or miss
        y_dev = abs(landing[0] - sim.HOOP_Y)
        z_dev = abs(landing[1] - sim.HOOP_Z)
        within_rim = y_dev < sim.HOOP_RADIUS and z_dev < 0.3
        print(f"  Within rim: {within_rim}")
    else:
        print("  Shot did not reach hoop (air ball or short)")

    print("\nSimulator test complete!")

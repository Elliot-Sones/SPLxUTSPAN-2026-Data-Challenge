"""
MuJoCo Basketball Simulation

Map our keypoint data to a physics simulation to predict ball trajectory.

Approach:
1. Create basketball + hoop scene
2. Extract hand position/velocity from our keypoints
3. Simulate ball release and trajectory
4. Tune simulation parameters using training data outcomes
5. Apply to test data
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_loader import iterate_shots, load_scalers, get_keypoint_columns

PROJECT_DIR = Path(__file__).parent.parent

# Basketball court constants (in meters for MuJoCo)
HOOP_HEIGHT = 3.05  # 10 feet = 3.05m
HOOP_RADIUS = 0.2286  # 9 inches = 0.2286m
BALL_RADIUS = 0.12  # ~4.7 inches
FREE_THROW_DIST = 4.57  # 15 feet = 4.57m


def create_basketball_scene():
    """Create MuJoCo XML for basketball scene."""
    xml = """
    <mujoco model="basketball">
        <option gravity="0 0 -9.81" timestep="0.002"/>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
            <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="512" height="512"/>
            <material name="floor_mat" texture="floor_tex" texrepeat="10 10"/>
            <material name="ball_mat" rgba="1 0.5 0 1"/>
            <material name="hoop_mat" rgba="1 0.3 0 1"/>
            <material name="backboard_mat" rgba="1 1 1 0.8"/>
        </asset>

        <worldbody>
            <!-- Floor -->
            <geom type="plane" size="10 10 0.1" material="floor_mat"/>

            <!-- Backboard -->
            <body name="backboard" pos="0 0 3.35">
                <geom type="box" size="0.9 0.02 0.6" material="backboard_mat"/>
            </body>

            <!-- Hoop -->
            <body name="hoop" pos="0 0.4 3.05">
                <geom type="cylinder" size="0.2286 0.01" material="hoop_mat"
                      euler="90 0 0"/>
            </body>

            <!-- Basketball -->
            <body name="ball" pos="-4 0 2">
                <freejoint name="ball_joint"/>
                <geom type="sphere" size="0.12" mass="0.625" material="ball_mat"/>
            </body>

            <!-- Shooter position marker -->
            <body name="shooter" pos="-4.57 0 0">
                <geom type="cylinder" size="0.3 0.01" rgba="0 1 0 0.5"/>
            </body>
        </worldbody>
    </mujoco>
    """
    return xml


def get_keypoint_indices():
    """Get keypoint name to index mapping."""
    cols = get_keypoint_columns()
    keypoint_names = []
    for col in cols:
        name = col.rsplit("_", 1)[0]
        if name not in keypoint_names:
            keypoint_names.append(name)
    return {name: i for i, name in enumerate(keypoint_names)}


def extract_release_params(timeseries, keypoint_idx, scale_factor=1.0):
    """
    Extract ball release parameters from keypoint data.

    Returns:
        release_pos: [x, y, z] position in meters
        release_vel: [vx, vy, vz] velocity in m/s
    """
    n_frames = len(timeseries)

    def get_joint(name, frame):
        if name not in keypoint_idx or frame >= n_frames:
            return np.zeros(3)
        idx = keypoint_idx[name]
        return timeseries[frame, idx*3:(idx+1)*3]

    # Release frame analysis (frames 150-155)
    release_frame = 153
    pre_release_frame = 150

    wrist_release = get_joint("right_wrist", release_frame)
    wrist_pre = get_joint("right_wrist", pre_release_frame)

    if not np.any(wrist_release) or not np.any(wrist_pre):
        return None, None

    # Scale from normalized to meters
    # Assume typical release height ~2m, our normalized z is around 0.8-0.9
    height_scale = 2.0 / (wrist_release[2] + 0.01)

    release_pos = wrist_release * height_scale * scale_factor

    # Velocity: 3 frames at 60fps = 0.05 seconds
    dt = 3 / 60.0
    velocity = (wrist_release - wrist_pre) / dt * height_scale * scale_factor

    # Adjust coordinate system: our data might have different axes
    # MuJoCo: x=forward, y=left/right, z=up
    # Need to map appropriately

    return release_pos, velocity


def simulate_shot(model, data, release_pos, release_vel, max_time=3.0):
    """
    Simulate a basketball shot.

    Returns:
        landing_pos: where ball crosses hoop plane
        success: whether ball went through hoop
    """
    # Reset simulation
    mujoco.mj_resetData(model, data)

    # Set ball initial position and velocity
    # Ball joint is the first free joint
    ball_qpos_start = 0  # Free joint: 3 pos + 4 quat
    ball_qvel_start = 0  # Free joint: 3 linear + 3 angular

    # Set position (x, y, z)
    data.qpos[ball_qpos_start:ball_qpos_start+3] = release_pos
    # Set quaternion (w, x, y, z) - no rotation
    data.qpos[ball_qpos_start+3:ball_qpos_start+7] = [1, 0, 0, 0]

    # Set velocity
    data.qvel[ball_qvel_start:ball_qvel_start+3] = release_vel
    data.qvel[ball_qvel_start+3:ball_qvel_start+6] = [0, 0, 0]  # No angular velocity

    # Run simulation
    trajectory = []
    hoop_z = HOOP_HEIGHT
    crossed_hoop_plane = False
    landing_pos = None

    while data.time < max_time:
        mujoco.mj_step(model, data)

        ball_pos = data.qpos[ball_qpos_start:ball_qpos_start+3].copy()
        ball_vel = data.qvel[ball_qvel_start:ball_qvel_start+3].copy()
        trajectory.append(ball_pos)

        # Check if ball crossed hoop plane (x ≈ 0, going forward)
        if ball_pos[0] > -0.5 and ball_pos[0] < 0.5 and not crossed_hoop_plane:
            if abs(ball_pos[2] - hoop_z) < 0.3:  # Near hoop height
                crossed_hoop_plane = True
                landing_pos = ball_pos.copy()

        # Check if ball hit ground
        if ball_pos[2] < BALL_RADIUS:
            break

    return np.array(trajectory), landing_pos


def compute_shot_outcome(landing_pos, hoop_pos=np.array([0, 0.4, 3.05])):
    """
    Compute shot outcome metrics similar to our targets.

    Returns:
        angle_error: lateral deviation (related to our 'angle' target)
        depth_error: front/back deviation (related to our 'depth' target)
        lr_error: left/right deviation (related to our 'left_right' target)
    """
    if landing_pos is None:
        return None, None, None

    # Deviation from hoop center
    dx = landing_pos[0] - hoop_pos[0]  # depth (front/back)
    dy = landing_pos[1] - hoop_pos[1]  # left/right
    dz = landing_pos[2] - hoop_pos[2]  # height difference

    # Angle error: based on lateral deviation
    angle_error = np.degrees(np.arctan2(dy, 4.57))  # angle from shooter position

    # Depth error: how far in front/behind hoop
    depth_error = dx

    # Left/right error
    lr_error = dy

    return angle_error, depth_error, lr_error


def main():
    print("="*80)
    print("MUJOCO BASKETBALL SIMULATION")
    print("="*80)

    # Create scene
    xml = create_basketball_scene()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    print(f"Model created: {model.nq} qpos, {model.nv} qvel")

    # Load keypoint indices
    keypoint_idx = get_keypoint_indices()
    scalers = load_scalers()

    # Test with a few training samples
    print("\nTesting simulation with training data...")

    results = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 10:  # Test with first 10 samples
            break

        # Extract release parameters
        release_pos, release_vel = extract_release_params(timeseries, keypoint_idx)

        if release_pos is None:
            print(f"  Shot {i}: No valid release data")
            continue

        # Adjust for MuJoCo coordinate system
        # Put shooter at free throw line
        release_pos[0] = -FREE_THROW_DIST + release_pos[0] * 0.5  # x: forward
        release_pos[1] = release_pos[1] * 0.5  # y: left/right
        release_pos[2] = max(1.5, min(2.5, release_pos[2]))  # z: height (clamp)

        # Scale velocity for realistic shot
        speed = np.linalg.norm(release_vel)
        if speed > 0:
            # Target speed ~7 m/s for free throw
            release_vel = release_vel / speed * 7.0
            # Ensure upward component
            release_vel[2] = max(3.0, release_vel[2])
            # Forward component toward hoop
            release_vel[0] = max(4.0, abs(release_vel[0]))

        # Simulate
        trajectory, landing_pos = simulate_shot(model, data, release_pos, release_vel)

        # Compute outcome
        angle_err, depth_err, lr_err = compute_shot_outcome(landing_pos)

        # Get actual targets
        actual_angle = scalers['angle'].transform([[metadata['angle']]])[0, 0]
        actual_depth = scalers['depth'].transform([[metadata['depth']]])[0, 0]
        actual_lr = scalers['left_right'].transform([[metadata['left_right']]])[0, 0]

        print(f"  Shot {i}: release_z={release_pos[2]:.2f}m, vel={np.linalg.norm(release_vel):.1f}m/s")
        print(f"    Landing: {landing_pos}")
        print(f"    Simulated: angle={angle_err:.2f}, depth={depth_err:.2f}, lr={lr_err:.2f}")
        print(f"    Actual:    angle={actual_angle:.2f}, depth={actual_depth:.2f}, lr={actual_lr:.2f}")

        if landing_pos is not None:
            results.append({
                'sim_angle': angle_err,
                'sim_depth': depth_err,
                'sim_lr': lr_err,
                'actual_angle': actual_angle,
                'actual_depth': actual_depth,
                'actual_lr': actual_lr,
            })

    # Check correlation
    if len(results) > 3:
        import pandas as pd
        df = pd.DataFrame(results)
        print("\n" + "="*60)
        print("CORRELATION BETWEEN SIMULATED AND ACTUAL")
        print("="*60)

        for target in ['angle', 'depth', 'lr']:
            corr = np.corrcoef(df[f'sim_{target}'], df[f'actual_{target}'])[0, 1]
            print(f"  {target}: {corr:.4f}")

    print("\nSimulation test complete.")
    print("Next steps:")
    print("1. Tune scale factors and coordinate mapping")
    print("2. Optimize release velocity estimation")
    print("3. Run on all training data to calibrate")
    print("4. Apply to test data")


if __name__ == "__main__":
    main()

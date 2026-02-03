"""
MuJoCo Ball-Hand Contact Simulation

Simulate the ball resting on the hand from the start of the shot.
As we move the hand according to skeleton data, MuJoCo contact physics will:
1. Push the ball through contact forces
2. Accumulate velocity in the ball
3. Naturally separate when contact force becomes zero

This gives us the TRUE release velocity from physics, not from differentiation.
"""

import numpy as np
import mujoco
from pathlib import Path
from scipy.signal import savgol_filter
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

# Physical constants (all in SI units: meters, kg, seconds)
BALL_RADIUS = 0.12  # 9.4 inch diameter = 0.12m radius
BALL_MASS = 0.625   # kg (22 oz)
GRAVITY = 9.81      # m/s^2

# Hoop position in meters (converted from feet)
# Original: [5.25, -25, 10] feet
HOOP_POSITION_M = np.array([5.25 * 0.3048, -25.0 * 0.3048, 10.0 * 0.3048])  # [1.6, -7.62, 3.05] meters

# Data is in feet, convert to meters
FEET_TO_METERS = 0.3048

FPS = 60
DT = 1.0 / FPS  # 16.67 ms per frame


def create_mujoco_model():
    """
    Create MuJoCo model with ball and a kinematic hand.

    The hand is a simple plate/paddle that we move according to skeleton data.
    The ball sits on top and is pushed by contact forces.
    """
    xml = """
    <mujoco model="basketball_contact">
        <option gravity="0 0 -9.81" timestep="0.0005" integrator="RK4">
            <flag contact="enable"/>
        </option>

        <default>
            <geom condim="3" friction="0.8 0.1 0.1"/>
        </default>

        <worldbody>
            <!-- Ground plane -->
            <geom name="ground" type="plane" size="20 20 0.1" rgba="0.3 0.3 0.3 1"/>

            <!-- Hand - kinematic body that we control -->
            <!-- Using a flat box to represent the palm/fingers surface -->
            <body name="hand" pos="0 0 2">
                <joint name="hand_x" type="slide" axis="1 0 0"/>
                <joint name="hand_y" type="slide" axis="0 1 0"/>
                <joint name="hand_z" type="slide" axis="0 0 1"/>
                <joint name="hand_rx" type="hinge" axis="1 0 0"/>
                <joint name="hand_ry" type="hinge" axis="0 1 0"/>
                <!-- Hand surface - slightly cupped to hold ball -->
                <geom name="palm" type="box" size="0.08 0.06 0.01" pos="0 0 0"
                      rgba="0.9 0.7 0.5 1" mass="0.5"/>
                <!-- Finger tips that push the ball -->
                <geom name="fingers" type="capsule" size="0.015 0.05" pos="0 0.07 0.02"
                      euler="90 0 0" rgba="0.9 0.7 0.5 1" mass="0.1"/>
            </body>

            <!-- Basketball - free body -->
            <body name="ball" pos="0 0 2.15">
                <freejoint name="ball_joint"/>
                <geom name="ball_geom" type="sphere" size="0.12" mass="0.625"
                      rgba="1 0.5 0 1" friction="0.8 0.1 0.1"/>
            </body>

            <!-- Hoop for reference (no collision) -->
            <body name="hoop" pos="1.6 -7.62 3.05">
                <geom type="cylinder" size="0.23 0.005" euler="90 0 0"
                      rgba="1 0.3 0 0.5" contype="0" conaffinity="0"/>
            </body>
        </worldbody>

        <actuator>
            <!-- Position control for hand -->
            <position name="hand_x_ctrl" joint="hand_x" kp="1000"/>
            <position name="hand_y_ctrl" joint="hand_y" kp="1000"/>
            <position name="hand_z_ctrl" joint="hand_z" kp="1000"/>
            <position name="hand_rx_ctrl" joint="hand_rx" kp="100"/>
            <position name="hand_ry_ctrl" joint="hand_ry" kp="100"/>
        </actuator>

        <sensor>
            <!-- Contact force sensor -->
            <touch name="palm_touch" site="palm_site"/>
        </sensor>

        <worldbody>
            <!-- Add site for touch sensor -->
            <body name="hand">
                <site name="palm_site" pos="0 0 0.01" size="0.08 0.06 0.01" type="box"/>
            </body>
        </worldbody>
    </mujoco>
    """

    # Simpler model without the sensor issue
    xml_simple = """
    <mujoco model="basketball_contact">
        <option gravity="0 0 -9.81" timestep="0.001" integrator="RK4">
            <flag contact="enable"/>
        </option>

        <default>
            <geom condim="3" friction="1.0 0.1 0.1"/>
        </default>

        <worldbody>
            <!-- Ground plane -->
            <geom name="ground" type="plane" size="20 20 0.1" rgba="0.3 0.3 0.3 1"/>

            <!-- Hand platform - kinematic body that we control -->
            <body name="hand" pos="0 0 1.5">
                <joint name="hand_x" type="slide" axis="1 0 0" damping="10"/>
                <joint name="hand_y" type="slide" axis="0 1 0" damping="10"/>
                <joint name="hand_z" type="slide" axis="0 0 1" damping="10"/>
                <!-- Tilted platform to represent cupped hand -->
                <geom name="palm" type="box" size="0.06 0.05 0.015" pos="0 0 0"
                      euler="20 0 0" rgba="0.9 0.7 0.5 1" mass="1.0"/>
                <!-- Fingertips -->
                <geom name="finger1" type="sphere" size="0.015" pos="0.03 0.05 0.03" rgba="0.9 0.7 0.5 1"/>
                <geom name="finger2" type="sphere" size="0.015" pos="0.01 0.055 0.035" rgba="0.9 0.7 0.5 1"/>
                <geom name="finger3" type="sphere" size="0.015" pos="-0.01 0.055 0.035" rgba="0.9 0.7 0.5 1"/>
                <geom name="finger4" type="sphere" size="0.015" pos="-0.03 0.05 0.03" rgba="0.9 0.7 0.5 1"/>
            </body>

            <!-- Basketball - free body -->
            <body name="ball" pos="0 0 1.7">
                <freejoint name="ball_joint"/>
                <geom name="ball_geom" type="sphere" size="0.12" mass="0.625"
                      rgba="1 0.5 0 1" friction="1.0 0.1 0.1"/>
            </body>
        </worldbody>

        <actuator>
            <!-- Velocity control for smooth motion -->
            <velocity name="hand_vx" joint="hand_x" kv="100"/>
            <velocity name="hand_vy" joint="hand_y" kv="100"/>
            <velocity name="hand_vz" joint="hand_z" kv="100"/>
        </actuator>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(xml_simple)
    data = mujoco.MjData(model)
    return model, data


def get_keypoint_map(keypoint_cols):
    """Build mapping from keypoint names to column indices."""
    keypoint_map = {}
    for i, col in enumerate(keypoint_cols):
        parts = col.rsplit('_', 1)
        if len(parts) == 2:
            name, axis = parts
            if name not in keypoint_map:
                keypoint_map[name] = {}
            keypoint_map[name][axis] = i
    return keypoint_map


def get_position(timeseries, keypoint_map, name, frame):
    """Get 3D position of a keypoint at a frame."""
    if name not in keypoint_map:
        return None
    km = keypoint_map[name]
    if 'x' not in km or 'y' not in km or 'z' not in km:
        return None
    pos = np.array([
        timeseries[frame, km['x']],
        timeseries[frame, km['y']],
        timeseries[frame, km['z']]
    ])
    if np.any(np.isnan(pos)):
        return None
    return pos


def get_hand_position(timeseries, keypoint_map, frame):
    """
    Get hand position from fingertip average.
    Returns position in METERS.
    """
    fingertips = [
        'right_second_finger_distal',  # Index
        'right_third_finger_distal',   # Middle
        'right_fourth_finger_distal',  # Ring
    ]

    positions = []
    for tip in fingertips:
        pos = get_position(timeseries, keypoint_map, tip, frame)
        if pos is not None:
            positions.append(pos)

    if len(positions) < 2:
        # Fallback to wrist
        wrist = get_position(timeseries, keypoint_map, 'right_wrist', frame)
        if wrist is not None:
            return wrist * FEET_TO_METERS
        return None

    # Average position, convert feet to meters
    avg_pos = np.mean(positions, axis=0) * FEET_TO_METERS
    return avg_pos


def check_ball_contact(model, data):
    """
    Check if ball is in contact with hand.
    Returns True if there's contact, False otherwise.
    """
    # Check all contacts
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

        # Check if ball is touching any hand geometry
        hand_geoms = ['palm', 'finger1', 'finger2', 'finger3', 'finger4']
        if (geom1 == 'ball_geom' and geom2 in hand_geoms) or \
           (geom2 == 'ball_geom' and geom1 in hand_geoms):
            return True

    return False


def get_ball_state(model, data):
    """Get ball position and velocity from MuJoCo."""
    ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'ball')

    # Position (center of mass)
    pos = data.xpos[ball_body_id].copy()

    # Velocity (linear)
    vel = data.cvel[ball_body_id, 3:6].copy()  # Linear velocity part

    # Alternative: get velocity from qvel for freejoint
    # Ball freejoint is first, so qvel[0:3] is linear velocity
    vel = data.qvel[0:3].copy()

    return pos, vel


def simulate_shot(timeseries, keypoint_map, model, data, verbose=False):
    """
    Simulate a shot using MuJoCo contact physics.

    1. Position ball on hand at start
    2. Move hand according to skeleton data
    3. Ball follows via contact physics
    4. Detect when ball separates (contact lost)
    5. Record ball velocity at separation
    """

    # Reset simulation
    mujoco.mj_resetData(model, data)

    # Find the start of the shooting motion (when hand starts rising)
    hand_positions = []
    valid_frames = []

    for frame in range(50, 200):
        pos = get_hand_position(timeseries, keypoint_map, frame)
        if pos is not None:
            hand_positions.append(pos)
            valid_frames.append(frame)

    if len(hand_positions) < 20:
        return None

    hand_positions = np.array(hand_positions)
    valid_frames = np.array(valid_frames)

    # Compute hand velocity to find shot start
    hand_vz = savgol_filter(hand_positions[:, 2], 11, 3, deriv=1) * FPS

    # Find when upward motion starts (vz becomes significantly positive)
    shot_start_idx = 0
    for i in range(len(hand_vz)):
        if hand_vz[i] > 0.5:  # 0.5 m/s threshold
            shot_start_idx = max(0, i - 5)  # Start a bit before
            break

    start_frame = valid_frames[shot_start_idx]

    # Initial hand position
    init_hand_pos = hand_positions[shot_start_idx]

    # Set initial hand position in MuJoCo
    # The hand joint positions are relative to initial body position
    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')

    # Set hand position via joint positions (qpos)
    # Joints: hand_x, hand_y, hand_z
    data.qpos[6] = init_hand_pos[0]  # hand_x (after ball's 7 DOF: 3 pos + 4 quat)
    data.qpos[7] = init_hand_pos[1]  # hand_y
    data.qpos[8] = init_hand_pos[2]  # hand_z

    # Position ball on hand
    data.qpos[0] = init_hand_pos[0]      # ball x
    data.qpos[1] = init_hand_pos[1]      # ball y
    data.qpos[2] = init_hand_pos[2] + BALL_RADIUS + 0.02  # ball z (on top of hand)
    data.qpos[3:7] = [1, 0, 0, 0]        # ball orientation (quaternion)

    # Zero velocities
    data.qvel[:] = 0

    # Forward to initialize
    mujoco.mj_forward(model, data)

    if verbose:
        print(f"  Shot start frame: {start_frame}")
        print(f"  Initial hand pos: {init_hand_pos}")
        ball_pos, ball_vel = get_ball_state(model, data)
        print(f"  Initial ball pos: {ball_pos}")

    # Simulation loop
    release_detected = False
    release_frame = None
    release_pos = None
    release_vel = None
    contact_frames = 0
    no_contact_frames = 0

    frame_idx = shot_start_idx
    sim_substeps = int(DT / model.opt.timestep)  # Substeps per data frame

    # Precompute hand velocities for actuator control
    hand_vx = savgol_filter(hand_positions[:, 0], 11, 3, deriv=1) * FPS
    hand_vy = savgol_filter(hand_positions[:, 1], 11, 3, deriv=1) * FPS
    hand_vz = savgol_filter(hand_positions[:, 2], 11, 3, deriv=1) * FPS

    while frame_idx < len(valid_frames) - 1 and not release_detected:
        # Target hand velocity for this frame
        target_vx = hand_vx[frame_idx]
        target_vy = hand_vy[frame_idx]
        target_vz = hand_vz[frame_idx]

        # Set actuator controls (velocity targets)
        data.ctrl[0] = target_vx  # hand_vx
        data.ctrl[1] = target_vy  # hand_vy
        data.ctrl[2] = target_vz  # hand_vz

        # Step simulation (multiple substeps per frame)
        for _ in range(sim_substeps):
            mujoco.mj_step(model, data)

        # Check contact
        in_contact = check_ball_contact(model, data)

        if in_contact:
            contact_frames += 1
            no_contact_frames = 0
        else:
            no_contact_frames += 1

            # If we had contact and now lost it for several frames, that's release
            if contact_frames > 5 and no_contact_frames >= 3:
                release_detected = True
                release_frame = valid_frames[frame_idx]
                release_pos, release_vel = get_ball_state(model, data)

                if verbose:
                    print(f"  Release detected at frame {release_frame}")
                    print(f"  Ball pos: {release_pos}")
                    print(f"  Ball vel: {release_vel}")
                    print(f"  Ball speed: {np.linalg.norm(release_vel):.2f} m/s")

        frame_idx += 1

    if not release_detected:
        # Ball never released - use last state
        release_frame = valid_frames[frame_idx - 1]
        release_pos, release_vel = get_ball_state(model, data)

        if verbose:
            print(f"  No release detected, using final state at frame {release_frame}")
            print(f"  Ball vel: {release_vel}, speed: {np.linalg.norm(release_vel):.2f} m/s")

    return {
        'release_frame': release_frame,
        'release_pos': release_pos,  # meters
        'release_vel': release_vel,  # m/s
        'release_speed': np.linalg.norm(release_vel),
        'contact_frames': contact_frames,
    }


def main():
    print("=" * 80)
    print("MUJOCO BALL-HAND CONTACT SIMULATION")
    print("=" * 80)
    print("\nSimulating ball resting on hand, pushed by contact forces")
    print("Release is detected when contact is lost")

    # Create MuJoCo model
    print("\nCreating MuJoCo model...")
    model, data = create_mujoco_model()
    print(f"  Model timestep: {model.opt.timestep} s")
    print(f"  DOF: {model.nv}")

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    print("\nProcessing shots...")
    results = []

    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        if i >= 10:  # Process first 10 shots
            break

        print(f"\nShot {i+1}: {metadata['id'][:8]} (Player {metadata['participant_id']})")

        result = simulate_shot(timeseries, keypoint_map, model, data, verbose=True)

        if result is not None:
            result['shot_id'] = metadata['id']
            result['true_angle'] = metadata['angle']
            result['true_depth'] = metadata['depth']
            result['true_lr'] = metadata['left_right']
            results.append(result)

            # Convert velocity to ft/s for comparison
            vel_fps = result['release_vel'] / FEET_TO_METERS
            print(f"  Velocity (ft/s): [{vel_fps[0]:.1f}, {vel_fps[1]:.1f}, {vel_fps[2]:.1f}]")
            print(f"  Speed (ft/s): {result['release_speed'] / FEET_TO_METERS:.1f}")

    # Summary
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        speeds = [r['release_speed'] for r in results]
        speeds_fps = [s / FEET_TO_METERS for s in speeds]

        print(f"\nRelease speeds:")
        print(f"  Mean: {np.mean(speeds):.2f} m/s ({np.mean(speeds_fps):.1f} ft/s)")
        print(f"  Std:  {np.std(speeds):.2f} m/s ({np.std(speeds_fps):.1f} ft/s)")
        print(f"  Min:  {np.min(speeds):.2f} m/s ({np.min(speeds_fps):.1f} ft/s)")
        print(f"  Max:  {np.max(speeds):.2f} m/s ({np.max(speeds_fps):.1f} ft/s)")

        print(f"\nRequired for free throw: ~7.3 m/s (24 ft/s)")


if __name__ == "__main__":
    main()

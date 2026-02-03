"""
Visualize the actual hand trajectory and ball physics.

Shows:
1. Hand position over time from skeleton data
2. Ball trajectory from MuJoCo simulation
3. Where the hoop should be
4. Velocity vectors
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from scipy.signal import savgol_filter
import sys

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loader import iterate_shots, get_keypoint_columns

FEET_TO_METERS = 0.3048
FPS = 60


def get_keypoint_map(keypoint_cols):
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


def simulate_ball_trajectory(release_pos, release_vel, duration=1.5, dt=0.01):
    """Simulate ball trajectory under gravity."""
    g = 32.174  # ft/s^2

    trajectory = [release_pos.copy()]
    pos = release_pos.copy()
    vel = release_vel.copy()

    t = 0
    while t < duration and pos[2] > 0:
        vel[2] -= g * dt  # Gravity
        pos = pos + vel * dt
        trajectory.append(pos.copy())
        t += dt

    return np.array(trajectory)


def main():
    print("=" * 80)
    print("TRAJECTORY VISUALIZATION")
    print("=" * 80)

    keypoint_cols = get_keypoint_columns()
    keypoint_map = get_keypoint_map(keypoint_cols)

    # Get first shot
    for i, (metadata, timeseries) in enumerate(iterate_shots(train=True)):
        shot_id = metadata['id'][:8]
        print(f"\nProcessing shot: {shot_id}")

        # Extract hand trajectory (in FEET)
        hand_positions = []
        wrist_positions = []
        shoulder_positions = []
        frames = []

        for frame in range(50, 200):
            finger = get_position(timeseries, keypoint_map, 'right_third_finger_distal', frame)
            wrist = get_position(timeseries, keypoint_map, 'right_wrist', frame)
            shoulder = get_position(timeseries, keypoint_map, 'right_shoulder', frame)

            if finger is not None and wrist is not None:
                hand_positions.append(finger)
                wrist_positions.append(wrist)
                shoulder_positions.append(shoulder if shoulder is not None else wrist)
                frames.append(frame)

        if len(hand_positions) < 30:
            continue

        hand_positions = np.array(hand_positions)
        wrist_positions = np.array(wrist_positions)
        shoulder_positions = np.array(shoulder_positions)
        frames = np.array(frames)

        # Compute velocities (ft/s)
        window = 11
        hand_vel = np.zeros_like(hand_positions)
        for j in range(3):
            hand_vel[:, j] = savgol_filter(hand_positions[:, j], window, 3, deriv=1) * FPS

        # Find peak upward velocity
        vz = hand_vel[:, 2]
        peak_idx = np.argmax(vz)
        peak_frame = frames[peak_idx]

        release_pos = hand_positions[peak_idx]
        release_vel = hand_vel[peak_idx]

        print(f"\nRelease at frame {peak_frame}:")
        print(f"  Position: ({release_pos[0]:.2f}, {release_pos[1]:.2f}, {release_pos[2]:.2f}) ft")
        print(f"  Velocity: ({release_vel[0]:.2f}, {release_vel[1]:.2f}, {release_vel[2]:.2f}) ft/s")
        print(f"  Horizontal speed: {np.sqrt(release_vel[0]**2 + release_vel[1]**2):.2f} ft/s")
        print(f"  Total speed: {np.linalg.norm(release_vel):.2f} ft/s")

        # Estimate player position (mid hip)
        player_pos = get_position(timeseries, keypoint_map, 'mid_hip', peak_frame)
        if player_pos is None:
            player_pos = np.array([release_pos[0], release_pos[1], 3.0])

        # Hoop position: 15 feet from player toward -X (typical setup)
        # But we need to figure out which direction!
        hoop_height = 10.0  # feet
        hoop_distance = 15.0  # feet from free throw line

        # Simulate ball trajectory with current velocity
        ball_traj_current = simulate_ball_trajectory(release_pos, release_vel)

        # What velocity WOULD be needed to reach hoop?
        # If hoop is 15 ft away in -X direction at height 10 ft:
        hoop_pos_x = np.array([release_pos[0] - hoop_distance, release_pos[1], hoop_height])
        hoop_pos_y = np.array([release_pos[0], release_pos[1] - hoop_distance, hoop_height])

        # Calculate required velocity for hoop in -X direction
        dx = hoop_pos_x[0] - release_pos[0]  # -15 ft
        dz = hoop_pos_x[2] - release_pos[2]  # ~2 ft up

        # For projectile: x = x0 + vx*t, z = z0 + vz*t - 0.5*g*t^2
        # Assume flight time ~0.8s for a typical shot
        t_flight = 0.8
        g = 32.174
        required_vx = dx / t_flight
        required_vz = (dz + 0.5 * g * t_flight**2) / t_flight

        print(f"\nRequired velocity to reach hoop (15ft in -X):")
        print(f"  Vx: {required_vx:.2f} ft/s")
        print(f"  Vz: {required_vz:.2f} ft/s")
        print(f"  Total: {np.sqrt(required_vx**2 + required_vz**2):.2f} ft/s")

        # Simulate with required velocity
        required_vel = np.array([required_vx, 0, required_vz])
        ball_traj_required = simulate_ball_trajectory(release_pos, required_vel)

        # Create figure
        fig = plt.figure(figsize=(16, 12))

        # 3D trajectory plot
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')

        # Hand trajectory
        ax1.plot(hand_positions[:, 0], hand_positions[:, 1], hand_positions[:, 2],
                 'b-', linewidth=2, label='Hand trajectory')
        ax1.scatter(release_pos[0], release_pos[1], release_pos[2],
                    c='g', s=100, marker='o', label='Release point')

        # Ball trajectory from data velocity
        ax1.plot(ball_traj_current[:, 0], ball_traj_current[:, 1], ball_traj_current[:, 2],
                 'r-', linewidth=2, label='Ball (data velocity)')

        # Ball trajectory with required velocity
        ax1.plot(ball_traj_required[:, 0], ball_traj_required[:, 1], ball_traj_required[:, 2],
                 'g--', linewidth=2, label='Ball (required velocity)')

        # Hoop positions
        theta = np.linspace(0, 2*np.pi, 50)
        hoop_radius = 0.75  # 9 inch radius = 0.75 ft

        # Hoop in -X direction
        hoop_x = hoop_pos_x[0] + hoop_radius * np.cos(theta)
        hoop_y = hoop_pos_x[1] + hoop_radius * np.sin(theta)
        hoop_z = np.full_like(hoop_x, hoop_pos_x[2])
        ax1.plot(hoop_x, hoop_y, hoop_z, 'orange', linewidth=3, label='Hoop (-X dir)')

        # Velocity arrow at release
        scale = 0.3  # Scale for visibility
        ax1.quiver(release_pos[0], release_pos[1], release_pos[2],
                   release_vel[0]*scale, release_vel[1]*scale, release_vel[2]*scale,
                   color='red', arrow_length_ratio=0.2, linewidth=2)
        ax1.quiver(release_pos[0], release_pos[1], release_pos[2],
                   required_vel[0]*scale, required_vel[1]*scale, required_vel[2]*scale,
                   color='green', arrow_length_ratio=0.2, linewidth=2)

        ax1.set_xlabel('X (feet)')
        ax1.set_ylabel('Y (feet)')
        ax1.set_zlabel('Z (feet)')
        ax1.set_title(f'Shot {shot_id} - 3D Trajectory\nRed arrow: actual velocity, Green arrow: required velocity')
        ax1.legend(loc='upper left')

        # Set equal aspect ratio
        max_range = 20
        mid_x = release_pos[0] - 7.5
        mid_y = release_pos[1]
        ax1.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax1.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax1.set_zlim(0, 15)

        # Top-down view (X-Y plane)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(hand_positions[:, 0], hand_positions[:, 1], 'b-', linewidth=2, label='Hand')
        ax2.plot(ball_traj_current[:, 0], ball_traj_current[:, 1], 'r-', linewidth=2, label='Ball (data)')
        ax2.plot(ball_traj_required[:, 0], ball_traj_required[:, 1], 'g--', linewidth=2, label='Ball (required)')
        ax2.scatter(release_pos[0], release_pos[1], c='g', s=100, zorder=5)
        ax2.add_patch(plt.Circle((hoop_pos_x[0], hoop_pos_x[1]), hoop_radius, fill=False, color='orange', linewidth=2))
        ax2.set_xlabel('X (feet)')
        ax2.set_ylabel('Y (feet)')
        ax2.set_title('Top-Down View (X-Y)')
        ax2.legend()
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        # Side view (X-Z plane)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(hand_positions[:, 0], hand_positions[:, 2], 'b-', linewidth=2, label='Hand')
        ax3.plot(ball_traj_current[:, 0], ball_traj_current[:, 2], 'r-', linewidth=2, label='Ball (data)')
        ax3.plot(ball_traj_required[:, 0], ball_traj_required[:, 2], 'g--', linewidth=2, label='Ball (required)')
        ax3.scatter(release_pos[0], release_pos[2], c='g', s=100, zorder=5)
        ax3.scatter(hoop_pos_x[0], hoop_pos_x[2], c='orange', s=200, marker='o', zorder=5, label='Hoop')

        # Draw backboard
        ax3.plot([hoop_pos_x[0]-0.5, hoop_pos_x[0]-0.5], [hoop_pos_x[2]-2, hoop_pos_x[2]+2], 'k-', linewidth=3)

        # Velocity arrows
        arr_scale = 0.1
        ax3.annotate('', xy=(release_pos[0]+release_vel[0]*arr_scale, release_pos[2]+release_vel[2]*arr_scale),
                     xytext=(release_pos[0], release_pos[2]),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax3.annotate('', xy=(release_pos[0]+required_vel[0]*arr_scale, release_pos[2]+required_vel[2]*arr_scale),
                     xytext=(release_pos[0], release_pos[2]),
                     arrowprops=dict(arrowstyle='->', color='green', lw=2))

        ax3.set_xlabel('X (feet)')
        ax3.set_ylabel('Z (feet)')
        ax3.set_title('Side View (X-Z) - Shows trajectory arc')
        ax3.legend()
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 15)

        # Velocity over time
        ax4 = fig.add_subplot(2, 2, 4)
        time = (frames - frames[0]) / FPS
        ax4.plot(time, hand_vel[:, 0], 'r-', label='Vx (horizontal)')
        ax4.plot(time, hand_vel[:, 1], 'g-', label='Vy (horizontal)')
        ax4.plot(time, hand_vel[:, 2], 'b-', label='Vz (vertical)')
        ax4.axhline(y=required_vx, color='r', linestyle='--', alpha=0.5, label=f'Required Vx: {required_vx:.1f}')
        ax4.axhline(y=required_vz, color='b', linestyle='--', alpha=0.5, label=f'Required Vz: {required_vz:.1f}')
        ax4.axvline(x=(peak_frame - frames[0])/FPS, color='k', linestyle=':', label='Release')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Velocity (ft/s)')
        ax4.set_title('Hand Velocity Components Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        output_path = PROJECT_DIR / "output" / "trajectory_visualization.png"
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved to: {output_path}")

        plt.show()

        break

    print("\n" + "=" * 80)
    print("KEY FINDING:")
    print("=" * 80)
    print(f"\nActual horizontal velocity from data: {np.sqrt(release_vel[0]**2 + release_vel[1]**2):.1f} ft/s")
    print(f"Required horizontal velocity: {abs(required_vx):.1f} ft/s")
    print(f"MISSING: {abs(required_vx) - np.sqrt(release_vel[0]**2 + release_vel[1]**2):.1f} ft/s")


if __name__ == "__main__":
    main()

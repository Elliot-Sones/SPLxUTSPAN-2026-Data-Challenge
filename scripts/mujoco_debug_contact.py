"""
Debug MuJoCo contact setup - verify ball touches hand.
"""

import numpy as np
import mujoco

def create_simple_model():
    """
    Simplest possible model: ball on platform.
    """
    xml = """
    <mujoco model="debug_contact">
        <option gravity="0 0 -9.81" timestep="0.001">
            <flag contact="enable"/>
        </option>

        <worldbody>
            <geom type="plane" size="10 10 0.1"/>

            <!-- Fixed platform at z=1.5 -->
            <body name="platform" pos="0 0 1.5">
                <geom name="platform_geom" type="box" size="0.15 0.15 0.02" rgba="0.8 0.8 0.8 1"/>
            </body>

            <!-- Ball starting just above platform -->
            <!-- Platform top is at z = 1.5 + 0.02 = 1.52 -->
            <!-- Ball radius = 0.12, so center at z = 1.52 + 0.12 = 1.64 -->
            <body name="ball" pos="0 0 1.64">
                <freejoint/>
                <geom name="ball_geom" type="sphere" size="0.12" mass="0.625" rgba="1 0.5 0 1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def check_contact(model, data):
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        if 'ball' in (g1 or '') or 'ball' in (g2 or ''):
            return True, g1, g2
    return False, None, None


def main():
    print("=" * 60)
    print("MUJOCO CONTACT DEBUG")
    print("=" * 60)

    model = create_simple_model()
    data = mujoco.MjData(model)

    print(f"\nModel info:")
    print(f"  nq: {model.nq}")
    print(f"  nv: {model.nv}")
    print(f"  ngeom: {model.ngeom}")
    print(f"  nbody: {model.nbody}")

    # Print geometry info
    print(f"\nGeometry:")
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        gtype = model.geom_type[i]
        size = model.geom_size[i]
        pos = model.geom_pos[i]
        print(f"  {i}: {name}, type={gtype}, size={size}, pos={pos}")

    # Reset and check initial state
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print(f"\nInitial state:")
    print(f"  Ball qpos: {data.qpos}")
    print(f"  Ball xpos: {data.xpos}")  # Cartesian positions of bodies

    # Check contact
    contact, g1, g2 = check_contact(model, data)
    print(f"  Contact: {contact} ({g1} - {g2})")
    print(f"  ncon: {data.ncon}")

    # Step a few times
    print(f"\nStepping simulation...")
    for step in range(100):
        mujoco.mj_step(model, data)
        if step % 20 == 0:
            contact, g1, g2 = check_contact(model, data)
            ball_z = data.qpos[2]
            ball_vz = data.qvel[2]
            print(f"  Step {step}: ball_z={ball_z:.4f}, vz={ball_vz:.4f}, contact={contact}")

    print(f"\nFinal state:")
    print(f"  Ball pos: {data.qpos[0:3]}")
    print(f"  Ball vel: {data.qvel[0:3]}")
    contact, g1, g2 = check_contact(model, data)
    print(f"  Contact: {contact}")

    # Now test with a moving platform
    print("\n" + "=" * 60)
    print("TEST WITH MOVING PLATFORM")
    print("=" * 60)

    xml_moving = """
    <mujoco model="moving_platform">
        <option gravity="0 0 -9.81" timestep="0.001"/>

        <worldbody>
            <geom type="plane" size="10 10 0.1"/>

            <!-- Movable platform -->
            <body name="platform" pos="0 0 1.5">
                <joint name="pz" type="slide" axis="0 0 1" damping="100"/>
                <geom name="platform_geom" type="box" size="0.15 0.15 0.02" mass="5"/>
            </body>

            <!-- Ball -->
            <body name="ball" pos="0 0 1.64">
                <freejoint/>
                <geom name="ball_geom" type="sphere" size="0.12" mass="0.625"/>
            </body>
        </worldbody>

        <actuator>
            <position name="pz_ctrl" joint="pz" kp="5000" kv="500"/>
        </actuator>
    </mujoco>
    """

    model2 = mujoco.MjModel.from_xml_string(xml_moving)
    data2 = mujoco.MjData(model2)

    mujoco.mj_resetData(model2, data2)

    # Ball freejoint: qpos[0:7] (pos + quat)
    # Platform joint pz: qpos[7]
    print(f"\nModel2 nq: {model2.nq}")

    # Set initial platform position
    data2.qpos[7] = 0  # Platform at z=1.5 (its default)

    mujoco.mj_forward(model2, data2)

    contact, g1, g2 = check_contact(model2, data2)
    print(f"Initial contact: {contact}")
    print(f"Ball pos: {data2.qpos[0:3]}")
    print(f"Platform qpos: {data2.qpos[7]}")

    # Let settle
    for _ in range(200):
        mujoco.mj_step(model2, data2)

    contact, g1, g2 = check_contact(model2, data2)
    print(f"After settle - contact: {contact}")
    print(f"Ball pos: {data2.qpos[0:3]}")

    # Now move platform up quickly
    print("\nMoving platform up at 3 m/s...")
    target_vel = 3.0  # m/s
    target_pos = data2.qpos[7]

    for step in range(100):  # 100ms
        target_pos += target_vel * 0.001  # 1ms timestep
        data2.ctrl[0] = target_pos

        mujoco.mj_step(model2, data2)

        if step % 20 == 0:
            ball_vel = data2.qvel[0:3]
            ball_speed = np.linalg.norm(ball_vel)
            contact, _, _ = check_contact(model2, data2)
            print(f"  Step {step}: platform_z={data2.qpos[7]:.3f}, ball_z={data2.qpos[2]:.3f}, "
                  f"ball_speed={ball_speed:.2f} m/s, contact={contact}")

    print(f"\nFinal ball velocity: {data2.qvel[0:3]}")
    print(f"Final ball speed: {np.linalg.norm(data2.qvel[0:3]):.2f} m/s")


if __name__ == "__main__":
    main()

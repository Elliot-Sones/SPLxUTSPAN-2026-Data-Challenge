"""
Debug MuJoCo contact - find why ball isn't contacting hand.
"""

import numpy as np
import mujoco

def create_model():
    xml = """
    <mujoco model="ball_hand_debug">
        <option gravity="0 0 -9.81" timestep="0.0005"/>

        <worldbody>
            <geom name="ground" type="plane" size="20 20 0.1"/>

            <!-- Hand with slide joints -->
            <body name="hand" pos="0 0 1.5">
                <joint name="hx" type="slide" axis="1 0 0" damping="200"/>
                <joint name="hy" type="slide" axis="0 1 0" damping="200"/>
                <joint name="hz" type="slide" axis="0 0 1" damping="200"/>
                <geom name="palm" type="cylinder" size="0.10 0.02" mass="3.0"/>
            </body>

            <!-- Ball with freejoint -->
            <body name="ball" pos="0 0 2">
                <freejoint name="ball_joint"/>
                <geom name="ball_geom" type="sphere" size="0.12" mass="0.625"/>
            </body>
        </worldbody>

        <actuator>
            <position name="hx_ctrl" joint="hx" kp="20000" kv="2000"/>
            <position name="hy_ctrl" joint="hy" kp="20000" kv="2000"/>
            <position name="hz_ctrl" joint="hz" kp="20000" kv="2000"/>
        </actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def print_contact_info(model, data):
    print(f"  ncon = {data.ncon}")
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        print(f"    Contact {i}: {g1} <-> {g2}, pos={c.pos}, dist={c.dist}")


def check_ball_palm_contact(model, data):
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        if (g1 == 'ball_geom' and g2 == 'palm') or (g2 == 'ball_geom' and g1 == 'palm'):
            return True
    return False


def main():
    print("=" * 80)
    print("MUJOCO CONTACT DEBUG v2")
    print("=" * 80)

    model = create_model()
    data = mujoco.MjData(model)

    print(f"\nModel info:")
    print(f"  nq = {model.nq} (7 ball freejoint + 3 hand slides)")
    print(f"  nv = {model.nv}")
    print(f"  nu = {model.nu}")

    # Print geom info
    print(f"\nGeometries:")
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        gtype = model.geom_type[i]
        size = model.geom_size[i]
        print(f"  {i}: {name}, type={gtype}, size={size}")

    # qpos layout:
    # [0:3] ball position (freejoint)
    # [3:7] ball quaternion (freejoint)
    # [7] hx joint
    # [8] hy joint
    # [9] hz joint

    print("\n" + "=" * 80)
    print("TEST 1: Ball at exact contact point (z=1.64)")
    print("=" * 80)

    mujoco.mj_resetData(model, data)
    # Palm top is at z = 1.5 + 0.02 = 1.52
    # Ball radius = 0.12
    # Ball center for exact contact = 1.52 + 0.12 = 1.64
    data.qpos[0:3] = [0, 0, 1.64]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:10] = [0, 0, 0]  # Hand at default [0,0,1.5]
    mujoco.mj_forward(model, data)

    print(f"\nInitial state:")
    print(f"  Ball qpos[0:3] = {data.qpos[0:3]}")
    print(f"  Hand qpos[7:10] = {data.qpos[7:10]}")
    print(f"  Ball xpos (computed) = {data.xpos[2]}")  # body 2 is ball
    print(f"  Hand xpos (computed) = {data.xpos[1]}")  # body 1 is hand
    print_contact_info(model, data)
    print(f"  Ball-palm contact: {check_ball_palm_contact(model, data)}")

    print("\n" + "=" * 80)
    print("TEST 2: Ball with 1cm penetration (z=1.63)")
    print("=" * 80)

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0, 0, 1.63]  # 1cm penetration
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:10] = [0, 0, 0]
    mujoco.mj_forward(model, data)

    print(f"\nInitial state:")
    print(f"  Ball qpos[0:3] = {data.qpos[0:3]}")
    print_contact_info(model, data)
    print(f"  Ball-palm contact: {check_ball_palm_contact(model, data)}")

    # Step to let physics settle
    print("\nAfter 100 steps:")
    for _ in range(100):
        mujoco.mj_step(model, data)
    print(f"  Ball qpos[0:3] = {data.qpos[0:3]}")
    print_contact_info(model, data)
    print(f"  Ball-palm contact: {check_ball_palm_contact(model, data)}")

    print("\n" + "=" * 80)
    print("TEST 3: Ball with 2cm penetration (z=1.62)")
    print("=" * 80)

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0, 0, 1.62]  # 2cm penetration
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:10] = [0, 0, 0]
    mujoco.mj_forward(model, data)

    print(f"\nInitial state:")
    print(f"  Ball qpos[0:3] = {data.qpos[0:3]}")
    print_contact_info(model, data)
    print(f"  Ball-palm contact: {check_ball_palm_contact(model, data)}")

    print("\n" + "=" * 80)
    print("TEST 4: Hand moved to match real shot position")
    print("=" * 80)

    mujoco.mj_resetData(model, data)

    # Real shot hand position: [5.75, -7.68, 1.17] meters
    hand_world = np.array([5.75, -7.68, 1.17])
    hand_offset = hand_world - np.array([0, 0, 1.5])  # Offset from body default

    data.qpos[7] = hand_offset[0]  # hx
    data.qpos[8] = hand_offset[1]  # hy
    data.qpos[9] = hand_offset[2]  # hz

    # Compute where ball should be
    # Hand world z = 1.5 + hand_offset[2] = 1.17
    # Palm top = hand world z + 0.02 = 1.19
    # Ball center for contact = palm top + 0.12 = 1.31

    palm_top_z = 1.5 + hand_offset[2] + 0.02
    ball_z = palm_top_z + 0.12 - 0.01  # With 1cm penetration

    data.qpos[0] = hand_world[0]  # Ball x = hand x
    data.qpos[1] = hand_world[1]  # Ball y = hand y
    data.qpos[2] = ball_z
    data.qpos[3:7] = [1, 0, 0, 0]

    mujoco.mj_forward(model, data)

    print(f"\nConfiguration:")
    print(f"  Hand offset = {hand_offset}")
    print(f"  Hand world pos = {hand_world}")
    print(f"  Palm top z = {palm_top_z}")
    print(f"  Ball z = {ball_z}")
    print(f"\nState:")
    print(f"  Ball qpos[0:3] = {data.qpos[0:3]}")
    print(f"  Hand qpos[7:10] = {data.qpos[7:10]}")
    print(f"  Ball xpos = {data.xpos[2]}")
    print(f"  Hand xpos = {data.xpos[1]}")
    print_contact_info(model, data)
    print(f"  Ball-palm contact: {check_ball_palm_contact(model, data)}")

    # Step and check
    print("\nAfter 100 steps:")
    for _ in range(100):
        mujoco.mj_step(model, data)
    print(f"  Ball qpos[0:3] = {data.qpos[0:3]}")
    print(f"  Ball vel = {data.qvel[0:3]}")
    print_contact_info(model, data)
    print(f"  Ball-palm contact: {check_ball_palm_contact(model, data)}")

    print("\n" + "=" * 80)
    print("TEST 5: Move hand upward to push ball")
    print("=" * 80)

    # Continue from test 4, move hand up
    print("Moving hand up at 3 m/s for 0.1 seconds...")

    target_z = hand_offset[2]
    for step in range(200):  # 0.1 seconds at 0.0005s timestep
        target_z += 3.0 * 0.0005  # 3 m/s
        data.ctrl[0] = hand_offset[0]
        data.ctrl[1] = hand_offset[1]
        data.ctrl[2] = target_z

        mujoco.mj_step(model, data)

        if step % 40 == 0:
            ball_vel = data.qvel[0:3]
            ball_speed = np.linalg.norm(ball_vel)
            contact = check_ball_palm_contact(model, data)
            print(f"  Step {step}: ball_z={data.qpos[2]:.3f}, ball_speed={ball_speed:.2f} m/s, contact={contact}")

    print(f"\nFinal ball velocity: {data.qvel[0:3]}")
    print(f"Final ball speed: {np.linalg.norm(data.qvel[0:3]):.2f} m/s ({np.linalg.norm(data.qvel[0:3])/0.3048:.1f} ft/s)")


if __name__ == "__main__":
    main()

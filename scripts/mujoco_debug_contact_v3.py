"""
Debug MuJoCo contact - verify body positions and indexing.
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


def main():
    print("=" * 80)
    print("MUJOCO CONTACT DEBUG v3 - Body indexing check")
    print("=" * 80)

    model = create_model()
    data = mujoco.MjData(model)

    print(f"\nModel structure:")
    print(f"  nbody = {model.nbody}")
    print(f"  ngeom = {model.ngeom}")
    print(f"  nq = {model.nq}")

    # Print all bodies
    print(f"\nBodies:")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        pos = model.body_pos[i]
        print(f"  Body {i}: {name}, default pos = {pos}")

    # Print all geoms
    print(f"\nGeoms:")
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        body_id = model.geom_bodyid[i]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        pos = model.geom_pos[i]
        print(f"  Geom {i}: {name}, attached to body {body_id} ({body_name}), local pos = {pos}")

    # Get body IDs by name
    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    print(f"\nBody IDs by name:")
    print(f"  hand_body_id = {hand_body_id}")
    print(f"  ball_body_id = {ball_body_id}")

    # Reset and set positions
    print("\n" + "=" * 80)
    print("TEST: Set ball at z=1.64, verify xpos")
    print("=" * 80)

    mujoco.mj_resetData(model, data)

    print(f"\nAfter resetData, before setting qpos:")
    print(f"  qpos = {data.qpos}")
    print(f"  xpos shape = {data.xpos.shape}")
    print(f"  Full xpos:\n{data.xpos}")

    # Set ball position
    data.qpos[0:3] = [0, 0, 1.64]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:10] = [0, 0, 0]

    print(f"\nAfter setting qpos, before mj_forward:")
    print(f"  qpos = {data.qpos}")
    print(f"  Full xpos:\n{data.xpos}")

    # Forward kinematics
    mujoco.mj_forward(model, data)

    print(f"\nAfter mj_forward:")
    print(f"  qpos = {data.qpos}")
    print(f"  Full xpos:\n{data.xpos}")

    # Check specific body positions
    print(f"\nBody positions from xpos:")
    print(f"  World (body 0): {data.xpos[0]}")
    print(f"  Hand (body {hand_body_id}): {data.xpos[hand_body_id]}")
    print(f"  Ball (body {ball_body_id}): {data.xpos[ball_body_id]}")

    # Check geom positions
    palm_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "palm")
    ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")

    print(f"\nGeom IDs:")
    print(f"  palm_geom_id = {palm_geom_id}")
    print(f"  ball_geom_id = {ball_geom_id}")

    print(f"\nGeom positions from geom_xpos:")
    print(f"  geom_xpos shape = {data.geom_xpos.shape}")
    print(f"  palm geom pos: {data.geom_xpos[palm_geom_id]}")
    print(f"  ball geom pos: {data.geom_xpos[ball_geom_id]}")

    # Check contacts
    print(f"\nContacts:")
    print(f"  ncon = {data.ncon}")
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        print(f"    Contact {i}: {g1} (geom {c.geom1}) <-> {g2} (geom {c.geom2})")
        print(f"      pos = {c.pos}, dist = {c.dist}")

    # Compute distance between palm and ball manually
    palm_pos = data.geom_xpos[palm_geom_id]
    ball_pos = data.geom_xpos[ball_geom_id]
    distance = np.linalg.norm(ball_pos - palm_pos)
    z_diff = ball_pos[2] - palm_pos[2]

    print(f"\nManual distance calculation:")
    print(f"  Palm geom center: {palm_pos}")
    print(f"  Ball geom center: {ball_pos}")
    print(f"  3D distance: {distance:.4f} m")
    print(f"  Z difference: {z_diff:.4f} m")
    print(f"  Palm half-height: 0.02 m")
    print(f"  Ball radius: 0.12 m")
    print(f"  Expected z_diff for contact: 0.02 + 0.12 = 0.14 m")
    print(f"  Actual z_diff: {z_diff:.4f} m")
    print(f"  Should contact: {z_diff <= 0.14}")


if __name__ == "__main__":
    main()

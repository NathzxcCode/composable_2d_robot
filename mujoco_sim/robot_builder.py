from typing import List, Tuple
import math
from limb_spec import LimbSpec

DEFAULT_KP = 100.0


def _build_limb_xml(
    i: int, limbs: List[LimbSpec],
    parent_length: float = 0,
    indent: int = 2
) -> str:
    if i >= len(limbs):
        return ""

    ind = "  " * indent
    ind2 = "  " * (indent + 1)
    limb = limbs[i]
    pos = f"{parent_length} 0 0" if i > 0 else "0 0 0"
    rgba = " ".join(str(c) for c in limb.color)

    # Sensor position: along the cylinder axis + radial offset onto surface
    sensor_x = limb.length * limb.sensor_offset_along
    sensor_y = limb.radius * math.cos(limb.sensor_mount_angle)
    sensor_z = limb.radius * math.sin(limb.sensor_mount_angle)

    lines = [
        f'{ind}<body name="limb_{i}" pos="{pos}">',
        f'{ind2}<joint name="joint_{i}" type="hinge" axis="0 1 0" damping="{limb.joint_damping}"/>',
        f'{ind2}<geom name="joint_{i}_sphere" type="sphere" size="{limb.radius * 1.5}" rgba="0.6 0.6 0.6 1"/>',
        f'{ind2}<geom name="limb_{i}_geom" type="capsule" '
        f'fromto="0 0 0 {limb.length} 0 0" size="{limb.radius}" '
        f'rgba="{rgba}" density="{limb.density}"/>',
        f'{ind2}<site name="tip_{i}" pos="{limb.length} 0 0" size="0.006" '
        f'rgba="1 0 0 1" type="sphere"/>',
        f'{ind2}<site name="sensor_{i}" pos="{sensor_x} {sensor_y} {sensor_z}" size="0.006" '
        f'rgba="0.0 1.0 1.0 1" type="sphere"/>',
    ]

    child = _build_limb_xml(i + 1, limbs, limb.length, indent + 1)
    if child:
        lines.append(child)
    lines.append(f"{ind}</body>")

    return "\n".join(lines)


def build_robot_xml(
    limbs: List[LimbSpec],
    base_pos: Tuple[float, float, float] = (0, 0.5, 0),
    kp: float = DEFAULT_KP,
) -> str:
    actuators = "\n".join(
        f'    <position name="act_joint_{i}" joint="joint_{i}" kp="{kp}"/>'
        for i in range(len(limbs))
    )
    sensors = "\n".join(
        f'    <framepos name="sensor_pos_{i}" objtype="site" objname="sensor_{i}"/>'
        for i in range(len(limbs))
    )
    limbs_xml = _build_limb_xml(0, limbs)

    return f'''<?xml version="1.0"?>
<mujoco model="composable_robot">
  <compiler angle="radian" coordinate="local"/>
  <option gravity="0 -9.81 0" timestep="0.002"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker"
             rgb1="0.1 0.1 0.1" rgb2="0.15 0.15 0.2"
             width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.1"/>
  </asset>

  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1" material="grid"/>
    <light name="light" pos="0 3 4" dir="0 -1 -1"/>

    <body name="base" pos="{base_pos[0]} {base_pos[1]} {base_pos[2]}">
      <geom name="base_geom" type="cylinder" size="0.04 0.015"
            rgba="0.4 0.4 0.4 1"/>
{limbs_xml}
    </body>
  </worldbody>

  <actuator>
{actuators}
  </actuator>

  <sensor>
{sensors}
  </sensor>
</mujoco>'''
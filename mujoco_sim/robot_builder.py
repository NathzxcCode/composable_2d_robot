from typing import List, Tuple
import math
from limb_spec import LimbSpec

DEFAULT_KP       = 30.0
# Peak torque limit per joint (N·m).  Caps maximum joint speed to roughly
# F_MAX / joint_damping ≈ 2.0 / 0.4 = 5 rad/s, matching a small servo arm
# (e.g. Dynamixel XM430 stall torque: 4.1 N·m, max speed: ~6 rad/s).
DEFAULT_F_MAX    = 1.0

def _build_limb_xml(
    i: int, 
    limbs: List[LimbSpec], 
    indent: int = 2
) -> str:
    # Base case: if we have processed all limbs, stop recursion
    if i >= len(limbs):
        return ""

    ind = "  " * indent
    ind2 = "  " * (indent + 1)
    limb = limbs[i]
    
    # Format vectors into MuJoCo XML string space-separated formats
    pos_str = " ".join(str(v) for v in limb.attach_pos)
    euler_str = " ".join(str(e) for e in limb.attach_euler)
    axis_str = " ".join(str(a) for a in limb.joint_axis)
    sensor_pos_str = " ".join(str(s) for s in limb.sensor_pos)
    sensor_euler_str = " ".join(str(se) for se in limb.sensor_euler)
    rgba = " ".join(str(c) for c in limb.color)

    lines = [
        # The body is positioned and oriented relative to its direct parent frame
        f'{ind}<body name="limb_{i}" pos="{pos_str}" euler="{euler_str}">',
        
        # The joint uses a dynamic axis allowing for 3D routing later
        f'{ind2}<joint name="joint_{i}" type="hinge" axis="{axis_str}" damping="{limb.joint_damping}"/>',
        
        # Visual/Physical geometry for the link anchor point and body bone
        f'{ind2}<geom name="joint_{i}_sphere" type="sphere" size="{limb.radius * 1.5}" rgba="0.6 0.6 0.6 1"/>',
        f'{ind2}<geom name="limb_{i}_geom" type="capsule" '
        f'fromto="0 0 0 {limb.length} 0 0" size="{limb.radius}" '
        f'rgba="{rgba}" density="{limb.density}"/>',
        
        # Static target site representing the absolute tip of the limb segment
        f'{ind2}<site name="tip_{i}" pos="{limb.length} 0 0" size="0.006" rgba="1 0 0 1" type="sphere"/>',
        
        # Sensor tracking target, explicitly placed based on known calibration coordinates
        f'{ind2}<site name="sensor_{i}" pos="{sensor_pos_str}" euler="{sensor_euler_str}" size="0.006" '
        f'rgba="0.0 1.0 1.0 1" type="sphere"/>',
    ]

    # Recurse down to the next child link nested inside this body tag
    child = _build_limb_xml(i + 1, limbs, indent + 1)
    if child:
        lines.append(child)
        
    # Close the body tag cleanly preserving the XML nesting tree hierarchy
    lines.append(f"{ind}</body>")

    return "\n".join(lines)

def build_robot_xml(
    limbs: List[LimbSpec],
    base_pos: Tuple[float, float, float] = (0, 0.5, 0),
    kp: float = DEFAULT_KP,
    f_max: float = DEFAULT_F_MAX,
) -> str:
    # forcelimited + forcerange cap the maximum joint torque, which in turn
    # limits peak joint speed to roughly f_max / joint_damping.
    actuators = "\n".join(
        f'    <position name="act_joint_{i}" joint="joint_{i}" kp="{kp}" '
        f'forcelimited="true" forcerange="-{f_max} {f_max}"/>'
        for i in range(len(limbs))
    )
    
    # Request standard absolute position telemetry for your python loop to process
    sensors_list = []
    for i in range(len(limbs)):
        sensors_list.append(f'    <framepos name="sensor_pos_{i}" objtype="site" objname="sensor_{i}"/>')
        # sensors_list.append(f'    <framequat name="sensor_quat_{i}" objtype="site" objname="sensor_{i}"/>')
    sensors = "\n".join(sensors_list)
    
    limbs_xml = _build_limb_xml(0, limbs)

    return f'''<?xml version="1.0"?>
<mujoco model="composable_robot">
  <compiler angle="radian" coordinate="local"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>

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
from dataclasses import dataclass, field
from typing import List

@dataclass
class LimbSpec:
    # --- Physical Dimensions ---
    # Sized to match a small desktop manipulator (~150 mm links, ~160g total).
    # Density of 500 kg/m³ represents lightweight hollow aluminium construction.
    length: float = 0.15         # meters
    radius: float = 0.015        # visual/collision capsule radius
    color: List[float] = field(default_factory=lambda: [0.3, 0.6, 0.9, 1.0])
    density: float = 500.0       # kg/m³ (lightweight hollow link)

    # --- Kinematic Attachment (Where does this limb attach to its parent?) ---
    # Default: attached at the nominal tip of the parent link along local X-axis.
    attach_pos: List[float] = field(default_factory=lambda: [0.15, 0.0, 0.0])
    # euler angles [roll, pitch, yaw] in radians applied to THIS body's frame
    # relative to the parent.  Use [0, 0, pi/2] to stand a limb upright (local X -> world Y).
    attach_euler: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # --- Joint Properties ---
    # Hinge axis in the LOCAL body frame (after attach_euler is applied).
    # Default [0,1,0] = local Y gives pitch motion when capsule runs along local X.
    joint_axis: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    # Damping ≈ 0.4 N·m·s/rad gives a slightly underdamped response at kp=30,
    # settling in ~0.3 s – realistic for a small servo-driven arm.
    joint_damping: float = 0.4

    # --- Joint motion limits ---
    # joint_centre: the resting / neutral angle the controller drives toward (radians).
    # joint_range:  half-width of the random motion range around that centre (radians).
    #               The controller picks targets uniformly in
    #               [joint_centre - joint_range, joint_centre + joint_range].
    joint_centre: float = 0.0
    joint_range: float = 0.785   # pi/4 ≈ 45°

    # --- Sensor Calibration (known transform from THIS limb's joint origin) ---
    # Sensor placed at 80 % of link length with a small lateral offset.
    sensor_pos: List[float] = field(default_factory=lambda: [0.12, 0.01, 0.0]) # [X, Y, Z]
    sensor_euler: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0]) # [Roll, Pitch, Yaw]

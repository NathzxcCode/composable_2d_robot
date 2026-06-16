from dataclasses import dataclass, field


@dataclass
class LimbSpec:
    length: float = 0.4          # meters
    radius: float = 0.02         # visual thickness
    color: list = field(default_factory=lambda: [0.3, 0.6, 0.9, 1.0])
    density: float = 1000.0      # kg/m^3 for auto mass
    joint_damping: float = 0.5   # joint damping coefficient
    sensor_offset_along: float = 0.8   # fraction of length (0-1), where along the limb
    sensor_mount_angle: float = 0.0    # radians around cylinder circumference
from dataclasses import dataclass, field
from typing import List

@dataclass
class LimbSpec:
    # --- Physical Dimensions ---
    length: float = 0.4          # meters
    radius: float = 0.02         # visual thickness
    color: List[float] = field(default_factory=lambda: [0.3, 0.6, 0.9, 1.0])
    density: float = 1000.0      # kg/m^3 for auto mass
    
    # --- Kinematic Attachment (Where does this limb attach to its parent?) ---
    # Default is attached at the tip of the parent link along the local X-axis
    attach_pos: List[float] = field(default_factory=lambda: [0.4, 0.0, 0.0])
    attach_euler: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0]) # [Roll, Pitch, Yaw]
    
    # --- Joint Properties ---
    joint_axis: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0]) # Rotates around Y for 2D
    joint_damping: float = 0.5   
    
    # --- Sensor Calibration (The explicit known transform from THIS limb's joint origin) ---
    # Instead of relative fractions, we define the exact translation vector to the sensor
    sensor_pos: List[float] = field(default_factory=lambda: [0.32, 0.02, 0.0]) # [X, Y, Z]
    sensor_euler: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0]) # [Roll, Pitch, Yaw] Orientation of sensor housing
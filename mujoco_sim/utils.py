import numpy as np
import mujoco

def render_covariance_ellipses(viewer, calibrations, global_positions, joint_rotations, sigma: float = 2.0, visual_scale: float = 0.1):
    """
    Renders 2D covariance ellipses overlayed on the MuJoCo X-Z simulation plane.

    The calibration mean and covariance are expressed in the PARENT limb's local frame
    (GTSAM 2D: x = along limb, y = perpendicular in X-Z plane).  Before rendering we
    rotate them into the MuJoCo world frame using the parent limb's orientation matrix.

    Parameters:
        viewer:           Passive viewer handle from launch_passive().
        calibrations:     List of dicts from extract_calibrations().
                          calibrations[k] is the CJ offset for the joint between
                          limb k and limb k+1 (0-indexed, so calibrations[0] -> limb 0->1).
        global_positions: List of length (num_limbs-1). global_positions[k] is the
                          expected (nominal) world position of the k-th joint connection
                          (between limb k and limb k+1).
        joint_rotations:  List of length num_limbs.  joint_rotations[k] is the 3x3
                          world rotation matrix of limb k's body frame.
        sigma:            Confidence-interval scale factor (e.g. 2 = ~95%).
        visual_scale:     Additional multiplier for visual size.
    """
    viewer.user_scn.ngeom = 0
    geom_idx = 0

    for i, calib in enumerate(calibrations):
        mean_local = np.array(calib["mean"], dtype=float)   # [cx, cy] in parent local frame
        cov_local  = np.array(calib["cov_xy"], dtype=float) # 2x2 in parent local frame

        # --- Parent limb rotation matrix (3x3 world frame) ---
        # joint_rotations[i] is the world rotation of limb i, which is the parent
        # for the i-th calibration offset (CJ between limb i and limb i+1).
        R_world_3x3 = joint_rotations[i]   # shape (3,3)

        # Extract the 2x2 rotation acting on the X-Z plane.
        # In MuJoCo: GTSAM-x maps to world-X (col 0), GTSAM-y maps to world-Z (col 2).
        #   world_x_component_of_local_x = R_world_3x3[0, 0]
        #   world_z_component_of_local_x = R_world_3x3[2, 0]
        #   world_x_component_of_local_y = R_world_3x3[0, 2]  (local y = world Z direction)
        #   world_z_component_of_local_y = R_world_3x3[2, 2]
        # This gives a 2x2 matrix R_2d that maps [local_x, local_y] -> [world_X, world_Z]
        R_2d = np.array([
            [R_world_3x3[0, 0], R_world_3x3[0, 2]],
            [R_world_3x3[2, 0], R_world_3x3[2, 2]],
        ])

        # Rotate mean into world X-Z coordinates
        mean_world_xz = R_2d @ mean_local   # shape (2,): [delta_X_world, delta_Z_world]

        # Rotate covariance: Sigma_world = R @ Sigma_local @ R^T
        cov_world = R_2d @ cov_local @ R_2d.T

        # --- Eigendecomposition of the world-frame covariance ---
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_world)
        except np.linalg.LinAlgError:
            continue

        # Semi-axis lengths along the two principal directions
        size_x = sigma * np.sqrt(max(0.0, eigenvalues[0])) * visual_scale
        size_z = sigma * np.sqrt(max(0.0, eigenvalues[1])) * visual_scale
        size_y = 0.002  # paper-thin in the out-of-plane Y direction

        # Map 2D eigenvectors (in world X-Z) into a 3x3 rotation matrix.
        # eigenvectors[:,0] is the first principal axis in [world_X, world_Z]
        # eigenvectors[:,1] is the second principal axis in [world_X, world_Z]
        R_ellipse = np.eye(3)
        R_ellipse[0, 0] = eigenvectors[0, 0]   # X component of axis 0
        R_ellipse[2, 0] = eigenvectors[1, 0]   # Z component of axis 0
        R_ellipse[0, 2] = eigenvectors[0, 1]   # X component of axis 1
        R_ellipse[2, 2] = eigenvectors[1, 1]   # Z component of axis 1

        if np.linalg.det(R_ellipse) < 0:
            R_ellipse[:, 2] = -R_ellipse[:, 2]

        # --- World position: nominal connection point + rotated mean offset ---
        # global_positions[i] is the nominal joint connection position in world 3D.
        # mean_world_xz adds the calibration offset (X and Z components).
        nominal_pos = global_positions[i]  # shape (3,): [X, Y, Z] world
        pos = [
            nominal_pos[0] + mean_world_xz[0],   # world X
            nominal_pos[1],                        # world Y (robot's out-of-plane axis)
            nominal_pos[2] + mean_world_xz[1],    # world Z
        ]

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_idx],
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size=[size_x, size_y, size_z],
            pos=pos,
            mat=R_ellipse.flatten(),
            rgba=[0.2, 0.8, 0.2, 0.35],
        )
        geom_idx += 1

    viewer.user_scn.ngeom = geom_idx

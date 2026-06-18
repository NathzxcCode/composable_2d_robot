import numpy as np
import mujoco

def render_covariance_ellipses(viewer, calibrations, global_positions, joint_rotations, sigma: float = 2.0, visual_scale: float = 0.1):
    """
    Renders 2D covariance ellipses overlayed on the MuJoCo X-Z simulation plane.
    
    Parameters:
        viewer: The passive viewer handle returned from launch_passive().
        calibrations (list): The list of dictionaries from extract_calibrations().
        sigma (float): The confidence interval scale.
        visual_scale (float): Multiplier to shrink/grow the shapes visually.
    """
    # 1. Reset user geometry count for this frame
    viewer.user_scn.ngeom = 0
    geom_idx = 0
    
    for i, calib in enumerate(calibrations):
        mean = calib["mean"]
        cov = np.array(calib["cov_xy"])
        
        # 2. Compute Eigenvalues and Eigenvectors of the covariance matrix
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
            
        # 3. Calculate semi-axis lengths
        # Apply the square root to eigenvalues and multiply by our scale factors
        size_x = sigma * np.sqrt(max(0.0, eigenvalues[0])) * visual_scale
        size_z = sigma * np.sqrt(max(0.0, eigenvalues[1])) * visual_scale
        size_y = 0.002  # Make it paper-thin along the Y-axis since we are flat on the X-Z plane
        
        # 4. Map the 2D Eigenvectors into a 3x3 Rotation Matrix for the X-Z Plane
        # In a standard X-Y plane, the 2D vectors go into rows/columns 0 and 1.
        # Since we are shifting to the X-Z plane, we map them to rows/columns 0 and 2.
        R = np.eye(3)
        R[0, 0] = eigenvectors[0, 0]
        R[0, 2] = eigenvectors[0, 1]
        R[2, 0] = eigenvectors[1, 0]
        R[2, 2] = eigenvectors[1, 1]
        
        # Handle reflection edge-cases to guarantee a valid rotation matrix
        if np.linalg.det(R) < 0:
            R[:, 2] = -R[:, 2]
            
        # 5. Position the Ellipse on the X-Z Plane
        # mean[0] is X, mean[1] is your graph's 2D 'Y', which maps to MuJoCo's Z!
        # We offset Y slightly forward (e.g., 0.02) to keep it from clipping inside the robot geoms.
        global_position = global_positions[i]
        pos = [mean[0]+global_position[0], global_position[1], mean[1]+global_position[2]]

        # 6. Initialize the abstract geometry slot
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_idx],
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size=[size_x, size_y, size_z], # width (X), depth (Y-thin), height (Z)
            pos=pos,
            mat=R.flatten(),
            rgba=[0.2, 0.8, 0.2, 0.35]  # Transparent green overlay
        )
        geom_idx += 1
        
    # 7. Inform the viewer engine how many custom shapes need rendering
    viewer.user_scn.ngeom = geom_idx
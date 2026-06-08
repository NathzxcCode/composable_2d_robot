import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def visualize_ellipsoid_overlap(xA1, xA2, xB1, xB2, error, dist):
    """
    Plots the 2D projection of the prolate spheroids to visually inspect
    the overlap relative to the calculated Bhattacharyya distance metrics.
    """
    # 1. Reconstruct parameters for visualization (using X-Y projection)
    mu_A = 0.5 * (xA1[:2] + xA2[:2])
    mu_B = 0.5 * (xB1[:2] + xB2[:2])
    
    v_A = xA2[:2] - xA1[:2]
    v_B = xB2[:2] - xB1[:2]
    
    r = 0.3 # Matching the base radius from your script
    Sigma_A = 0.25 * np.outer(v_A, v_A) + (r**2) * np.eye(2)
    Sigma_B = 0.25 * np.outer(v_B, v_B) + (r**2) * np.eye(2)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 2. Dynamic background color based on the collision factor danger
    # Error close to 1 = Red (Collision), Error close to 0 = White/Green (Safe)
    bg_intensity = min(max(error, 0.0), 1.0)
    ax.set_facecolor((1.0, 1.0 - bg_intensity * 0.8, 1.0 - bg_intensity * 0.8, 0.15))

    def plot_ellipse(mean, cov, ax, color, label):
        # Extract eigenvalues and eigenvectors for geometry
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        
        # 2-sigma confidence scale factor for 2D distribution (~86.5% area boundary)
        scale_factor = 2 * np.sqrt(2.41) 
        width = scale_factor * np.sqrt(eigenvalues[0])
        height = scale_factor * np.sqrt(eigenvalues[1])
        
        # Calculate rotation angle in degrees
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # Create and add the filled patch
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                          edgecolor=color, fc=color, alpha=0.35, lw=2, label=label)
        ax.add_patch(ellipse)

    # 3. Render the geometries
    plot_ellipse(mu_A, Sigma_A, ax, color='royalblue', label='Limb A Envelope')
    plot_ellipse(mu_B, Sigma_B, ax, color='crimson', label='Limb B Envelope')
    
    # Plot the underlying bone skeletons (endpoints)
    ax.plot([xA1[0], xA2[0]], [xA1[1], xA2[1]], 'o-', color='navy', lw=3, label='Limb A Bone')
    ax.plot([xB1[0], xB2[0]], [xB1[1], xB2[1]], 'o-', color='darkred', lw=3, label='Limb B Bone')
    
    # Draw vector between centres of mass
    ax.annotate('', xy=mu_B, xytext=mu_A, arrowprops=dict(arrowstyle="<->", color='black', lw=1.5, linestyle=':'))
    
    # 4. Interface Formatting
    ax.set_title(f"Bhattacharyya $D_B$: {dist:.4f}  |  Factor Penalty $E$: {error:.4f}", 
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')
    
    # Uniform aspect ratio ensuring circles/ellipses aren't distorted
    ax.set_aspect('equal', adjustable='datalim')
    
    plt.show()

def compute_collision_factor(xA1, xA2, xB1, xB2, k=2.0, r=0.3):
    """
    Computes the exponential Bhattacharyya collision factor and its Jacobians 
    with respect to 4 endpoint nodes in 3D space.
    
    Parameters:
      xA1, xA2: 3D NumPy arrays representing endpoints of Limb A
      xB1, xB2: 3D NumPy arrays representing endpoints of Limb B
      k: Sharpness factor of the exponential repulsion field
      r: Base physical radius (uncertainty baseline) of the limbs
    """
    # ---------------------------------------------------------
    # 1. KINEMATIC MAP (Nodes -> Mean & Covariances)
    # ---------------------------------------------------------
    mu_A = 0.5 * (xA1 + xA2)
    mu_B = 0.5 * (xB1 + xB2)
    
    v_A = xA2 - xA1
    v_B = xB2 - xB1
    
    # Constructing covariances scaled with the limb direction + base radius
    Sigma_A = 0.25 * np.outer(v_A, v_A) + (r**2) * np.eye(3)
    Sigma_B = 0.25 * np.outer(v_B, v_B) + (r**2) * np.eye(3)
    
    Sigma = 0.5 * (Sigma_A + Sigma_B)
    Sigma_inv = np.linalg.inv(Sigma)
    
    # ---------------------------------------------------------
    # 2. BHATTACHARYYA DISTANCE & COST
    # ---------------------------------------------------------
    d_mu = mu_A - mu_B
    
    # Term 1: Mahalanobis Distance (Translational shift)
    mahalanobis = 0.125 * np.dot(d_mu, np.dot(Sigma_inv, d_mu))
    
    # Term 2: Determinant Component (Shape and Volume expansion)
    det_A = np.linalg.det(Sigma_A)
    det_B = np.linalg.det(Sigma_B)
    det_Sigma = np.linalg.det(Sigma)
    shape_term = 0.5 * np.log(det_Sigma / np.sqrt(det_A * det_B))
    
    D_B = mahalanobis + shape_term
    
    # Final Smooth Repulsion Cost (Error)
    E = np.exp(-k * D_B)
    
    # ---------------------------------------------------------
    # 3. ANALYTICAL JACOBIANS (Chain Rule Integration)
    # ---------------------------------------------------------
    # Layer 1: Scalar Exponential Gradient
    dE_dDB = -k * E
    
    # Layer 2: Mean Position Gradients
    dDB_dmuA = 0.25 * np.dot(Sigma_inv, d_mu)
    dDB_dmuB = -dDB_dmuA
    
    # Propagate Mean Position Shifts to Endpoints (Equally distributed)
    dE_dxA1_pos = dE_dDB * (0.5 * dDB_dmuA)
    dE_dxA2_pos = dE_dDB * (0.5 * dDB_dmuA)
    dE_dxB1_pos = dE_dDB * (0.5 * dDB_dmuB)
    dE_dxB2_pos = dE_dDB * (0.5 * dDB_dmuB)
    
    # Layer 3: Matrix Derivative w.r.t Average Covariance
    dDB_dSigma = -0.125 * np.outer(np.dot(Sigma_inv, d_mu), np.dot(Sigma_inv, d_mu)) + 0.5 * Sigma_inv
    dE_dSigma = dE_dDB * dDB_dSigma
    
    # Project Matrix Gradients back to the Vector Endpoints (dSigma/dx)
    dE_dxA2_cov = np.zeros(3)
    dE_dxB2_cov = np.zeros(3)
    for i in range(3):
        # Element-wise differential contraction for Limb A
        M_Ai = np.zeros((3,3))
        M_Ai[i, :] += v_A
        M_Ai[:, i] += v_A
        dSigma_dxA2_i = 0.5 * (0.25 * M_Ai)
        dE_dxA2_cov[i] = np.sum(dE_dSigma * dSigma_dxA2_i)
        
        # Element-wise differential contraction for Limb B
        M_Bi = np.zeros((3,3))
        M_Bi[i, :] += v_B
        M_Bi[:, i] += v_B
        dSigma_dxB2_i = 0.5 * (0.25 * M_Bi)
        dE_dxB2_cov[i] = np.sum(dE_dSigma * dSigma_dxB2_i)
        
    # Opposite sign applies to origin nodes because v = x2 - x1
    dE_dxA1_cov = -dE_dxA2_cov
    dE_dxB1_cov = -dE_dxB2_cov
    
    # Combined Gradients (Translation + Orientation deformation)
    J_xA1 = dE_dxA1_pos + dE_dxA1_cov
    J_xA2 = dE_dxA2_pos + dE_dxA2_cov
    J_xB1 = dE_dxB1_pos + dE_dxB1_cov
    J_xB2 = dE_dxB2_pos + dE_dxB2_cov
    
    return E, D_B, J_xA1, J_xA2, J_xB1, J_xB2

# ---------------------------------------------------------
# EXECUTION EXAMPLE: Two limbs colliding in close proximity
# ---------------------------------------------------------
if __name__ == "__main__":
    # Limb A (Base at origin, extending up the Y axis)
    node_xA1 = np.array([0.0, 0.0, 0.0])
    node_xA2 = np.array([0.0, 2.0, 0.0])
    
    # Limb B (Parallel to A, but breaking into its envelope on the X/Y axes)
    node_xB1 = np.array([0.8, 2.3, 0.0])  
    node_xB2 = np.array([2.8, 2.3, 0.0])
    
    # Compute Cost and Jacobians
    error, dist, J_A1, J_A2, J_B1, J_B2 = compute_collision_factor(
        node_xA1, node_xA2, node_xB1, node_xB2, k=1.0, r=0.3
    )
    
    print("--- COLLISION FACTOR OUTPUT ---")
    print(f"Bhattacharyya Distance (DB): {dist:.4f}")
    print(f"Factor Cost (Error Value):   {error:.4f}  (High = High Danger)")
    print("\n--- GRADIENS (DIRECTIONS TO UPDATE NODES) ---")
    print(f"Node A1 Gradient: {J_A1}")
    print(f"Node A2 Gradient: {J_A2}")
    print(f"Node B1 Gradient: {J_B1}")
    print(f"Node B2 Gradient: {J_B2}")

    visualize_ellipsoid_overlap(node_xA1, node_xA2, node_xB1, node_xB2, error, dist)
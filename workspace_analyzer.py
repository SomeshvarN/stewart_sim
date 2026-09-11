import numpy as np
import matplotlib.pyplot as plt
from platform_core import AdvancedStewartPlatform

def compute_workspace_3d(hexapod, resolution=25, max_kappa=15.0):
    """
    Evaluates achievable 3D translation volume filtered by leg constraints
    and Jacobian Condition Number limits (Singularity Avoidance).
    """
    x_range = np.linspace(-0.6, 0.6, resolution)
    y_range = np.linspace(-0.6, 0.6, resolution)
    z_range = np.linspace(-0.4, 0.4, resolution)
    
    valid_points = []
    kappa_values = []
    
    # Fixed operational tilt during sampling (5 deg roll/pitch)
    eval_orientation = [np.radians(5), np.radians(5), 0.0]
    
    for x in x_range:
        for y in y_range:
            for z in z_range:
                pose = [x, y, z, *eval_orientation]
                
                # Check IK constraints
                _, _, _, valid_legs, _ = hexapod.inverse_kinematics(pose)
                
                if np.all(valid_legs):
                    kappa = hexapod.compute_condition_number(pose)
                    
                    if kappa <= max_kappa:
                        valid_points.append([x, y, z])
                        kappa_values.append(kappa)
                        
    return np.array(valid_points), np.array(kappa_values)


def plot_workspace_heatmaps(hexapod):
    print("Computing 3D Workspace volume... (please wait a few seconds)")
    points, kappas = compute_workspace_3d(hexapod, resolution=25, max_kappa=15.0)
    
    if len(points) == 0:
        print("No valid workspace points found. Check leg length limits (l_min, l_max).")
        return

    fig = plt.figure(figsize=(14, 6))
    
    # 3D Scatter Cloud colored by Dexterity Metric
    ax_3d = fig.add_subplot(121, projection='3d')
    sc = ax_3d.scatter(
        points[:, 0], points[:, 1], points[:, 2], 
        c=kappas, cmap='viridis_r', s=12, alpha=0.7
    )
    cbar = fig.colorbar(sc, ax=ax_3d, shrink=0.6, pad=0.1)
    cbar.set_label(r'Condition Number $\kappa(J)$ (Lower = Higher Dexterity)')
    
    ax_3d.set_title("Achievable Dexterous Workspace Cloud")
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z offset (m)")
    
    # XY Projection Slice at Nominal Z
    ax_2d = fig.add_subplot(122)
    z_mask = np.abs(points[:, 2]) < 0.05
    
    if np.any(z_mask):
        sc2 = ax_2d.scatter(
            points[z_mask, 0], points[z_mask, 1], 
            c=kappas[z_mask], cmap='viridis_r', s=40, edgecolors='none'
        )
        fig.colorbar(sc2, ax=ax_2d, label=r'$\kappa(J)$')
    
    ax_2d.set_title("XY Workspace Cross-Section (Z ~ 0)")
    ax_2d.set_xlabel("X Translation (m)")
    ax_2d.set_ylabel("Y Translation (m)")
    ax_2d.grid(True)
    ax_2d.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    platform = AdvancedStewartPlatform()
    plot_workspace_heatmaps(platform)
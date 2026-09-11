import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from platform_core import AdvancedStewartPlatform

def run_simulation():
    hexapod = AdvancedStewartPlatform(l_min=1.2, l_max=1.8)
    
    fig = plt.figure(figsize=(16, 8))
    ax_3d = fig.add_subplot(131, projection='3d')
    ax_vel = fig.add_subplot(132)
    ax_cond = fig.add_subplot(133)
    
    # Storage for dynamic plotting
    time_history = []
    leg_vel_history = [[] for _ in range(6)]
    cond_num_history = []
    fk_error_history = []
    
    dt = 0.05

    def update(frame):
        ax_3d.clear()
        
        t = frame * dt
        time_history.append(t)
        
        # 1. Target Trajectory Definition (Task Space Pose: [x, y, z, roll, pitch, yaw])
        x_pos = 0.2 * np.sin(t)
        y_pos = 0.2 * np.cos(t)
        z_pos = 0.1 * np.sin(2 * t)
        
        roll  = np.radians(10 * np.sin(t))
        pitch = np.radians(10 * np.cos(t))
        yaw   = np.radians(5 * np.sin(1.5 * t))
        
        target_pose = np.array([x_pos, y_pos, z_pos, roll, pitch, yaw])
        
        # 2. Task space velocities (Analytical derivatives of trajectory)
        vx = 0.2 * np.cos(t)
        vy = -0.2 * np.sin(t)
        vz = 0.2 * np.cos(2 * t)
        wx = np.radians(10 * np.cos(t))
        wy = -np.radians(10 * np.sin(t))
        wz = np.radians(7.5 * np.cos(1.5 * t))
        
        platform_vel = [vx, vy, vz, wx, wy, wz]
        
        # 3. Inverse Kinematics
        lengths, leg_vectors, P_world, valid_legs, R = hexapod.inverse_kinematics(target_pose)
        
        # 4. Actuator Velocities using Jacobian
        leg_velocities, J = hexapod.compute_leg_velocities(target_pose, platform_vel)
        
        # 5. Singularity Analysis (Condition Number Calculation)
        cond_num = hexapod.compute_condition_number(target_pose)
        cond_num_history.append(cond_num)
        
        # 6. Newton-Raphson Forward Kinematics Verification
        fk_estimated_pose, fk_success = hexapod.forward_kinematics(lengths, initial_pose_guess=target_pose * 0.9)
        fk_error = np.linalg.norm(target_pose - fk_estimated_pose)
        fk_error_history.append(fk_error)
        
        for i in range(6):
            leg_vel_history[i].append(leg_velocities[i])
        
        # --- 3D RENDERING ---
        base_closed = np.vstack([hexapod.B, hexapod.B[0]])
        ax_3d.plot(base_closed[:, 0], base_closed[:, 1], base_closed[:, 2], 'k-', lw=2)
        
        all_valid = np.all(valid_legs)
        top_closed = np.vstack([P_world, P_world[0]])
        ax_3d.plot(top_closed[:, 0], top_closed[:, 1], top_closed[:, 2], 'g-' if all_valid else 'r-', lw=2)
        
        for i in range(6):
            color = 'b--' if valid_legs[i] else 'r-'
            ax_3d.plot(
                [hexapod.B[i, 0], P_world[i, 0]],
                [hexapod.B[i, 1], P_world[i, 1]],
                [hexapod.B[i, 2], P_world[i, 2]],
                color, lw=1.5
            )
            
        ax_3d.set_xlim([-1.5, 1.5])
        ax_3d.set_ylim([-1.5, 1.5])
        ax_3d.set_zlim([0, 2.5])
        ax_3d.set_title("Hexapod Pose 3D View")
        
        # --- VELOCITY PLOT ---
        ax_vel.clear()
        start_idx = max(0, len(time_history) - 50)
        for i in range(6):
            ax_vel.plot(
                time_history[start_idx:], 
                leg_vel_history[i][start_idx:], 
                label=f'Leg {i+1}'
            )
            
        ax_vel.set_title("Actuator Velocities (m/s)")
        ax_vel.set_xlabel("Time (s)")
        ax_vel.set_ylabel("Velocity (m/s)")
        ax_vel.grid(True)
        ax_vel.legend(loc='upper right', fontsize=8)

        # --- CONDITION NUMBER METRIC PLOT ---
        ax_cond.clear()
        ax_cond.plot(time_history[start_idx:], cond_num_history[start_idx:], 'm-', label=r'Condition Number $\kappa(J)$')
        ax_cond.set_title(r"Dexterity & Singularity Metric ($\kappa$)")
        ax_cond.set_xlabel("Time (s)")
        ax_cond.set_ylabel(r"$\kappa(J)$ (1.0 = Ideal)")
        ax_cond.grid(True)
        ax_cond.legend(loc='upper left', fontsize=8)

    ani = FuncAnimation(fig, update, frames=200, interval=50)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()
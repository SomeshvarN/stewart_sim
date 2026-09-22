import numpy as np
import matplotlib.pyplot as plt
from platform_core import AdvancedStewartPlatform

class TaskSpaceTrajectoryGenerator:
    """
    Smooth Task-Space Motion Planner for 6-DOF Stewart Platforms.
    Implements Jerk-Bounded S-Curves and Quintic (5th-order) Polynomial Interpolation.
    """
    def __init__(self, dt=0.01):
        self.dt = dt

    def generate_quintic_s_curve(self, start_pose, target_pose, duration):
        """
        Generates a quintic polynomial trajectory between start and target 6-DOF poses.
        Guarantees zero velocity and acceleration at start (t=0) and end (t=T).
        
        Parameters:
            start_pose:  (6,) initial [x, y, z, roll, pitch, yaw]
            target_pose: (6,) target [x, y, z, roll, pitch, yaw]
            duration:    float, total move time in seconds
            
        Returns:
            time:   (N,) time vector
            pose:   (N, 6) pose trajectory
            vel:    (N, 6) task-space linear/angular velocities
            accel:  (N, 6) task-space linear/angular accelerations
        """
        N = int(duration / self.dt) + 1
        time = np.linspace(0, duration, N)
        
        p0 = np.array(start_pose, dtype=float)
        pf = np.array(target_pose, dtype=float)
        
        pose = np.zeros((N, 6))
        vel = np.zeros((N, 6))
        accel = np.zeros((N, 6))
        
        for i, t in enumerate(time):
            # Normalized time ratio s in [0, 1]
            s = t / duration
            
            # Quintic Polynomial S-curve Blend: s(t) = 10*s^3 - 15*s^4 + 6*s^5
            s_poly = 10 * (s**3) - 15 * (s**4) + 6 * (s**5)
            s_dot = (30 * (s**2) - 60 * (s**3) + 30 * (s**4)) / duration
            s_ddot = (60 * s - 180 * (s**2) + 120 * (s**3)) / (duration**2)
            
            # Position, Velocity, and Acceleration
            pose[i] = p0 + (pf - p0) * s_poly
            vel[i] = (pf - p0) * s_dot
            accel[i] = (pf - p0) * s_ddot
            
        return time, pose, vel, accel

    def generate_multi_waypoint_path(self, waypoints, segment_durations):
        """
        Connects multiple 6-DOF waypoints using smooth quintic S-curves.
        
        Parameters:
            waypoints:         List of (6,) poses
            segment_durations: List of move times for each waypoint segment
        """
        all_time = []
        all_pose = []
        all_vel = []
        all_accel = []
        
        t_accum = 0.0
        for i in range(len(waypoints) - 1):
            p_start = waypoints[i]
            p_end = waypoints[i + 1]
            T = segment_durations[i]
            
            t_seg, pos_seg, vel_seg, acc_seg = self.generate_quintic_s_curve(p_start, p_end, T)
            
            # Remove duplicate boundary point between segments
            if i > 0:
                t_seg = t_seg[1:] + t_accum
                pos_seg = pos_seg[1:]
                vel_seg = vel_seg[1:]
                acc_seg = acc_seg[1:]
            else:
                t_seg = t_seg + t_accum
                
            t_accum = t_seg[-1]
            
            all_time.append(t_seg)
            all_pose.append(pos_seg)
            all_vel.append(vel_seg)
            all_accel.append(acc_seg)
            
        return (
            np.concatenate(all_time),
            np.vstack(all_pose),
            np.vstack(all_vel),
            np.vstack(all_accel)
        )


def run_trajectory_demo():
    planner = TaskSpaceTrajectoryGenerator(dt=0.02)
    hexapod = AdvancedStewartPlatform()
    
    # Define a set of 6-DOF Waypoints: [x, y, z, roll, pitch, yaw]
    waypoints = [
        [0.0,  0.0,  0.0, 0.0, 0.0, 0.0],                          # Home Position
        [0.15, 0.10, 0.05, np.radians(10), 0.0, 0.0],              # Move 1: Right-Up-Roll
        [-0.10, 0.15, -0.05, 0.0, np.radians(10), np.radians(5)],  # Move 2: Left-Forward-Pitch
        [0.0,  -0.10, 0.08, np.radians(-8), np.radians(-8), 0.0],  # Move 3: Back-Up-Tilt
        [0.0,  0.0,  0.0, 0.0, 0.0, 0.0]                           # Return Home
    ]
    durations = [2.0, 2.5, 2.0, 2.5]
    
    print("Generating smooth S-Curve trajectory through waypoints...")
    time, pose, vel, accel = planner.generate_multi_waypoint_path(waypoints, durations)
    
    # Compute Actuator Velocities across the path using platform_core
    actuator_vels = np.zeros((len(time), 6))
    for i in range(len(time)):
        l_vels, _ = hexapod.compute_leg_velocities(pose[i], vel[i])
        actuator_vels[i] = l_vels
        
    # --- PLOTTING TRAJECTORY PROFILES ---
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    
    # Plot Pose Translations
    axs[0].plot(time, pose[:, 0], label='X (m)')
    axs[0].plot(time, pose[:, 1], label='Y (m)')
    axs[0].plot(time, pose[:, 2], label='Z (m)')
    axs[0].set_ylabel("Translation (m)")
    axs[0].set_title("Task-Space Pose Trajectory (Quintic S-Curves)")
    axs[0].grid(True)
    axs[0].legend(loc='upper right', fontsize=8)
    
    # Plot Linear Velocities
    axs[1].plot(time, vel[:, 0], label='Vx')
    axs[1].plot(time, vel[:, 1], label='Vy')
    axs[1].plot(time, vel[:, 2], label='Vz')
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].set_title("Task-Space Linear Velocities")
    axs[1].grid(True)
    axs[1].legend(loc='upper right', fontsize=8)
    
    # Plot Joint Actuator Speeds
    for i in range(6):
        axs[2].plot(time, actuator_vels[:, i], label=f'Actuator {i+1}')
    axs[2].set_ylabel("Actuator Velocity (m/s)")
    axs[2].set_title("Resulting Leg Velocities (Joint Space)")
    axs[2].set_xlabel("Time (s)")
    axs[2].grid(True)
    axs[2].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_trajectory_demo()
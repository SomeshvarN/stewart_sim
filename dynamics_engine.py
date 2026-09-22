import numpy as np
import matplotlib.pyplot as plt
from platform_core import AdvancedStewartPlatform

class StewartPlatformDynamics:
    """
    Inverse Dynamics Engine for 6-DOF Stewart Platform using Newton-Euler 
    Formulation & Virtual Work Mapping.
    """
    def __init__(self, hexapod: AdvancedStewartPlatform, mass=5.0, inertia_diag=(0.1, 0.1, 0.2)):
        self.hexapod = hexapod
        self.mass = mass  # Mass of upper platform + payload (kg)
        self.g = np.array([0, 0, -9.81])  # Gravity vector (m/s^2)
        
        # Principal Moments of Inertia in Body Frame (kg*m^2)
        self.I_body = np.diag(inertia_diag)

    def compute_actuator_forces(self, pose, accel_linear, accel_angular, vel_angular):
        """
        Calculates required force for each of the 6 linear actuators.
        
        Parameters:
            pose:          (6,) platform pose [x, y, z, roll, pitch, yaw]
            accel_linear:  (3,) linear acceleration vector [ax, ay, az] (m/s^2)
            accel_angular: (3,) angular acceleration vector [alpha_x, alpha_y, alpha_z] (rad/s^2)
            vel_angular:   (3,) angular velocity vector [wx, wy, wz] (rad/s^2)
            
        Returns:
            actuator_forces: (6,) vector of required actuator forces (N)
            wrench:          (6,) total 6-DOF net spatial wrench [Fx, Fy, Fz, Tx, Ty, Tz]
        """
        roll, pitch, yaw = pose[3:6]
        R = self.hexapod.rotation_matrix(roll, pitch, yaw)
        
        # 1. Total Translational Force (F_net = m * (a - g))
        # Note: subtracting g converts gravity into an upward reaction force
        F_net = self.mass * (np.array(accel_linear) - self.g)
        
        # 2. Total Rotational Torque in World Frame
        # Transform inertia tensor to world frame: I_world = R * I_body * R^T
        I_world = R @ self.I_body @ R.T
        w = np.array(vel_angular)
        alpha = np.array(accel_angular)
        
        # Euler's equation of motion: Tau = I*alpha + w x (I*w)
        Tau_net = I_world @ alpha + np.cross(w, I_world @ w)
        
        # Combine into 6-DOF Wrench Vector
        wrench = np.hstack([F_net, Tau_net])
        
        # 3. Jacobian Calculation
        J, _, _, _ = self.hexapod.compute_jacobian(pose)
        
        # 4. Force Mapping via Transpose Jacobian (Virtual Work)
        # Wrench = J^T * Actuator_Forces  ==> Actuator_Forces = inv(J^T) * Wrench
        JT_inv = np.linalg.pinv(J.T)
        actuator_forces = JT_inv @ wrench
        
        return actuator_forces, wrench


def run_dynamics_demo():
    hexapod = AdvancedStewartPlatform()
    dynamics = StewartPlatformDynamics(hexapod, mass=10.0)  # 10 kg payload
    
    time = np.linspace(0, 5, 200)
    dt = time[1] - time[0]
    
    leg_force_history = [[] for _ in range(6)]
    
    print("Simulating Inverse Dynamics along trajectory...")
    for t in time:
        # Define trajectory position and derivatives analytically
        pos = [0.1 * np.sin(t), 0.1 * np.cos(t), 0.05 * np.sin(2 * t)]
        ori = [np.radians(5 * np.sin(t)), np.radians(5 * np.cos(t)), 0.0]
        pose = np.array([*pos, *ori])
        
        # Accelerations (numerical approximations for demo trajectory)
        accel_lin = [-0.1 * np.sin(t), -0.1 * np.cos(t), -0.2 * np.sin(2 * t)]
        accel_ang = [np.radians(-5 * np.sin(t)), np.radians(-5 * np.cos(t)), 0.0]
        vel_ang   = [np.radians(5 * np.cos(t)), np.radians(-5 * np.sin(t)), 0.0]
        
        # Compute dynamic forces
        forces, _ = dynamics.compute_actuator_forces(pose, accel_lin, accel_ang, vel_ang)
        
        for i in range(6):
            leg_force_history[i].append(forces[i])
            
    # Plot Actuator Forces
    plt.figure(figsize=(10, 5))
    for i in range(6):
        plt.plot(time, leg_force_history[i], label=f'Actuator {i+1}')
        
    plt.title("Actuator Dynamic Force Requirements (10 kg Payload)")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N) [+ = Tension, - = Compression]")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_dynamics_demo()
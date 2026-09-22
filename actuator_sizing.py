import numpy as np
import matplotlib.pyplot as plt
from platform_core import AdvancedStewartPlatform
from dynamics_engine import StewartPlatformDynamics

class ActuatorSizingAnalyzer:
    """
    Analyzes trajectory execution data to compute engineering specifications
    for actuator hardware selection (motors, ball screws, drivers, structural safety).
    """
    def __init__(self, hexapod: AdvancedStewartPlatform, dynamics: StewartPlatformDynamics):
        self.hexapod = hexapod
        self.dynamics = dynamics

    def analyze_trajectory(self, time_array, pose_fn, accel_lin_fn, accel_ang_fn, vel_ang_fn):
        """
        Executes a trajectory time-series and calculates full dynamic loading profiles.
        """
        dt = time_array[1] - time_array[0]
        n_steps = len(time_array)
        
        forces = np.zeros((n_steps, 6))
        velocities = np.zeros((n_steps, 6))
        lengths = np.zeros((n_steps, 6))
        power = np.zeros((n_steps, 6))
        
        for idx, t in enumerate(time_array):
            pose = pose_fn(t)
            accel_lin = accel_lin_fn(t)
            accel_ang = accel_ang_fn(t)
            vel_ang = vel_ang_fn(t)
            
            # Kinematics
            l_lengths, _, _, _, _ = self.hexapod.inverse_kinematics(pose)
            
            # Task-space velocity estimate
            if idx > 0:
                pose_prev = pose_fn(time_array[idx - 1])
                platform_vel = (pose - pose_prev) / dt
            else:
                platform_vel = np.zeros(6)
                
            l_vels, _ = self.hexapod.compute_leg_velocities(pose, platform_vel)
            
            # Dynamics Forces
            f_act, _ = self.dynamics.compute_actuator_forces(pose, accel_lin, accel_ang, vel_ang)
            
            forces[idx] = f_act
            velocities[idx] = l_vels
            lengths[idx] = l_lengths
            power[idx] = f_act * l_vels  # Instantaneous mechanical power (W = N * m/s)
            
        return {
            "time": time_array,
            "forces": forces,
            "velocities": velocities,
            "lengths": lengths,
            "power": power,
            "dt": dt
        }

    def generate_hardware_report(self, results, rod_diameter_mm=12.0, E_GPa=210.0, safety_factor=1.5):
        """
        Computes key sizing metrics and checks mechanical buckling (Euler's Critical Load).
        
        Parameters:
            rod_diameter_mm: Piston/screw shaft diameter in mm (for buckling check)
            E_GPa:           Young's Modulus in GPa (210 GPa = Structural Steel)
            safety_factor:   Design safety margin applied to peak loads
        """
        forces = results["forces"]
        vels = results["velocities"]
        power = results["power"]
        lengths = results["lengths"]
        
        # 1. Force Metrics
        peak_compression = np.min(forces)  # Negative forces indicate compression
        peak_tension = np.max(forces)      # Positive forces indicate tension
        max_abs_force = np.max(np.abs(forces))
        rms_force = np.sqrt(np.mean(forces**2, axis=0))
        
        # 2. Speed & Stroke Metrics
        max_velocity = np.max(np.abs(vels))
        stroke_per_leg = np.ptp(lengths, axis=0)  # Peak-to-peak amplitude
        max_stroke = np.max(stroke_per_leg)
        
        # 3. Power & Energy Metrics
        peak_mech_power = np.max(np.abs(power))
        total_energy_joules = np.sum(np.abs(power)) * results["dt"]
        
        # 4. Euler Buckling Load Calculation: F_crit = (pi^2 * E * I) / (K * L)^2
        r_m = (rod_diameter_mm / 1000.0) / 2.0
        I_m4 = (np.pi * (r_m**4)) / 4.0  # Second moment of area (solid circular rod)
        E_Pa = E_GPa * 1e9
        max_len_m = np.max(lengths)
        K = 1.0  # Pin-pin joint boundary condition
        
        f_crit_buckling = (np.pi**2 * E_Pa * I_m4) / ((K * max_len_m)**2)
        
        # Output Terminal Report
        print("=" * 65)
        print("         STEWART PLATFORM ACTUATOR HARDWARE SIZING REPORT       ")
        print("=" * 65)
        print(f"Design Safety Factor Applied : {safety_factor}x")
        print("-" * 65)
        print(f" Peak Operating Force (Abs)  : {max_abs_force:.2f} N")
        print(f" Sized Peak Force (+Margin)  : {max_abs_force * safety_factor:.2f} N")
        print(f" Peak Tension Force          : {peak_tension:.2f} N")
        print(f" Peak Compression Force      : {peak_compression:.2f} N")
        print(f" Max Continuous RMS Force    : {np.max(rms_force):.2f} N")
        print("-" * 65)
        print(f" Max Linear Actuator Speed   : {max_velocity:.4f} m/s ({max_velocity * 1000:.1f} mm/s)")
        print(f" Max Required Stroke Length  : {max_stroke:.4f} m ({max_stroke * 1000:.1f} mm)")
        print(f" Min/Max Extended Lengths    : {np.min(lengths):.3f} m / {np.max(lengths):.3f} m")
        print("-" * 65)
        print(f" Peak Mechanical Power       : {peak_mech_power:.2f} W")
        print(f" Total Energy Spent (Cycle)  : {total_energy_joules:.2f} Joules")
        print("-" * 65)
        print(" MECHANICAL BUCKLING ANALYSIS (Euler Pin-Pin Rod)")
        print(f" Shaft Diameter Evaluated    : {rod_diameter_mm:.1f} mm (Steel, E={E_GPa} GPa)")
        print(f" Critical Buckling Limit     : {f_crit_buckling:.2f} N")
        
        peak_comp_abs = abs(peak_compression)
        if peak_comp_abs * safety_factor > f_crit_buckling:
            print(f" WARNING: Risk of Mechanical Buckling! Peak load ({peak_comp_abs:.1f}N)")
            print(f"          exceeds critical threshold ({f_crit_buckling:.1f}N). Increase rod diameter.")
        else:
            print(f" STATUS : SAFE from Buckling (Margin: {f_crit_buckling / (peak_comp_abs + 1e-6):.2f}x)")
        print("=" * 65)


def run_sizing_analysis():
    hexapod = AdvancedStewartPlatform()
    dynamics = StewartPlatformDynamics(hexapod, mass=15.0)  # 15 kg heavy payload
    analyzer = ActuatorSizingAnalyzer(hexapod, dynamics)
    
    # Define trajectory
    time_arr = np.linspace(0, 10, 400)
    
    def pose_fn(t):
        return np.array([
            0.15 * np.sin(t), 0.15 * np.cos(t), 0.08 * np.sin(2 * t),
            np.radians(8 * np.sin(t)), np.radians(8 * np.cos(t)), np.radians(5 * np.sin(1.5 * t))
        ])
        
    def accel_lin_fn(t):
        return np.array([-0.15 * np.sin(t), -0.15 * np.cos(t), -0.32 * np.sin(2 * t)])
        
    def accel_ang_fn(t):
        return np.array([np.radians(-8 * np.sin(t)), np.radians(-8 * np.cos(t)), np.radians(-11.25 * np.sin(1.5 * t))])
        
    def vel_ang_fn(t):
        return np.array([np.radians(8 * np.cos(t)), np.radians(-8 * np.sin(t)), np.radians(7.5 * np.cos(1.5 * t))])
        
    # Run analysis
    results = analyzer.analyze_trajectory(time_arr, pose_fn, accel_lin_fn, accel_ang_fn, vel_ang_fn)
    
    # Print Hardware Report
    analyzer.generate_hardware_report(results, rod_diameter_mm=14.0, safety_factor=1.5)
    
    # Plot Sizing Breakdown Graphs
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    for i in range(6):
        axs[0].plot(results["time"], results["forces"][:, i], label=f'Actuator {i+1}')
        axs[1].plot(results["time"], results["velocities"][:, i] * 1000)
        axs[2].plot(results["time"], results["power"][:, i])
        
    axs[0].set_ylabel("Force (N)")
    axs[0].set_title("Actuator Dynamic Loading Profiles")
    axs[0].grid(True)
    axs[0].legend(loc='upper right', fontsize=8)
    
    axs[1].set_ylabel("Velocity (mm/s)")
    axs[1].set_title("Linear Actuator Speeds")
    axs[1].grid(True)
    
    axs[2].set_ylabel("Mechanical Power (W)")
    axs[2].set_title("Instantaneous Power Output")
    axs[2].set_xlabel("Time (s)")
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_sizing_analysis()
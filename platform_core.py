import numpy as np

class AdvancedStewartPlatform:
    """
    Core Kinematics Engine for a 6-DOF Stewart Platform (Hexapod).
    
    Includes Inverse Kinematics (IK), Analytical Jacobian Matrix Computation,
    Jacobian Condition Indexing (Singularity Metrics), and Newton-Raphson 
    Numerical Forward Kinematics (FK).
    """
    def __init__(self, r_base=1.0, r_top=0.6, default_height=1.5, l_min=1.2, l_max=1.8):
        self.r_base = r_base
        self.r_top = r_top
        self.default_height = default_height
        self.l_min = l_min
        self.l_max = l_max
        
        # Base and Top Joint Distributions (paired joint layout)
        angles_base = np.radians([15, 105, 135, 225, 255, 345])
        angles_top  = np.radians([45, 75, 165, 195, 285, 315])
        
        # Base joint locations in World Frame (B)
        self.B = np.zeros((6, 3))
        self.B[:, 0] = self.r_base * np.cos(angles_base)
        self.B[:, 1] = self.r_base * np.sin(angles_base)
        
        # Top joint locations in Local Platform Frame (P_local)
        self.P_local = np.zeros((6, 3))
        self.P_local[:, 0] = self.r_top * np.cos(angles_top)
        self.P_local[:, 1] = self.r_top * np.sin(angles_top)

    def rotation_matrix(self, roll, pitch, yaw):
        """ Computes Z-Y-X Euler Angle Rotation Matrix (R = Rz * Ry * Rx) """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        
        return Rz @ Ry @ Rx

    def inverse_kinematics(self, pose):
        """
        Calculates leg lengths and joint positions given a 6-DOF platform pose.
        
        Parameters:
            pose: list or np.array -> [x, y, z, roll, pitch, yaw]
            
        Returns:
            leg_lengths: (6,) array of actuator lengths
            leg_vectors: (6, 3) array of leg direction vectors
            P_world:     (6, 3) array of top joint coordinates in world space
            valid_legs:  (6,) boolean mask indicating if legs are within min/max limits
            R:           (3, 3) rotation matrix
        """
        pos = pose[0:3] + np.array([0, 0, self.default_height])
        roll, pitch, yaw = pose[3:6]
        R = self.rotation_matrix(roll, pitch, yaw)
        
        leg_vectors = np.zeros((6, 3))
        leg_lengths = np.zeros(6)
        P_world = np.zeros((6, 3))
        valid_legs = np.ones(6, dtype=bool)
        
        for i in range(6):
            P_world[i] = pos + R @ self.P_local[i]
            leg_vectors[i] = P_world[i] - self.B[i]
            length = np.linalg.norm(leg_vectors[i])
            leg_lengths[i] = length
            
            if length < self.l_min or length > self.l_max:
                valid_legs[i] = False
                
        return leg_lengths, leg_vectors, P_world, valid_legs, R

    def compute_jacobian(self, pose):
        """
        Computes the 6x6 Inverse Kinematic Jacobian Matrix.
        
        Returns:
            J:           (6, 6) matrix mapping task-space velocities [vx, vy, vz, wx, wy, wz]
                         to joint-space speeds [l1_dot, ..., l6_dot]
            leg_lengths: (6,) array of actuator lengths
            leg_vectors: (6, 3) array of direction vectors
            R:           (3, 3) rotation matrix
        """
        leg_lengths, leg_vectors, _, _, R = self.inverse_kinematics(pose)
        J = np.zeros((6, 6))
        
        for i in range(6):
            s_i = leg_vectors[i] / leg_lengths[i]  # Unit vector along actuator
            r_i = R @ self.P_local[i]              # Top joint vector in world frame
            
            J[i, 0:3] = s_i
            J[i, 3:6] = np.cross(r_i, s_i)
            
        return J, leg_lengths, leg_vectors, R

    def compute_condition_number(self, pose):
        """
        Calculates Jacobian Condition Number kappa(J).
        
        Returns:
            kappa: float (1.0 = ideal/isotropic dexterity; higher values indicate singularity)
        """
        J, _, _, _ = self.compute_jacobian(pose)
        return np.linalg.cond(J)

    def forward_kinematics(self, target_leg_lengths, initial_pose_guess=None, max_iter=20, tol=1e-6):
        """
        Solves Forward Kinematics numerically using Newton-Raphson optimization.
        Maps target leg lengths back to a 6-DOF Platform Pose [x, y, z, roll, pitch, yaw].
        
        Parameters:
            target_leg_lengths: (6,) array of measured actuator lengths
            initial_pose_guess: (6,) initial seed pose array
            max_iter:           int, maximum iterations
            tol:                float, convergence error threshold
            
        Returns:
            pose:      (6,) solved platform pose [x, y, z, roll, pitch, yaw]
            converged: bool, True if algorithm met tolerance criteria
        """
        if initial_pose_guess is None:
            pose = np.zeros(6)
        else:
            pose = np.array(initial_pose_guess, dtype=float)
            
        for iteration in range(max_iter):
            current_lengths, _, _, _, _ = self.inverse_kinematics(pose)
            error = current_lengths - target_leg_lengths
            
            if np.linalg.norm(error) < tol:
                return pose, True  # Converged successfully
                
            J, _, _, _ = self.compute_jacobian(pose)
            
            # Newton-Raphson update step: x_{k+1} = x_k - J_pinv * error
            pose_update = np.linalg.pinv(J) @ error
            pose -= pose_update
            
        return pose, False  # Failed to converge within max_iter

    def compute_leg_velocities(self, pose, platform_velocity):
        """
        Maps platform task-space velocity to actuator linear speeds.
        
        Parameters:
            pose:              (6,) target pose [x, y, z, roll, pitch, yaw]
            platform_velocity: (6,) task space velocity vector [vx, vy, vz, wx, wy, wz]
            
        Returns:
            leg_velocities: (6,) actuator velocities (m/s)
            J:              (6, 6) Jacobian matrix
        """
        J, _, _, _ = self.compute_jacobian(pose)
        leg_velocities = J @ np.array(platform_velocity)
        return leg_velocities, J
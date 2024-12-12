
import numpy as np
import mujoco
from simulator import Simulator
from pathlib import Path
from typing import Dict
import os 
import pinocchio as pin
import matplotlib.pyplot as plt
from scipy.linalg import logm
from scipy.spatial.transform import Rotation as R

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
model = pin.buildModelFromMJCF(xml_path)
data = model.createData()

def skew_to_vector(skew_matrix):
    """Extract the vector from a skew-symmetric matrix"""
    skew_matrix = skew_matrix.reshape((3,3))
    return np.array([skew_matrix[2,1],
                     skew_matrix[0,2],
                     skew_matrix[1,0]])    
    
def vector_to_skew(x):
    """Get the skew-symmetric matrix from a vector"""
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])
    
def so3_error(R, Rd):
    """Compute orientation error"""    
    is_zero = (np.allclose(R, np.zeros(R.shape))) or (np.allclose(Rd, np.zeros(Rd.shape)))
    if is_zero: return np.zeros((3,))
    
    error_matrix = Rd@R.T
    error_log = logm(error_matrix)
    error_vector = skew_to_vector(error_log)
    return error_vector

def forward_kinematics(q_pos):
    pass

def create_circle_trajectory(radius, center, n_points, duration):
    """
    Create a circular trajectory with end effector pointing to the center.
    
    Args:
    radius (float): Radius of the circle
    center (array-like): Center of the circle [x, y, z]
    n_points (int): Number of points in the trajectory
    duration (float): Total duration of the trajectory in seconds
    
    Returns:
    dict: Trajectory information
    """
    t = np.linspace(0, duration, n_points)
    dt = duration / (n_points - 1)
    
    # Position
    theta = 2 * np.pi * t / duration
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(x, center[2])
    
    positions = np.column_stack((x, y, z))
    
    # Velocity
    dx = -radius * np.sin(theta) * (2 * np.pi / duration)
    dy = radius * np.cos(theta) * (2 * np.pi / duration)
    dz = np.zeros_like(dx)
    
    velocities = np.column_stack((dx, dy, dz))
    
    # Acceleration
    ddx = -radius * np.cos(theta) * (2 * np.pi / duration)**2
    ddy = -radius * np.sin(theta) * (2 * np.pi / duration)**2
    ddz = np.zeros_like(ddx)
    
    accelerations = np.column_stack((ddx, ddy, ddz))
    
    # Orientation (pointing to center)
    orientations = []
    angular_velocities = []
    angular_accelerations = []
    
    for i in range(n_points):
        # Direction vector from point on circle to center
        direction = center - positions[i]
        direction /= np.linalg.norm(direction)
        
        # Create rotation matrix
        rot_matrix = R.from_rotvec(np.cross([0, 0, 1], direction)).as_matrix()
        orientations.append(rot_matrix.flatten())
        
        if i > 0:
            # Angular velocity (finite difference)
            ang_vel = (R.from_matrix(rot_matrix.reshape(3,3)) * 
                       R.from_matrix(np.array(orientations[i-1]).reshape(3,3)).inv()).as_rotvec() / dt
            angular_velocities.append(ang_vel)
            
            if i > 1:
                # Angular acceleration (finite difference)
                ang_acc = (ang_vel - angular_velocities[-2]) / dt
                angular_accelerations.append(ang_acc)
    
    # Pad angular velocities and accelerations
    angular_velocities = [angular_velocities[0]] + angular_velocities
    angular_accelerations = [angular_accelerations[0]] + [angular_accelerations[0]] + angular_accelerations
    
    # Combine position and orientation
    state = np.column_stack((positions, orientations))
    d_state = np.column_stack((velocities, angular_velocities))
    dd_state = np.column_stack((accelerations, angular_accelerations))
    
    return {
        'state': state,
        'd_state': d_state,
        'dd_state': dd_state,
        'time': t
    }

def tsid_controller(sim, data: mujoco.MjData, state:np.array, target: np.array) -> np.ndarray:
    """Task space inverse dynamics controller."""
    model = sim.model
    
    kp = np.array([1000, 1000, 1000, 10, 10, 0.1])
    kd = np.array([200, 200, 200, 2, 2, 0.01])
    K0 = kp*np.eye(6)
    K1 = kd*np.eye(6)
    
    Xd = target['state'] # (12, )
    Rd = Xd[3:].reshape((3,3))
    Xd_dot = target['d_state'] # (6, )
    Xd_ddot = target['dd_state'] # (6, )
    
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
    ee_position = data.xpos[ee_id]
    ee_rotation = data.xmat[ee_id].reshape(3, 3)
    
    J = np.zeros((6, model.nv))
    mujoco.mj_jacBody(model, data, J[:3], J[3:], ee_id)
    J_inv = np.linalg.pinv(J)
    
    # Approximate J_d using finite differences
    h = 1e-6
    mujoco.mj_forward(model, data)
    J_prev = J.copy()
    data.qvel += h
    mujoco.mj_forward(model, data)
    mujoco.mj_jacBody(model, data, J[:3], J[3:], ee_id)
    J_d = (J - J_prev) / h
    data.qvel -= h
    mujoco.mj_forward(model, data)
    
    pos_term2 = K0[:3, :3]@(ee_position-Xd[:3]).reshape((3,1))
    ee_velocity = J@state['dq']
    pos_term3 = K1[:3, :3]@(ee_velocity[:3]-Xd_dot[:3]).reshape((3,1))
    R_dot = vector_to_skew(ee_velocity[3:]) @ ee_rotation
    Rd_dot = vector_to_skew(Xd_dot[3:]) @ Rd
    
    a_pos = Xd_ddot[:3] - pos_term2.flatten() - pos_term3.flatten() # (3, 1)

    a_angular = Xd_ddot[3:] - skew_to_vector(K0[3:, 3:]@vector_to_skew(so3_error(ee_rotation, Rd))) - skew_to_vector(K1[3:, 3:]@vector_to_skew(so3_error(Rd_dot, R_dot)))
    
    a_X = np.hstack((a_pos, a_angular))
    
    a_q = J_inv @ (a_X - J_d@sim.get_state()['dq'])
    print(f'a_q {a_q}')
    
    return a_q
    
def plot_results(times: np.ndarray, positions: np.ndarray, velocities: np.ndarray):
    """Plot and save simulation results."""
    # Joint positions plot
    plt.figure(figsize=(10, 6))
    for i in range(positions.shape[1]):
        plt.plot(times, positions[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Positions [rad]')
    plt.title('Joint Positions over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/05_positions.png')
    plt.close()
    
    # Joint velocities plot
    plt.figure(figsize=(10, 6))
    for i in range(velocities.shape[1]):
        plt.plot(times, velocities[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Velocities [rad/s]')
    plt.title('Joint Velocities over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('logs/plots/05_velocities.png')
    plt.close()

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning task space controller...")
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=True,
        show_viewer=False,
        record_video=True,
        video_path="logs/videos/05_joint_space.mp4",
        fps=30,
        width=1920,
        height=1080
    )
    sim.set_controller(tsid_controller)
    sim.reset()

    # Simulation parameters
    t = 0
    dt = sim.dt
    time_limit = 10.0
    sim.trajectory = create_circle_trajectory(
        radius=0.8,
        center=np.array([.0,.0,1.]),
        n_points=int(time_limit//dt),
        duration=int(time_limit) # TODO: check! 
    )
    current_target = 0
    reached = False
    
    # Data collection
    times = []
    positions = []
    velocities = []
    
    while t < time_limit:
        target = {'state': sim.trajectory['state'][current_target], # (12, )
                  'd_state': sim.trajectory['d_state'][current_target], # (6, )
                  'dd_state': sim.trajectory['dd_state'][current_target]} # (6, )
        while not reached and t < time_limit:
            state = sim.get_state()
            times.append(t)
            positions.append(state['q'])
            velocities.append(state['dq'])
            
            reached = np.allclose(target['state'][:3], state['ee_pos'], atol=0.005)
            tau = tsid_controller(sim, sim.data, state, target)
            
            sim.step(tau)
            
            if sim.record_video and len(sim.frames) < sim.fps * t:
                sim.frames.append(sim._capture_frame())
            t += dt
            # print(t)
        current_target += 1
        reached = False
    
    # Process and save results
    times = np.array(times)
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    print(f"Simulation completed: {len(times)} steps")
    print(f"Final joint positions: {positions[-1]}")

    print(positions)
    
    sim._save_video()
    plot_results(times, positions, velocities)

if __name__ == "__main__":
    main()
    # Example usage
    # radius = 0.5
    # center = np.array([0, 0, 1])
    # n_points = 100
    # duration = 10.0

    # trajectory = create_circle_trajectory(radius, center, n_points, duration)

    # print("State shape:", trajectory['state'].shape)
    # print("d_state shape:", trajectory['d_state'].shape)
    # print("dd_state shape:", trajectory['dd_state'].shape)
    # print(trajectory['state'][0])
    # print(trajectory['dd_state'][0])
    
    # print(trajectory['state'])
    
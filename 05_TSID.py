
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
pin_model = pin.buildModelFromMJCF(xml_path)
pin_data = pin_model.createData()

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
    # print(f'R {R}')
    # print(f'Rd {Rd}')
    return pin.log3(Rd@R.T)

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

def trajectory_to_point():
    state = np.array([.0, 1.0, 1.0, 2.0, .0, .0, .0, 1.0, .0, .0, .0, 1.0])
    d_state = 0.5*np.ones(6)
    dd_state = np.zeros(6)
    return {
        'state': [state],
        'd_state': [d_state],
        'dd_state': [dd_state],
        # 'time': t
    }
    

def tsid_controller(sim, data: mujoco.MjData, state:np.array, target: np.array) -> np.ndarray:
    """Task space inverse dynamics controller."""
    global pin_model, pin_data
    pin.computeAllTerms(pin_model, pin_data, state['q'], state['dq'])
    
    kp = np.array([200,  200, 200, 200,200,200])
    kd = np.array([200,  200, 200, 200,200,200])
    # kp = np.array([1000, 1000, 1000, 10, 10, 0.1])
    # kd = np.array([200, 200, 200, 2.5, 2.5, 0.01])
    K0 = kp*np.eye(6)
    K1 = kd*np.eye(6)
    
    Xd = target['state'] # (12, )
    Rd = Xd[3:].reshape((3,3))
    Xd_dot = target['d_state'] # (6, )
    Xd_ddot = target['dd_state'] # (6, )
    # print(f'target {target}')
    # a_q computations 
    ee_frame_id = pin_model.getFrameId("end_effector")
    ee_pose = pin_data.oMf[ee_frame_id]
    ee_position = ee_pose.translation # [x,y,z]
    ee_rotation = ee_pose.rotation # R
    frame = pin.WORLD
    J = pin.getFrameJacobian(pin_model, pin_data, ee_frame_id, frame)
    J_inv = np.linalg.inv(J)
    J_d = pin.computeJointJacobiansTimeVariation(pin_model, 
                                                 pin_data, 
                                                 state['q'], 
                                                 state['dq'])
    
    pos_err = ee_position-Xd[:3]
    orientation_err = so3_error(R=ee_rotation, Rd=Rd)
    err = np.hstack((pos_err, orientation_err))
    
    ee_twist = pin.getFrameVelocity(pin_model, pin_data, ee_frame_id, frame).vector
    d_pos_err = ee_twist[:3]-Xd_dot[:3]
    d_orientation_err = so3_error(R=vector_to_skew(ee_twist[3:]) @ ee_rotation,
                                  Rd=vector_to_skew(Xd_dot[3:]) @ Rd)
    d_err = np.hstack((d_pos_err, d_orientation_err))
    
    a_X = Xd_ddot - K0@err - K1@d_err
    
    a_q = J_inv @ (a_X - J_d@state['dq'])

    # computing control from a_q
    M = pin_data.M
    nle = pin_data.nle
    
    tau = M@a_q + nle
    # print(f'ee_rotation {ee_rotation}')
    
    return tau
    
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
    # sim.trajectory = create_circle_trajectory(
    #     radius=1.0,
    #     center=np.array([1.0,1.0,1.]),
    #     n_points=int(time_limit//dt),
    #     duration=int(time_limit) # TODO: check! 
    # )
    
    sim.trajectory = trajectory_to_point()
    current_target = 0
    reached = False
    
    # Data collection
    times = []
    positions = []
    velocities = []
    
    while t < time_limit and current_target < 1:
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
    # print(trajectory['state'][:10])
    # print(trajectory['dd_state'][:10])
    
    # print(trajectory['state'])
    
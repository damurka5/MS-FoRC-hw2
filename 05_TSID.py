
import numpy as np
from simulator import Simulator
from pathlib import Path
from typing import Dict
import os 
import pinocchio as pin
import matplotlib.pyplot as plt
from scipy.linalg import logm

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
model = pin.buildModelFromMJCF(xml_path)
data = model.createData()

def skew_to_vector(skew_matrix):
    """Extract the vector from a skew-symmetric matrix"""
    return np.array([skew_matrix[2,1],
                     skew_matrix[0,2],
                     skew_matrix[1,0]])    
    
def so3_error(R, Rd):
    """Compute orientation error"""    
    error_matrix = Rd@R.T
    error_log = logm(error_matrix)
    error_vector = skew_to_vector(error_log)
    return error_vector

def task_space_controller(q: np.ndarray, dq: np.ndarray, t: float, desired: Dict) -> np.ndarray:
    """Example task space controller."""
    
    kp = np.array([1000, 1000, 1000, 10, 10, 0.1])
    kd = np.array([200, 200, 200, 2, 2, 0.01])
    
    
    
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
    sim.set_controller(task_space_controller)
    sim.run(time_limit=10.0)

if __name__ == "__main__":
    main()
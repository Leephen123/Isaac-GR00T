import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
import os
import glob
from typing import Dict, List, Tuple, Optional
import json


default_joint_pos = [
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0
]

class RobotDataAnalyzer:
    """Robot Data Analyzer - Save separate charts for each joint"""
    
    def __init__(self, data_file: str):
        """
        Initialize data analyzer
        
        Args:
            data_file: Path to data file
        """
        self.data_file = data_file
        self.data_dir = os.path.dirname(data_file)
        
        # Joint names mapping
        self.joint_names = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
        ]
        
        # Set font to avoid Chinese character issues
        plt.rcParams['font.family'] = 'DejaVu Sans'  # Use a common font that supports more characters
        plt.rcParams['axes.unicode_minus'] = False  # Properly display minus signs
        
        # Load data
        self.df = self.load_data()
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        print(f"Loading data: {self.data_file}")
        print(f"Time range: {self.df['time'].min():.2f} - {self.df['time'].max():.2f} seconds")
        print(f"Number of joints: {self.df['joint_idx'].nunique()}")
        print(f"Number of joint names: {len(self.joint_names)}")
    
    def load_data(self) -> pd.DataFrame:
        """Load CSV data"""
        df = pd.read_csv(self.data_file)
        
        # Data cleaning
        df = df.dropna()  # Remove NaN values
        df = df[df['time'] >= 0]  # Remove negative time
        
        
        return df
    
    def get_joint_name(self, joint_idx: int) -> str:
        """Get joint name from joint index"""
        if 0 <= joint_idx < len(self.joint_names):
            return self.joint_names[joint_idx]
        else:
            return f"joint_{joint_idx}"
    
    def get_joint_data(self, joint_idx: int) -> pd.DataFrame:
        """Get data for specific joint"""
        return self.df[self.df['joint_idx'] == joint_idx].copy()
    
    def get_all_joints_data(self) -> Dict[int, pd.DataFrame]:
        """Get data dictionary for all joints"""
        joints = self.df['joint_idx'].unique()
        return {joint: self.get_joint_data(joint) for joint in joints}
    
    def calculate_joint_statistics(self, joint_idx: int) -> Dict:
        """Calculate statistics for single joint"""
        joint_data = self.get_joint_data(joint_idx)
        
        if len(joint_data) == 0:
            return {}
        
        stats = {
            'joint_idx': joint_idx,
            'joint_name': self.get_joint_name(joint_idx),
            'data_points': len(joint_data),
            'time_span': joint_data['time'].max() - joint_data['time'].min(),
            'mean_action': joint_data['action'].mean(),
            'std_action': joint_data['action'].std(),
            'mean_dof_pos': joint_data['dof_pos'].mean(),
            'std_dof_pos': joint_data['dof_pos'].std(),
            'mean_dof_vel': joint_data['dof_vel'].mean(),
            'std_dof_vel': joint_data['dof_vel'].std(),
            'mean_dof_torque': joint_data['dof_torque'].mean(),
            'max_dof_torque': joint_data['dof_torque'].abs().max(),
            'sampling_frequency': len(joint_data) / joint_data['time'].max() if joint_data['time'].max() > 0 else 0
        }

        return stats
    
    def plot_single_joint_analysis(self, joint_idx: int, save_dir: str = None):
        """Generate complete analysis chart for single joint"""
        if save_dir is None:
            save_dir = self.data_dir
        
        joint_data = self.get_joint_data(joint_idx)

        # action_idx = actor2env[joint_idx]
        # action_data = self.get_joint_data(action_idx)



        joint_name = self.get_joint_name(joint_idx)
        
        if len(joint_data) == 0:
            print(f"Warning: Joint {joint_idx} ({joint_name}) has no data")
            return
        
        # Create figure - 2x3 layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Joint Analysis: {joint_name} (Index: {joint_idx})', fontsize=16, y=0.98)
        
        # 1. Position Tracking
        ax = axes[0, 0]
        ax.plot(joint_data['time'], joint_data['action'] * 0.25 - default_joint_pos[joint_idx], 
               label='action', color=self.colors[0], linewidth=2, alpha=0.8)
        ax.plot(joint_data['time'], joint_data['dof_pos'], 
               label='dof_pos', color=self.colors[1], linewidth=2, alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.set_title('Position Tracking')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Velocity Tracking
        ax = axes[0, 1]
        ax.plot(joint_data['time'], joint_data['dof_vel'], 
               label='dof_vel', color=self.colors[2], linewidth=2, alpha=0.8)
        ax.plot(joint_data['time'], joint_data['dof_vel'], 
               label='dof_vel', color=self.colors[3], linewidth=2, alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (rad/s)')
        ax.set_title('Velocity Tracking')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Torque Comparison
        ax = axes[0, 2]
        ax.plot(joint_data['time'], joint_data['dof_torque'], 
               label='dof_torque', color=self.colors[4], linewidth=2, alpha=0.8)
        ax.plot(joint_data['time'], joint_data['dof_torque'], 
               label='dof_torque', color=self.colors[5], linewidth=2, alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Torque (Nm)')
        ax.set_title('Torque Output')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        safe_joint_name = joint_name.replace(' ', '_').replace('/', '_')
        filename = f"joint_{joint_idx:02d}_{safe_joint_name}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"Saved joint {joint_idx} chart: {filename}")
        
        return save_path
    
    def plot_joint_comparison(self, joint_indices: List[int], save_dir: str = None):
        """Compare performance of multiple joints"""
        if save_dir is None:
            save_dir = self.data_dir
        
        n_joints = len(joint_indices)
        if n_joints == 0:
            return
        
        # Create comparison chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-Joint Performance Comparison', fontsize=16)
        
        # Create joint labels
        joint_labels = [f"{idx}\n{self.get_joint_name(idx)[:15]}..." for idx in joint_indices]
        
        # 3. Position Standard Deviation Comparison
        pos_stds = []
        for joint_idx in joint_indices:
            stats = self.calculate_joint_statistics(joint_idx)
            if stats:
                pos_stds.append(stats['std_dof_pos'])
        
        axes[1, 0].bar(range(len(pos_stds)), pos_stds, color=self.colors[:len(pos_stds)])
        axes[1, 0].set_xticks(range(len(pos_stds)))
        axes[1, 0].set_xticklabels(joint_labels, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Position Std Dev (rad)')
        axes[1, 0].set_title('Position Variation by Joint')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Maximum Torque Comparison
        max_torques = []
        for joint_idx in joint_indices:
            stats = self.calculate_joint_statistics(joint_idx)
            if stats:
                max_torques.append(stats['max_dof_torque'])
        
        axes[1, 1].bar(range(len(max_torques)), max_torques, color=self.colors[:len(max_torques)])
        axes[1, 1].set_xticks(range(len(max_torques)))
        axes[1, 1].set_xticklabels(joint_labels, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Max Torque (Nm)')
        axes[1, 1].set_title('Max Computed Torque by Joint')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save comparison chart
        save_path = os.path.join(save_dir, "joint_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved joint comparison chart: {save_path}")
        
        return save_path
    
       
    def parse_imu_data(self, imu_str: str) -> np.ndarray:
        """Parse IMU string data to numpy array"""
        try:
            # Remove brackets and split by spaces
            clean_str = imu_str.replace('[', '').replace(']', '').strip()
            # Split by spaces and convert to float
            values = [float(x) for x in clean_str.split()]
            return np.array(values)
        except Exception as e:
            print(f"Error parsing IMU data: {imu_str}, error: {e}")
            return np.array([])
    
    def quat_rotate_inverse_on_gravity(self, q: np.ndarray) -> np.ndarray:
        """
        Rotate gravity vector [0, 0, -1] by inverse quaternion
        
        Args:
            q: quaternion in [w, x, y, z] format
            
        Returns:
            projected_gravity: gravity vector in body frame
        """
        if len(q) != 4:
            return np.array([np.nan, np.nan, np.nan])
        
        # Gravity vector in world frame (pointing down)
        v = np.array([0., 0., -1.], dtype=np.float32)
        
        q_w = q[0]  # w component
        q_vec = q[1:]  # [x, y, z] components
        
        # Quaternion inverse rotation formula
        a = v * (2.0 * q_w**2 - 1.0)
        b = 2.0 * q_w * np.cross(q_vec, v)
        c = 2.0 * q_vec * np.dot(q_vec, v)
        
        projected_gravity = a - b + c
        return projected_gravity
    
    def quaternion_to_rpy(self, quaternion: np.ndarray) -> np.ndarray:
        """
        Convert quaternion [w, x, y, z] to roll, pitch, yaw angles in radians
        
        Args:
            quaternion: numpy array of shape (4,) in [w, x, y, z] format
            
        Returns:
            numpy array of shape (3,) containing [roll, pitch, yaw] in radians
        """
        if len(quaternion) != 4:
            return np.array([np.nan, np.nan, np.nan])
        
        w, x, y, z = quaternion
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def get_imu_data(self) -> pd.DataFrame:
        """Get and parse IMU data - 每29行取一个IMU数据"""
        # Check if IMU columns exist
        if 'root_rot' not in self.df.columns or 'root_ang_vel' not in self.df.columns:
            return pd.DataFrame()
        
        imu_indices = list(range(0, len(self.df), 29))
        imu_data = self.df.iloc[imu_indices][['time', 'root_rot', 'root_ang_vel']].copy()
        
        if len(imu_data) == 0:
            return pd.DataFrame()
        
        # Parse IMU data strings to arrays
        imu_data['root_rot_parsed'] = imu_data['root_rot'].apply(self.parse_imu_data)
        imu_data['root_ang_vel_parsed'] = imu_data['root_ang_vel'].apply(self.parse_imu_data)
        
        # Check if parsing was successful
        valid_rot = imu_data['root_rot_parsed'].apply(lambda x: len(x) > 0)
        valid_vel = imu_data['root_ang_vel_parsed'].apply(lambda x: len(x) > 0)
        
        imu_data = imu_data[valid_rot & valid_vel].copy()
        
        if len(imu_data) == 0:
            return pd.DataFrame()
        
        # Convert quaternion to RPY angles
        imu_data['rpy_angles'] = imu_data['root_rot_parsed'].apply(self.quaternion_to_rpy)
        
        # Extract individual RPY components (in degrees)
        for i, component in enumerate(['roll', 'pitch', 'yaw']):
            imu_data[f'rpy_{component}'] = imu_data['rpy_angles'].apply(
                lambda x: np.degrees(x[i]) if len(x) == 3 and not np.isnan(x[i]) else np.nan
            )
        
        # Calculate projected gravity
        imu_data['projected_gravity'] = imu_data['root_rot_parsed'].apply(
            lambda q: self.quat_rotate_inverse_on_gravity(q) if len(q) == 4 else np.array([np.nan, np.nan, np.nan])
        )
        
        # Extract gravity components
        for i, axis in enumerate(['x', 'y', 'z']):
            imu_data[f'gravity_{axis}'] = imu_data['projected_gravity'].apply(
                lambda g: g[i] if len(g) == 3 and not np.isnan(g[i]) else np.nan
            )
        
        # Calculate gravity magnitude
        imu_data['gravity_magnitude'] = imu_data['projected_gravity'].apply(
            lambda g: np.linalg.norm(g) if len(g) == 3 and not np.isnan(g).any() else np.nan
        )
        
        # Extract angular velocity components
        for i in range(3):  # root_ang_vel has 3 components
            imu_data[f'root_ang_vel_{i}'] = imu_data['root_ang_vel_parsed'].apply(
                lambda x: x[i] if len(x) > i else np.nan
            )
        
        return imu_data
    
    def calculate_imu_statistics(self) -> Dict:
        """Calculate statistics for IMU data"""
        imu_data = self.get_imu_data()
        
        if len(imu_data) == 0:
            return {}
        
        stats = {
            'data_points': len(imu_data),
            'time_span': imu_data['time'].max() - imu_data['time'].min(),
            'sampling_frequency': len(imu_data) / imu_data['time'].max() if imu_data['time'].max() > 0 else 0
        }
        
        # Add statistics for RPY angles
        rpy_components = ['roll', 'pitch', 'yaw']
        for component in rpy_components:
            col = f'rpy_{component}'
            if col in imu_data.columns:
                stats.update({
                    f'mean_{col}': imu_data[col].mean(),
                    f'std_{col}': imu_data[col].std(),
                    f'min_{col}': imu_data[col].min(),
                    f'max_{col}': imu_data[col].max()
                })
        
        # Add statistics for gravity projection
        gravity_components = ['x', 'y', 'z']
        for component in gravity_components:
            col = f'gravity_{component}'
            if col in imu_data.columns:
                stats.update({
                    f'mean_{col}': imu_data[col].mean(),
                    f'std_{col}': imu_data[col].std(),
                    f'min_{col}': imu_data[col].min(),
                    f'max_{col}': imu_data[col].max()
                })
        
        # Add statistics for gravity magnitude
        if 'gravity_magnitude' in imu_data.columns:
            stats.update({
                'mean_gravity_magnitude': imu_data['gravity_magnitude'].mean(),
                'std_gravity_magnitude': imu_data['gravity_magnitude'].std(),
                'min_gravity_magnitude': imu_data['gravity_magnitude'].min(),
                'max_gravity_magnitude': imu_data['gravity_magnitude'].max()
            })
        
        # Add statistics for angular velocity
        for i in range(3):
            col = f'root_ang_vel_{i}'
            if col in imu_data.columns:
                stats.update({
                    f'mean_{col}': imu_data[col].mean(),
                    f'std_{col}': imu_data[col].std(),
                    f'min_{col}': imu_data[col].min(),
                    f'max_{col}': imu_data[col].max()
                })
        
        return stats
    
    def plot_imu_analysis(self, save_dir: str = None):
        """Generate complete analysis chart for IMU data using RPY angles and gravity projection"""
        if save_dir is None:
            save_dir = self.data_dir
        
        imu_data = self.get_imu_data()
        
        if len(imu_data) == 0:
            print(f"Warning: No IMU data available")
            return
        
        # Create figure with subplots for IMU components
        fig, axes = plt.subplots(3, 1, figsize=(15, 15))
        fig.suptitle('IMU Data Analysis - RPY Angles, Gravity Projection and Angular Velocity', fontsize=16, y=0.98)
        
        # 1. RPY Angles (in degrees)
        ax = axes[0]
        rpy_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Colors for roll, pitch, yaw
        
        rpy_components = ['roll', 'pitch', 'yaw']
        for i, component in enumerate(rpy_components):
            col = f'rpy_{component}'
            if col in imu_data.columns:
                ax.plot(imu_data['time'], imu_data[col], 
                       label=f'{component.capitalize()}', color=rpy_colors[i], linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Body Orientation - Roll, Pitch, Yaw Angles')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Gravity Projection Components
        ax = axes[1]
        gravity_colors = ['#d62728', '#9467bd', '#8c564b']  # Colors for gravity x, y, z
        
        gravity_components = ['x', 'y', 'z']
        for i, component in enumerate(gravity_components):
            col = f'gravity_{component}'
            if col in imu_data.columns:
                ax.plot(imu_data['time'], imu_data[col], 
                       label=f'Gravity {component.upper()}', color=gravity_colors[i], linewidth=2, alpha=0.8)
        
        # Add reference line for ideal gravity (should be [0, 0, -1] when upright)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Zero Reference')
        ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Ideal Downward Gravity')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Gravity Projection')
        ax.set_title('Projected Gravity in Body Frame')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Root Angular Velocity Components
        ax = axes[2]
        vel_colors = ['#9467bd', '#8c564b', '#e377c2']  # Different colors for each component
        vel_labels = ['X-axis', 'Y-axis', 'Z-axis']
        
        for i in range(3):
            col = f'root_ang_vel_{i}'
            if col in imu_data.columns:
                ax.plot(imu_data['time'], imu_data[col], 
                       label=vel_labels[i], color=vel_colors[i], linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.set_title('Angular Velocity Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        filename = "imu_analysis_rpy_gravity.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"Saved IMU analysis chart: {filename}")
        
        # Also create individual component plots for better detail
        self.plot_imu_individual_components(imu_data, save_dir)
        
        return save_path
    
    def plot_imu_individual_components(self, imu_data: pd.DataFrame, save_dir: str):
        """Create individual plots for each IMU component using RPY and gravity"""
        
        # 1. Individual RPY Angle Components
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Body Orientation - Individual RPY Components', fontsize=16, y=0.98)
        
        rpy_components = ['Roll', 'Pitch', 'Yaw']
        rpy_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, component in enumerate(['roll', 'pitch', 'yaw']):
            col = f'rpy_{component}'
            if col in imu_data.columns:
                axes[i].plot(imu_data['time'], imu_data[col], 
                           color=rpy_colors[i], linewidth=2, alpha=0.8)
                axes[i].set_xlabel('Time (s)')
                axes[i].set_ylabel(f'{rpy_components[i]} (degrees)')
                axes[i].set_title(f'Body {rpy_components[i]} Angle')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = "imu_rpy_components.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved RPY components chart: {filename}")
        
        # 2. Individual Gravity Projection Components
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Projected Gravity - Individual Components', fontsize=16, y=0.98)
        
        gravity_components = ['X', 'Y', 'Z']
        gravity_colors = ['#d62728', '#9467bd', '#8c564b']
        
        for i, component in enumerate(['x', 'y', 'z']):
            col = f'gravity_{component}'
            if col in imu_data.columns:
                axes[i].plot(imu_data['time'], imu_data[col], 
                           color=gravity_colors[i], linewidth=2, alpha=0.8)
                axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[i].axhline(y=-1 if component == 'z' else 0, color='red', linestyle='--', alpha=0.5)
                axes[i].set_xlabel('Time (s)')
                axes[i].set_ylabel(f'Gravity {gravity_components[i]}')
                axes[i].set_title(f'Projected Gravity - {gravity_components[i]} Component')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = "imu_gravity_components.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved gravity components chart: {filename}")
        
        # 3. Individual Angular Velocity Components
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Angular Velocity - Individual Components', fontsize=16, y=0.98)
        
        component_names = ['X-axis', 'Y-axis', 'Z-axis']
        colors_vel = ['#9467bd', '#8c564b', '#e377c2']
        
        for i in range(3):
            col = f'root_ang_vel_{i}'
            if col in imu_data.columns:
                axes[i].plot(imu_data['time'], imu_data[col], 
                           color=colors_vel[i], linewidth=2, alpha=0.8)
                axes[i].set_xlabel('Time (s)')
                axes[i].set_ylabel(f'Velocity (rad/s)')
                axes[i].set_title(f'Angular Velocity - {component_names[i]}')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = "imu_velocity_components.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved velocity components chart: {filename}")
        
        # 4. Gravity Magnitude and 3D Visualization
        self.plot_gravity_analysis(imu_data, save_dir)
        
        # 5. RPY 3D Scatter Plot (optional visualization)
        self.plot_imu_3d_rpy(imu_data, save_dir)
    
    def plot_gravity_analysis(self, imu_data: pd.DataFrame, save_dir: str):
        """Create specialized gravity analysis plots"""
        
        # Gravity magnitude over time
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if 'gravity_magnitude' in imu_data.columns:
            ax.plot(imu_data['time'], imu_data['gravity_magnitude'], 
                   color='purple', linewidth=2, alpha=0.8, label='Gravity Magnitude')
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, 
                      label='Ideal Magnitude (1.0)')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Gravity Magnitude')
            ax.set_title('Gravity Vector Magnitude Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = "imu_gravity_magnitude.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved gravity magnitude chart: {filename}")
        
        # 3D gravity vector visualization
        self.plot_gravity_3d_trajectory(imu_data, save_dir)
    
    def plot_gravity_3d_trajectory(self, imu_data: pd.DataFrame, save_dir: str):
        """Create 3D visualization of gravity vector trajectory"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            if all(col in imu_data.columns for col in ['gravity_x', 'gravity_y', 'gravity_z']):
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Color points by time for temporal visualization
                time_norm = (imu_data['time'] - imu_data['time'].min()) / (imu_data['time'].max() - imu_data['time'].min())
                scatter = ax.scatter(imu_data['gravity_x'], imu_data['gravity_y'], imu_data['gravity_z'],
                                   c=time_norm, cmap='viridis', alpha=0.6, s=20)
                
                # Plot ideal gravity vector (pointing down)
                ax.quiver(0, 0, 0, 0, 0, -1, color='red', arrow_length_ratio=0.1, linewidth=3, label='Ideal Down')
                
                ax.set_xlabel('Gravity X')
                ax.set_ylabel('Gravity Y')
                ax.set_zlabel('Gravity Z')
                ax.set_title('3D Gravity Vector Trajectory in Body Frame')
                ax.legend()
                
                # Set equal aspect ratio
                max_range = max(imu_data['gravity_x'].max() - imu_data['gravity_x'].min(),
                              imu_data['gravity_y'].max() - imu_data['gravity_y'].min(),
                              imu_data['gravity_z'].max() - imu_data['gravity_z'].min())
                mid_x = (imu_data['gravity_x'].max() + imu_data['gravity_x'].min()) * 0.5
                mid_y = (imu_data['gravity_y'].max() + imu_data['gravity_y'].min()) * 0.5
                mid_z = (imu_data['gravity_z'].max() + imu_data['gravity_z'].min()) * 0.5
                ax.set_xlim(mid_x - max_range * 0.6, mid_x + max_range * 0.6)
                ax.set_ylim(mid_y - max_range * 0.6, mid_y + max_range * 0.6)
                ax.set_zlim(mid_z - max_range * 0.6, mid_z + max_range * 0.6)
                
                # Add colorbar for time
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Normalized Time')
                
                plt.tight_layout()
                filename = "imu_gravity_3d_trajectory.png"
                save_path = os.path.join(save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved 3D gravity trajectory chart: {filename}")
                
        except ImportError:
            print("3D plotting not available - skipping 3D gravity visualization")
    
    def plot_imu_3d_rpy(self, imu_data: pd.DataFrame, save_dir: str):
        """Create 3D scatter plot of RPY angles"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            if all(col in imu_data.columns for col in ['rpy_roll', 'rpy_pitch', 'rpy_yaw']):
                # Color points by time for temporal visualization
                time_norm = (imu_data['time'] - imu_data['time'].min()) / (imu_data['time'].max() - imu_data['time'].min())
                scatter = ax.scatter(imu_data['rpy_roll'], imu_data['rpy_pitch'], imu_data['rpy_yaw'],
                                   c=time_norm, cmap='viridis', alpha=0.6, s=20)
                
                ax.set_xlabel('Roll (degrees)')
                ax.set_ylabel('Pitch (degrees)')
                ax.set_zlabel('Yaw (degrees)')
                ax.set_title('3D RPY Orientation Trajectory')
                
                # Add colorbar for time
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Normalized Time')
                
                plt.tight_layout()
                filename = "imu_rpy_3d_trajectory.png"
                save_path = os.path.join(save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved 3D RPY trajectory chart: {filename}")
                
        except ImportError:
            print("3D plotting not available - skipping 3D RPY visualization")
    
    def generate_all_joints_analysis(self, save_dir: str = None):
        """Generate analysis charts for all joints"""
        if save_dir is None:
            save_dir = self.data_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Get all joint indices with data
        all_joints = sorted(self.df['joint_idx'].unique())
        print(f"Starting to generate analysis charts for {len(all_joints)} joints...")
        
        # Generate detailed analysis chart for each joint
        saved_files = []
        for joint_idx in all_joints:
            try:
                save_path = self.plot_single_joint_analysis(joint_idx, save_dir)
                if save_path:
                    saved_files.append(save_path)
            except Exception as e:
                print(f"Error generating chart for joint {joint_idx}: {e}")
        
        # Generate IMU analysis
        try:
            self.plot_imu_analysis(save_dir)
            print("Generated IMU analysis charts")
        except Exception as e:
            print(f"Error generating IMU analysis charts: {e}")
        
        # Generate joint comparison chart (select first 12 joints to avoid overcrowding)
        if len(all_joints) > 1:
            try:
                joints_to_compare = all_joints[:min(12, len(all_joints))]
                self.plot_joint_comparison(joints_to_compare, save_dir)
            except Exception as e:
                print(f"Error generating joint comparison chart: {e}")
        
        # Generate statistics report
        stats_report = self.generate_statistics_report(all_joints, save_dir)
        
        print(f"\nAnalysis completed! Generated {len(saved_files)} joint analysis charts")
        print(f"All files saved in: {save_dir}")
        
        return saved_files
    
    def generate_statistics_report(self, joint_indices: List[int], save_dir: str):
        """Generate statistics report"""
        stats_data = []
        
        for joint_idx in joint_indices:
            stats = self.calculate_joint_statistics(joint_idx)
            if stats:
                stats_data.append(stats)
        
        if not stats_data:
            return None
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(stats_data)
        
        # Save as CSV
        csv_path = os.path.join(save_dir, "joint_statistics.csv")
        stats_df.to_csv(csv_path, index=False, float_format='%.6f')
        
        # Generate text report
        report_path = os.path.join(save_dir, "analysis_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Robot Joint Control Performance Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Data File: {os.path.basename(self.data_file)}\n")
            f.write(f"Analysis Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Joints: {len(stats_data)}\n")
            f.write(f"Time Range: {self.df['time'].min():.2f} - {self.df['time'].max():.2f} seconds\n")
            f.write(f"Total Data Points: {len(self.df)}\n\n")
            
            # IMU统计信息
            imu_stats = self.calculate_imu_statistics()
            if imu_stats:
                f.write("IMU Statistics:\n")
                f.write(f"  IMU Data Points: {imu_stats['data_points']}\n")
                f.write(f"  IMU Time Span: {imu_stats['time_span']:.2f} s\n")
                f.write(f"  Sampling Frequency: {imu_stats['sampling_frequency']:.1f} Hz\n\n")
                
                f.write("RPY Angle Statistics (degrees):\n")
                for component in ['roll', 'pitch', 'yaw']:
                    col = f'rpy_{component}'
                    if f'mean_{col}' in imu_stats:
                        f.write(f"  {component.capitalize()}: mean={imu_stats[f'mean_{col}']:.2f}°, "
                               f"std={imu_stats[f'std_{col}']:.2f}°, "
                               f"range=[{imu_stats[f'min_{col}']:.2f}°, {imu_stats[f'max_{col}']:.2f}°]\n")
                
                f.write("\nProjected Gravity Statistics:\n")
                for component in ['x', 'y', 'z']:
                    col = f'gravity_{component}'
                    if f'mean_{col}' in imu_stats:
                        f.write(f"  Gravity {component.upper()}: mean={imu_stats[f'mean_{col}']:.3f}, "
                               f"std={imu_stats[f'std_{col}']:.3f}, "
                               f"range=[{imu_stats[f'min_{col}']:.3f}, {imu_stats[f'max_{col}']:.3f}]\n")
                
                if 'mean_gravity_magnitude' in imu_stats:
                    f.write(f"  Gravity Magnitude: mean={imu_stats['mean_gravity_magnitude']:.3f}, "
                           f"std={imu_stats['std_gravity_magnitude']:.3f}, "
                           f"range=[{imu_stats['min_gravity_magnitude']:.3f}, {imu_stats['max_gravity_magnitude']:.3f}]\n")
                
                f.write("\nAngular Velocity Statistics (rad/s):\n")
                for i in range(3):
                    col = f'root_ang_vel_{i}'
                    if f'mean_{col}' in imu_stats:
                        f.write(f"  Component {i}: mean={imu_stats[f'mean_{col}']:.3f}, "
                               f"std={imu_stats[f'std_{col}']:.3f}, "
                               f"range=[{imu_stats[f'min_{col}']:.3f}, {imu_stats[f'max_{col}']:.3f}]\n")
                f.write("\n")
            
            f.write("Detailed Statistics by Joint:\n")
            f.write("-" * 60 + "\n")
            
            for stats in stats_data:
                f.write(f"\nJoint {stats['joint_idx']}: {stats['joint_name']}\n")
                f.write(f"  Data Points: {stats['data_points']}\n")
                f.write(f"  Time Span: {stats['time_span']:.2f} s\n")
                f.write(f"  Sampling Frequency: {stats['sampling_frequency']:.1f} Hz\n")
                f.write(f"  Max Computed Torque: {stats['max_dof_torque']:.3f} Nm\n")
        
        print(f"Statistics report saved to: {report_path}")
        print(f"Statistics data saved to: {csv_path}")
        
        return stats_df

def batch_analyze_folder(folder_path: str):
    """批量分析文件夹中的所有CSV文件"""
    # 查找所有CSV文件
    csv_pattern = os.path.join(folder_path, "**", "*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    # 过滤掉不需要的文件
    csv_files = [f for f in csv_files if not any(skip in f for skip in 
                ['_statistics.csv', '_config.json', '.npz', 'joint_'])]
    
    if not csv_files:
        print(f"在 {folder_path} 中未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    success_count = 0
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] 分析: {os.path.basename(csv_file)}")
        print(f"数据目录: {os.path.dirname(csv_file)}")
        
        try:
            # 创建分析器 - 会自动使用数据文件所在目录保存图表
            analyzer = RobotDataAnalyzer(csv_file)
            analyzer.generate_all_joints_analysis()
            
            success_count += 1
            print(f"✓ 完成分析，图表已保存到数据目录")
            
        except Exception as e:
            print(f"✗ 分析失败: {e}")
    
    print(f"\n批量分析完成! 成功: {success_count}/{len(csv_files)}")


# 使用示例
if __name__ == "__main__":

        # folder_path = "tmp/tracking_data/20251112/"
        # batch_analyze_folder(folder_path)
        
        # data_file = "tmp/tracking_data/20251112/133538/CmdTask_20251112_133538.csv"
        data_file = "tmp/tracking_data/20251128/163758/CmdTask_20251128_163758.csv"
        analyzer = RobotDataAnalyzer(data_file)
        analyzer.generate_all_joints_analysis()
        
        # # 如果只想分析特定关节，可以使用：
        # # analyzer.plot_single_joint_analysis(0)  # 分析关节0
        # # analyzer.plot_single_joint_analysis(1)  # 分析关节1
        
        # print("分析完成！")
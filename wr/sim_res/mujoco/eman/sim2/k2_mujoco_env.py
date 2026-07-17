# -*- coding: utf-8 -*-
import sys, os 

import time 
import numpy as np
import torch
import threading
from tqdm import tqdm

import mujoco, mujoco.viewer
import imageio
from scipy.spatial.transform import Rotation as R

from sim_res.mujoco.eman import EI_ROOT_DIR
from sim_res.mujoco.eman.base.config import Config
from sim_res.mujoco.eman.sim2.base_env import BaseEnv

from sim_res.mujoco.eman.sim2.g1_mujoco_msg import LowCmd_, LowState_
from sim_res.mujoco.eman.sim2.g1_mujoco_msg import MujocoLowCmdSubscriber, MujocoLowCmdPublisher
from sim_res.mujoco.eman.sim2.g1_mujoco_msg import MujocoLowStateSubscriber, MujocoLowStatePublisher
from sim_res.mujoco.eman.sim2.g1_mujoco_msg import MujocoKeypointsSubscriber
from sim_res.mujoco.unitree_sdk2py.utils.crc import CRC

class K2MujocoEnv():
    def __init__(self, robot: str, DT: float, hands: str):
        self.robot = robot
        self.DT = DT
        self.hands = hands
        assert self.hands in ["box", "dex3"], "Only support box and dex3 hands now"
        
        self.num_skeleton_dof = 28  

        left_hand_num_dof = 0 if self.hands == "box" else 7
        right_hand_num_dof = 0 if self.hands == "box" else 7
        
        self.joint_concat_index = [*range(21)] + \
            [*range(28, 28 + left_hand_num_dof)] + \
            [*range(21, 28)] + \
            [*range(28 + left_hand_num_dof, 28 + left_hand_num_dof + right_hand_num_dof)]
            
        self.joint_split_index = {
            "skeleton": [*range(21)] + [*range(21 + left_hand_num_dof, 28 + left_hand_num_dof)],
            "hands": [*range(21, 21 + left_hand_num_dof)] + [*range(28 + left_hand_num_dof, 28 + left_hand_num_dof + right_hand_num_dof)],
        }
        
        self._start_pose = None
        self._write_start_pose = False
        self._lock_write_start_pose = threading.Lock()

        self.lowcmd_subscriber = MujocoLowCmdSubscriber()
        self.lowstate_publisher = MujocoLowStatePublisher()
        time.sleep(1)

        if self.hands == "dex3":
            from sim_res.mujoco.eman.sim2.g1_mujoco_msg import MujocoLeftHandCmdPublisher, MujocoLeftHandCmdSubscriber
            from sim_res.mujoco.eman.sim2.g1_mujoco_msg import MujocoLeftHandStatePublisher, MujocoLeftHandStateSubscriber
            from sim_res.mujoco.eman.sim2.g1_mujoco_msg import MujocoRightHandCmdPublisher, MujocoRightHandCmdSubscriber
            from sim_res.mujoco.eman.sim2.g1_mujoco_msg import MujocoRightHandStatePublisher, MujocoRightHandStateSubscriber

            self.left_handcmd_subscriber = MujocoLeftHandCmdSubscriber()
            self.left_handstate_publisher = MujocoLeftHandStatePublisher()
            self.right_handcmd_subscriber = MujocoRightHandCmdSubscriber()
            self.right_handstate_publisher = MujocoRightHandStatePublisher()
            time.sleep(1)

        # visualization
        self.visualization = True
        self.keypoints_subscriber = MujocoKeypointsSubscriber()
        self.recording = False
        self.recording_fps = 60
        self.video_filename = "tmp/k2_mujoco_simulation.mp4"
        self.video_writer = None  
        
        self._keyboard_state = [0] * 40
        self._lock_keyboard_state = threading.Lock()
        self.setup_keyboard_listener()

        self.step_thread()
    
    def stop_recording_video(self):
        self.recording = False 
        if self.video_writer is not None:
            try:
                self.video_writer.close()
            except:
                pass
            self.video_writer = None

    def resolve_compatibility(self, operation: str, **kwargs):
        if operation == "concat":
            index = self.joint_concat_index
            return kwargs["state"][index]
        if operation == "split":
            skeleton_index = self.joint_split_index["skeleton"]
            hands_index = self.joint_split_index["hands"]
            return kwargs["state"][skeleton_index], kwargs["state"][hands_index]
        
    def step_thread(self):
        max_fps = 800

        if self.robot == "k2":
           xml = "{EI_ROOT_DIR}/resources/robots/kepler/scene_28dof.xml".format(EI_ROOT_DIR=EI_ROOT_DIR)
        else: 
           raise ValueError(f"Invalid robot: {self.robot}")

        mj_model = mujoco.MjModel.from_xml_path(xml)
        mj_model.opt.timestep = 1. / max_fps
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_step(mj_model, mj_data)
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        renderer = mujoco.Renderer(mj_model, 480, 480) 
        
        video_dir = os.path.dirname(self.video_filename)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)  
        # self.video_writer = imageio.get_writer(self.video_filename, fps=50)
        
        written_frames = 0
        cam = setup_tracking_camera(mj_model, body_name="pelvis") 
        
        effort_limit = np.inf

        from loop_rate_limiters import RateLimiter
        rate_limiter = RateLimiter(frequency=max_fps, warn=False)
        last_render_time = time.time()
        last_record_time = time.time()

        for _ in range(30):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))

        step_count = 0
        step_time = time.time()
        try:  
            while True:
                step_count += 1
                if step_count % max_fps == 0:
                    print(f"Step: {int(time.time() - step_time)}s, FPS: {int(step_count / (time.time() - step_time))}")

                with self._lock_write_start_pose:
                    if self._write_start_pose:
                        start_pos = self._start_pose.copy()
                        mj_data.qpos[:3] = start_pos["root_xyz"].copy()
                        mj_data.qpos[3:7] = start_pos["root_rot"].copy()
                        mj_data.qpos[7:7+self.num_skeleton_dof] = start_pos["joint_pos"].copy()

                        mj_data.qvel = 0.
                        mj_data.ctrl = 0.
                        mujoco.mj_step(mj_data.model, mj_data)
                        self._write_start_pose = False
                
                action, kp, kd = [], [], []
                low_cmd = self.lowcmd_subscriber.read()
                
                for i in range(self.num_skeleton_dof):
                    action.append(low_cmd.motor_cmd[i].q)
                    kp.append(low_cmd.motor_cmd[i].kp)
                    kd.append(low_cmd.motor_cmd[i].kd)
                    
                action = np.array(action, dtype=np.float32)
                kp = np.array(kp, dtype=np.float32)
                kd = np.array(kd, dtype=np.float32)
                
                if action.any() != 0 and self.recording == False:
                    self.recording = True
                    print("Start recording video to ", self.video_filename)

                if self.hands == "dex3":
                    _action, _kp, _kd = [], [], []
                    left_hand_cmd = self.left_handcmd_subscriber.read()
                    right_hand_cmd = self.right_handcmd_subscriber.read()
                    for i in range(7):
                        _action.append(left_hand_cmd.motor_cmd[i].q)
                        _kp.append(left_hand_cmd.motor_cmd[i].kp)
                        _kd.append(left_hand_cmd.motor_cmd[i].kd)
                    for i in range(7):
                        _action.append(right_hand_cmd.motor_cmd[i].q)
                        _kp.append(right_hand_cmd.motor_cmd[i].kp)
                        _kd.append(right_hand_cmd.motor_cmd[i].kd)
                    _action = np.array(_action, dtype=np.float32)
                    _kp = np.array(_kp, dtype=np.float32)
                    _kd = np.array(_kd, dtype=np.float32)
                    
                    action = self.resolve_compatibility("concat", state=np.concatenate([action, _action], axis=0))
                    kp = self.resolve_compatibility("concat", state=np.concatenate([kp, _kp], axis=0))
                    kd = self.resolve_compatibility("concat", state=np.concatenate([kd, _kd], axis=0))

                # Attention: the first seven in mujoco are pos+quat of Root, real joint from [7:]
                joint_pos = mj_data.qpos.astype(np.float32).copy()[7:]
                joint_vel = mj_data.qvel.astype(np.float32).copy()[6:]

                torque = (action - joint_pos) * kp - joint_vel * kd
                torque = np.clip(torque, -effort_limit, effort_limit)
                
                actual_nu = mj_data.model.nu 
                mj_data.ctrl[:actual_nu] = torque[:actual_nu].copy()
                mujoco.mj_step(mj_data.model, mj_data)

                rate_limiter.sleep()

                joint_pos = mj_data.qpos.astype(np.float32).copy()[7:]
                joint_vel = mj_data.qvel.astype(np.float32).copy()[6:]
                joint_torques = mj_data.ctrl.astype(np.float32).copy()
                root_rot = mj_data.qpos.astype(np.float32).copy()[3:7]
                root_xyz = mj_data.qpos.astype(np.float32).copy()[0:3]
                root_ang_vel = mj_data.qvel.astype(np.float32).copy()[3:6]

                joint_pos, hands_joint_pos = self.resolve_compatibility("split", state=joint_pos)
                joint_vel, hands_joint_vel = self.resolve_compatibility("split", state=joint_vel)
                
                # Pad torques if actual_nu < num_skeleton_dof due to XML commenting
                if len(joint_torques) < len(action):
                    padded_torques = np.zeros_like(action)
                    padded_torques[:len(joint_torques)] = joint_torques
                    joint_torques = padded_torques
                    
                joint_torques, hands_joint_torques = self.resolve_compatibility("split", state=joint_torques)

                self.lowstate_publisher.low_state_msg.tick = int(time.time())
                for i in range(self.num_skeleton_dof):
                    self.lowstate_publisher.low_state_msg.motor_state[i].q = joint_pos[i]
                    self.lowstate_publisher.low_state_msg.motor_state[i].dq = joint_vel[i]
                    self.lowstate_publisher.low_state_msg.motor_state[i].tau_est = joint_torques[i]
                
                self.lowstate_publisher.low_state_msg.imu_state.quaternion = root_rot
                self.lowstate_publisher.low_state_msg.imu_state.gyroscope = root_ang_vel
                with self._lock_keyboard_state:
                    self.lowstate_publisher.low_state_msg.wireless_remote = self._keyboard_state.copy()
                # self.lowstate_publisher.low_state_msg.crc = CRC().Crc(self.lowstate_publisher.low_state_msg)
                self.lowstate_publisher.low_state_msg.crc = 0
                self.lowstate_publisher.publish()

                if self.visualization:
                    draw_points = self.keypoints_subscriber.read().keypoints
                    draw_points = np.array(draw_points, dtype=np.float32).reshape(-1, 3)
                    max_geom_idx = min(draw_points.shape[0], viewer.user_scn.ngeom)
                    for i in range(max_geom_idx):
                        viewer.user_scn.geoms[i].pos = draw_points[i]
            
                if self.recording and time.time() - last_record_time > 1./self.recording_fps:
                    renderer.update_scene(mj_data, cam)
                    frame = renderer.render()
                    # self.video_writer.append_data(frame)  
                    written_frames += 1
                    last_record_time = time.time()
            
                if time.time() - last_render_time > self.DT:
                    viewer.sync()
                    last_render_time = time.time()
        except KeyboardInterrupt:  
            print("\nReceived exit signal, cleaning up...")
        finally:  
            self.stop_recording_video()
            viewer.close()
            print("Simulation stopped, resources released.")

    def setup_keyboard_listener(self):
        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char in ["s", "a", "t"]:
                    data = {"s": [0x04, 0x00], "a": [0x00, 0x01], "t": [0x00, 0x04]}[key.char]
                    with self._lock_keyboard_state:
                        self._keyboard_state[2] = data[0]
                        self._keyboard_state[3] = data[1]
            except AttributeError: 
                pass

        def on_release(key):
            try:
                if hasattr(key, 'char') and key.char in ["s", "a", "t"]:
                    with self._lock_keyboard_state:
                        self._keyboard_state[2] = 0
                        self._keyboard_state[3] = 0
            except AttributeError: 
                pass

        from pynput import keyboard
        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

def add_visual_capsule(scene, point1, point2, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    vec = point2 - point1
    length = np.linalg.norm(vec)
    if length < 1e-6:  
        return
    center = (point1 + point2) / 2.0
    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, vec)
    angle = np.arccos(np.clip(np.dot(z_axis, vec/length), -1.0, 1.0))
    if np.linalg.norm(axis) < 1e-6:
        rot_mat = np.eye(3)  
    else:
        axis = axis / np.linalg.norm(axis)
        c, s = np.cos(angle), np.sin(angle)
        cx = 1 - c
        rot_mat = np.array([
            [c + axis[0]*axis[0]*cx, axis[0]*axis[1]*cx - axis[2]*s, axis[0]*axis[2]*cx + axis[1]*s],
            [axis[1]*axis[0]*cx + axis[2]*s, c + axis[1]*axis[1]*cx, axis[1]*axis[2]*cx - axis[0]*s],
            [axis[2]*axis[0]*cx - axis[1]*s, axis[2]*axis[1]*cx + axis[0]*s, c + axis[2]*axis[2]*cx]
        ])
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.array([radius, length/2, 0.0]),      
        center, rot_mat.flatten(), rgba.astype(np.float32)           
    )
    scene.ngeom += 1

def setup_tracking_camera(model, body_name="Body"):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING  
    try:
        cam.trackbodyid = model.body(body_name).id    
    except KeyError:
        # Fallback if pelvic link has different name
        pass
    cam.distance = 3.5                         
    cam.azimuth = 60                        
    cam.elevation = -15                           
    return cam

if __name__ == "__main__":
    env = K2MujocoEnv(robot='k2', DT=0.02, hands="box")
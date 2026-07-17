from typing import Union
import numpy as np
import time
import torch

from sim_res.mujoco.eman.base.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from sim_res.mujoco.eman.base.rotation_helper import transform_imu_data

class RIS_Mode:
    def __init__(self, id=0, status=0x01, timeout=0):
        self.motor_mode = 0
        self.id = id & 0x0F  
        self.status = status & 0x07  
        self.timeout = timeout & 0x01  

    def mode_to_uint8(self):
        self.motor_mode |= (self.id & 0x0F)
        self.motor_mode |= (self.status & 0x07) << 4
        self.motor_mode |= (self.timeout & 0x01) << 7
        return self.motor_mode

class K2Env():
    def __init__(self, sim2, config, hands: str):
        self.sim2 = sim2
        self.config = config
        self.hands = hands

        net = None
        global KeyMap

        if self.config.robot in ["g1", "k2"]:
            from sim_res.mujoco.unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
            from sim_res.mujoco.unitree_sdk2py.core.channel import ChannelSubscriber
            from sim_res.mujoco.unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
            from sim_res.mujoco.unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
            from sim_res.mujoco.unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
            from sim_res.mujoco.unitree_sdk2py.utils.crc import CRC
            from sim_res.mujoco.eman.base.remote_controller import RemoteController
            from sim_res.mujoco.eman.base.remote_controller import KeyMap as KM
            assert self.config.msg_type == "hg"
        elif self.config.robot == "o1":
            from westlake_sdkpy.core.channel import ChannelPublisher, ChannelFactoryInitialize
            from westlake_sdkpy.core.channel import ChannelSubscriber
            from westlake_sdkpy.idl.default import agile_msg_dds__LowCmd_ as westlake_msg_dds__LowCmd_
            from westlake_sdkpy.idl.default import agile_msg_dds__LowState_ as westlake_msg_dds__LowState_
            from westlake_sdkpy.idl.agile.msg.dds_ import LowCmd_ as westlake_LowCmdHG
            from westlake_sdkpy.idl.agile.msg.dds_ import LowState_ as westlake_LowStateHG
            from sim_res.mujoco.eman.base.remote_controller import RemoteController
            from sim_res.mujoco.eman.base.remote_controller import KeyMap as KM
            assert self.config.msg_type == "agile"
        else: raise ValueError(f"Invalid robot: {self.config.robot}")

        ChannelFactoryInitialize(0, net)
        self.remote_controller = RemoteController()
        KeyMap = KM

        if self.sim2 == "mujoco":
            from sim_res.mujoco.eman.sim2.topic import TOPIC_MUJOCO_LOW_CMD, TOPIC_MUJOCO_LOW_STATE, TOPIC_MUJOCO_LEFTHAND_CMD, TOPIC_MUJOCO_RIGHTHAND_CMD
            _lowcmd_topic = TOPIC_MUJOCO_LOW_CMD
            _lowstate_topic = TOPIC_MUJOCO_LOW_STATE
            _left_handcmd_topic = TOPIC_MUJOCO_LEFTHAND_CMD
            _right_handcmd_topic = TOPIC_MUJOCO_RIGHTHAND_CMD
        if self.sim2 == "real":
            _lowcmd_topic = "rt/lowcmd"
            _lowstate_topic = "rt/lowstate"
            _left_handcmd_topic = "left_lowcmd_topic"
            _right_handcmd_topic = "right_lowcmd_topic"

        if self.config.robot in ["g1", "k2"]:
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0
            self.motor_mode = 0
            self.lowcmd_publisher_ = ChannelPublisher(_lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(_lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

            assert self.hands in ["box", "dex3"], "Only support dex3 hands now"
            if self.hands == "dex3":
                from sim_res.mujoco.unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
                from sim_res.mujoco.unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
                self.left_hand_cmd = unitree_hg_msg_dds__HandCmd_()
                self.LeftHandCmb_publisher_ = ChannelPublisher(_left_handcmd_topic, HandCmd_)
                self.LeftHandCmb_publisher_.Init()

                self.right_hand_cmd = unitree_hg_msg_dds__HandCmd_()
                self.RightHandCmb_publisher_ = ChannelPublisher(_right_handcmd_topic, HandCmd_)
                self.RightHandCmb_publisher_.Init()

        elif self.config.robot == "o1":
            self.low_cmd = westlake_msg_dds__LowCmd_()
            self.low_state = westlake_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0
            self.motor_mode = 0x02
            self.lowcmd_publisher_ = ChannelPublisher(_lowcmd_topic, westlake_LowCmdHG)
            self.lowcmd_publisher_.Init()
            self.lowstate_subscriber = ChannelSubscriber(_lowstate_topic, westlake_LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        
        self.wait_for_low_state()
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

        from sim_res.mujoco.eman.sim2.g1_mujoco_msg import MujocoKeypointsPublisher
        self.mujoco_keypoints_publisher = MujocoKeypointsPublisher()

    def LowStateHgHandler(self, msg):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd):
        self.lowcmd_publisher_.Write(cmd)

    def send_hand_cmd(self, left_cmd, right_cmd):
        if self.hands == "dex3":
            self.LeftHandCmb_publisher_.Write(left_cmd)
            self.RightHandCmb_publisher_.Write(right_cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.DT)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.DT)

    def move_to_default_pos(self):
        total_time = 2
        num_step = int(total_time / self.config.DT)
        kps = self.config.kps
        kds = self.config.kds
        default_joint_pos = self.config.default_joint_pos
        
        dof_size = self.config.num_actions 
        
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[i].q
        
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = j 
                target_pos = default_joint_pos[j]
                self.low_cmd.motor_cmd[motor_idx].mode = self.motor_mode
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.DT)

        if self.hands == "dex3":
            hand_dof_size = 7
            for i in range(hand_dof_size):
                ris_mode = RIS_Mode(id = i, status = 0x01)
                motor_mode = ris_mode.mode_to_uint8()
                self.left_hand_cmd.motor_cmd[i].mode = motor_mode
                self.left_hand_cmd.motor_cmd[i].q    = 0
                self.left_hand_cmd.motor_cmd[i].dq   = 0
                self.left_hand_cmd.motor_cmd[i].tau  = 0
                self.left_hand_cmd.motor_cmd[i].kp   = self.config.hands[self.hands]["kps"]["left"][i]
                self.left_hand_cmd.motor_cmd[i].kd   = self.config.hands[self.hands]["kds"]["left"][i]

                self.right_hand_cmd.motor_cmd[i].mode = motor_mode
                self.right_hand_cmd.motor_cmd[i].q    = 0
                self.right_hand_cmd.motor_cmd[i].dq   = 0
                self.right_hand_cmd.motor_cmd[i].tau  = 0
                self.right_hand_cmd.motor_cmd[i].kp   = self.config.hands[self.hands]["kps"]["right"][i]
                self.right_hand_cmd.motor_cmd[i].kd   = self.config.hands[self.hands]["kds"]["right"][i]
            self.send_hand_cmd(self.left_hand_cmd, self.right_hand_cmd)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(self.config.num_actions):
                motor_idx = i
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_joint_pos[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.DT)
    
    def get_raw_obs(self):
        num_joints = self.config.num_actions
        joint_pos = np.zeros(num_joints, dtype=np.float32)
        joint_vel = np.zeros(num_joints, dtype=np.float32)
        joint_torque = np.zeros(num_joints, dtype=np.float32)

        for i in range(num_joints):
            joint_pos[i] = self.low_state.motor_state[i].q
            joint_vel[i] = self.low_state.motor_state[i].dq
            joint_torque[i] = self.low_state.motor_state[i].tau_est

        quat = self.low_state.imu_state.quaternion
        quat = np.array(quat, dtype=np.float32)
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)

        root_rot = np.array(quat, dtype=np.float32)
        root_ang_vel = np.array(ang_vel, dtype=np.float32)

        return {"joint_pos": joint_pos, "joint_vel": joint_vel, "joint_torque": joint_torque,
                "root_rot": root_rot, "root_ang_vel": root_ang_vel, "KeyMap.X": self.remote_controller.button[KeyMap.X]}
    
    def step(self, step_action):
        action = step_action["action"]
        num_joints = self.config.num_actions
        for i in range(num_joints):
            motor_idx = i
            self.low_cmd.motor_cmd[motor_idx].q = action[i]
            self.low_cmd.motor_cmd[motor_idx].dq = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        self.send_cmd(self.low_cmd)

        if self.hands == "dex3":
            left_hand_target = step_action["hands"][:7]
            right_hand_target = step_action["hands"][7:]
            hand_num_joints = 7
            for i in range(hand_num_joints):
                self.left_hand_cmd.motor_cmd[i].q = left_hand_target[i]
                self.right_hand_cmd.motor_cmd[i].q = right_hand_target[i]
            self.send_hand_cmd(self.left_hand_cmd, self.right_hand_cmd)

        return
    
    def visualize(self, visualization):
        if self.sim2 == "real": return
        if (not visualization) or (visualization.get("keypoints", None) is None): return
        self.mujoco_keypoints_publisher.publish(visualization["keypoints"])
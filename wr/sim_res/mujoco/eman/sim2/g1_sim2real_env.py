from typing import Union
import numpy as np
import time
import torch

from sim_res.mujoco.unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from sim_res.mujoco.unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from sim_res.mujoco.unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from sim_res.mujoco.unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from sim_res.mujoco.unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from sim_res.mujoco.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from sim_res.mujoco.unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from sim_res.mujoco.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from sim_res.mujoco.unitree_sdk2py.utils.crc import CRC


from sim_res.mujoco.eman.base.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from sim_res.mujoco.eman.base.rotation_helper import transform_imu_data
from sim_res.mujoco.eman.base.remote_controller import RemoteController, KeyMap

from .base_env import BaseEnv


class RIS_Mode:
    def __init__(self, id=0, status=0x01, timeout=0):
        self.motor_mode = 0
        self.id = id & 0x0F  # 4 bits for id
        self.status = status & 0x07  # 3 bits for status
        self.timeout = timeout & 0x01  # 1 bit for timeout

    def mode_to_uint8(self):
        self.motor_mode |= (self.id & 0x0F)
        self.motor_mode |= (self.status & 0x07) << 4
        self.motor_mode |= (self.timeout & 0x01) << 7
        return self.motor_mode

class G1Sim2RealEnv(BaseEnv):
    def __init__(self, net, config, hands: str):
        super().__init__(config=config, hands=hands)

        # Initialize DDS communication
        ChannelFactoryInitialize(0, net)
        self.remote_controller = RemoteController()

        if self.config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(self.config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(self.config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        else: raise ValueError("Invalid msg_type")

        assert self.hands in ["box", "dex3"], "Only support dex3 hands now"
        if self.hands == "dex3":
            from sim_res.mujoco.unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
            from sim_res.mujoco.unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
            self.left_hand_cmd = unitree_hg_msg_dds__HandCmd_()
            self.LeftHandCmb_publisher_ = ChannelPublisher(self.config.hands[self.hands]["left_lowcmd_topic"], HandCmd_)
            self.LeftHandCmb_publisher_.Init()

            self.right_hand_cmd = unitree_hg_msg_dds__HandCmd_()
            self.RightHandCmb_publisher_ = ChannelPublisher(self.config.hands[self.hands]["right_lowcmd_topic"], HandCmd_)
            self.RightHandCmb_publisher_.Init()
        
        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if self.config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        else: raise ValueError("Invalid msg_type")

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def send_hand_cmd(self, left_cmd, right_cmd):
        if self.hands == "dex3":
            self.LeftHandCmb_publisher_.Write(left_cmd)
            self.RightHandCmb_publisher_.Write(right_cmd)

    def wait_for_low_state(self):
        while self.low_state.sequence == 0:
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
        # print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.DT)
        
        # dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps
        kds = self.config.kds
        default_joint_pos = self.config.default_joint_pos
        dof_size = 29 #len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[i].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = j #dof_idx[j]
                target_pos = default_joint_pos[j]
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

            for i in range(hand_dof_size):
                ris_mode = RIS_Mode(id = i, status = 0x01)
                motor_mode = ris_mode.mode_to_uint8()
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
            for i in range(29):
                motor_idx = i
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_joint_pos[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.DT)
    
    def get_raw_obs(self):
        num_joints = 29
        joint_pos = np.zeros(num_joints, dtype=np.float32)
        joint_vel = np.zeros(num_joints, dtype=np.float32)

        for i in range(num_joints):
            joint_pos[i] = self.low_state.motor_state[i].q
            joint_vel[i] = self.low_state.motor_state[i].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        quat = np.array(quat, dtype=np.float32) #[[1,2,3,0]]
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        root_rot = np.array(quat, dtype=np.float32)
        root_ang_vel = np.array(ang_vel, dtype=np.float32)

        return {"joint_pos": joint_pos, "joint_vel": joint_vel, 
                "root_rot": root_rot, "root_ang_vel": root_ang_vel}
    
    def step(self, step_action):
        action = step_action["action"]
        num_joints = 29
        for i in range(num_joints):
            motor_idx = i
            self.low_cmd.motor_cmd[motor_idx].q = action[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        self.send_cmd(self.low_cmd)

        if self.hands == "dex3":
            left_hand_target = step_action["hands"][:7]
            right_hand_target = step_action["hands"][7:]
            assert len(left_hand_target) == 7 and len(right_hand_target) == 7
            hand_num_joints = 7
            for i in range(hand_num_joints):
                self.left_hand_cmd.motor_cmd[i].q = left_hand_target[i]
                self.right_hand_cmd.motor_cmd[i].q = right_hand_target[i]
            self.send_hand_cmd(self.left_hand_cmd, self.right_hand_cmd)

        return
    
    def visualize(self, visualization):
        return

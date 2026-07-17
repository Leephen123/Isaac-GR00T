# from legged_gym import LEGGED_GYM_ROOT_DIR
from sim_res.mujoco.eman import EI_ROOT_DIR
import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.robot = config["robot"]
            self.xml = config["xml"]
            self.DT = config["DT"]
            self.robot = config['robot']

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.actor_path = config["actor_path"].replace("{EI_ROOT_DIR}", EI_ROOT_DIR)
            self.motion_file = config["motion_file"].replace("{EI_ROOT_DIR}", EI_ROOT_DIR)

            self.kps = config["kps"]
            self.kds = config["kds"]
            self.effort_limit = None
            self.default_joint_pos = np.array(config["default_joint_pos"], dtype=np.float32)

            self.hands = config["hands"]

            
            self.num_actions = config["num_actions"]
            self.action_scale = config["action_scale"]

            self.obs_scale_projected_gravity_b = config["obs_scale_projected_gravity_b"]
            self.obs_scale_root_ang_vel_b = config["obs_scale_root_ang_vel_b"]
            self.obs_scale_joint_pos = config["obs_scale_joint_pos"]
            self.obs_scale_joint_vel = config["obs_scale_joint_vel"]
            self.obs_scale_action = config["obs_scale_action"]

            self.obs_scale_cmd = np.array(config["obs_scale_cmd"], dtype=np.float32)
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

            self.actor2env = [config["actor_joint_names"].index(j) for j in config["env_joint_names"]]
            self.env2actor = [config["env_joint_names"].index(j) for j in config["actor_joint_names"]]

            # self.BODY_NAMES = config["BODY_NAMES"]


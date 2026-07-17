import sys
import os
from collections import OrderedDict

from sim_res.mujoco.eman import EI_ROOT_DIR
import time
import numpy as np
import torch
from loop_rate_limiters import RateLimiter

from sim_res.mujoco.eman.base.config import Config
from sim_res.mujoco.eman.base.base_controller import BaseController
from sim_res.mujoco.eman.base.command_helper import create_damping_cmd
from sim_res.mujoco.eman.base.rotation_helper import get_gravity_orientation, quat_rotate_inverse_on_gravity

class G1Controller(BaseController):
    def __init__(self, net, sim2: str, registers: OrderedDict, config: Config, hands: str, device: str) -> None:
        if device not in ["cpu", "cuda"]: device = {"mujoco": "cpu", "real": "cuda"}[sim2]
        super().__init__(registers=registers, DT=config.DT, device=device)

        self.config = config
        self.sim2 = sim2
        hu_fps = int(1. / self.config.DT)
        
        from sim_res.mujoco.eman.sim2.g1_env import G1Env
        self.hu_env = G1Env(sim2=self.sim2, config=config, hands=hands)

        from sim_res.mujoco.eman.tasks.cmd_task import CmdTask
        cmd_task = CmdTask(DT=config.DT, device=device, robot=self.config.robot)
        
        for task_name, task_id in registers.items():
            self.tasks[task_id] = cmd_task
            self.tasks[task_name] = cmd_task

        self.counter = 0
        self.last_action = np.zeros(config.num_actions, dtype=np.float32)

        self.remote_controller = self.hu_env.remote_controller
        self.rate_limiter = RateLimiter(frequency=hu_fps, warn=True, name="main loop")

    def zero_torque_state(self):
        self.hu_env.zero_torque_state()
    
    def move_to_default_pos(self):
        print("Moving to default pos.")
        self.hu_env.move_to_default_pos()

    def default_pos_state(self):
        self.hu_env.default_pos_state()

    def add_joint_data(self, last_action, raw_obs):
        actions = last_action.copy()  
        dof_pos = raw_obs['joint_pos'].copy()    
        dof_vel = raw_obs['joint_vel'].copy()  
        dof_torque = raw_obs['joint_torque'].copy()
        root_rot = raw_obs['root_rot'].copy()  
        root_vel = raw_obs['root_ang_vel'].copy()  
        current_time = time.time() - self.start_time
        
        for i in range(self.config.num_actions):
            data_row = [current_time, i, actions[i], dof_pos[i], dof_vel[i], dof_torque[i], root_rot, root_vel]
            self.data_saver.add_data(data_row)

    def run(self):
        self.counter += 1
        raw_obs = self.hu_env.get_raw_obs()
        raw_obs["last_action"] = self.last_action
        
        self.add_joint_data(self.last_action[self.tasks[self.register].actor2env], raw_obs)

        action_dict = self.get_action(raw_obs)
        raw_action = action_dict.pop("raw_action")
        # print(raw_action)
        # print(f"Actor Raw Action Max: {np.max(raw_action):.2f}, Min: {np.min(raw_action):.2f}")

        visualization = self.get_visualization(raw_obs, raw_action)
        self.last_action = raw_action

        self.hu_env.step(action_dict)
        self.hu_env.visualize(visualization)

        self.rate_limiter.sleep()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, help="network interface")
    parser.add_argument("--config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    config_path = f"{EI_ROOT_DIR}/eman/{args.config}"
    config = Config(config_path)

    registers = OrderedDict(CmdTask="1")
    controller = G1Controller(args.net, sim2="mujoco", registers=registers, config=config, hands="box", device="cpu")
    
    if controller.config.robot in ["g1", "k2", "sber", "pm"]:
        from sim_res.mujoco.eman.base.remote_controller import KeyMap
    elif controller.config.robot == "o1":
        from sim_res.mujoco.eman.base.remote_controller_o1 import KeyMap

    controller.zero_torque_state()
    controller.move_to_default_pos()
    controller.default_pos_state()

    # reset time to prevent the accumulation of action errors
    hu_fps = int(1. / controller.config.DT)
    controller.rate_limiter = RateLimiter(frequency=hu_fps, warn=True, name="main loop")

    running_step = 0
    while True:
        try:
            running_step += 1
            controller.run()
            if running_step % 500 == 0: print(f"Step: {running_step}")
            
            if controller.config.robot in ["g1", "k2", "sber", "pm"]:
                if (controller.remote_controller is not None) and controller.remote_controller.button[KeyMap.select] == 1:
                    break
            elif controller.config.robot == "o1":
                if (controller.remote_controller is not None) and controller.remote_controller.button[KeyMap.BACK] == 1:
                    break
        except KeyboardInterrupt:
            break

    create_damping_cmd(controller.hu_env.low_cmd)
    controller.hu_env.send_cmd(controller.hu_env.low_cmd)
    print("Exit")
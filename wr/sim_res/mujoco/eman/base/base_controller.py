import os.path
from abc import ABC, abstractmethod

import threading

import numpy as np
import time
import torch

from collections import OrderedDict

from .config import Config
from .remote_controller import RemoteController, KeyMap

from sim_res.mujoco.eman.base.data import DataSaverThread

class BaseController(ABC):
    def __init__(self, registers: OrderedDict, DT: float, device: str):
        self.tasks = dict()
        self.keyboards = dict()
        for register, keyboard in registers.items():
            self.keyboards[register] = keyboard
        
        self.registers = [register for register in registers.keys()]
        self.register = self.registers[0]  # Default register

        self.data_lock = threading.Lock()
        self.start_time = time.time()
        
        self.data_dir = "tmp/tracking_data/" + time.strftime('%Y%m%d')+"/" + time.strftime('%H%M%S')
        os.makedirs(self.data_dir, exist_ok=True)
        self.data_file_name = os.path.join(
            self.data_dir, 
            f"{self.register }_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        self.data_file = self.data_file_name + '.csv'
        
        self.data_saver = DataSaverThread(self.data_file)
        self.data_saver.start()
        
        print(f"save data to: {self.data_file}")

        self.register_lock = threading.Lock()
        keyboard_controller_thread = threading.Thread(target=self._run_keyboard_controller_thread)
        # keyboard_controller_thread.setDaemon(True)
        keyboard_controller_thread.start()
    
    def add_joint_data(self, last_action, raw_obs):
        actions = last_action.copy()  
        dof_pos = raw_obs['joint_pos'].copy()    
        dof_vel = raw_obs['joint_vel'].copy()  
        dof_torque = raw_obs['joint_torque'].copy()
        root_rot = raw_obs['root_rot'].copy()  
        root_vel = raw_obs['root_ang_vel'].copy()  
        current_time = time.time() - self.start_time
        for i in range(29):
            data_row = [current_time, i, actions[i], dof_pos[i], dof_vel[i], dof_torque[i], root_rot, root_vel]
            self.data_saver.add_data(data_row)

    def get_action(self, raw_obs):
        with self.register_lock: register = self.register
        action = self.tasks[register].get_action(raw_obs)
        return action
    
    def get_visualization(self, raw_obs, action):
        with self.register_lock: register = self.register
        visualization = self.tasks[register].get_visualization(raw_obs, action)
        return visualization

    def _run_keyboard_controller_thread(self):
        def on_press(key):
            # print("Key pressed:", key)
            try:
                for register, keyboard in self.keyboards.items():
                    if not hasattr(key, "char") or not (key.char == keyboard): continue
                    with self.register_lock: self.register = register
                    self.tasks[register].reset()
                    return
                
                if hasattr(key, "char") and (key.char == "v"): 
                    register = self.registers[(self.registers.index(self.register) + 1) % len(self.registers)]
                    with self.register_lock: self.register = register
                    self.tasks[register].reset()
                    return
                
                print("Current register:", self.register)
                self.tasks[self.register].resolve_keyboard_input(key)
            except AttributeError: print(f"Invalid key pressed: {key}")
        
        import pynput
        with pynput.keyboard.Listener(on_press=on_press) as listener:
            listener.join()
        
    @abstractmethod
    def zero_torque_state(self):
        pass

    @abstractmethod
    def move_to_default_pos(self):
        pass

    @abstractmethod
    def default_pos_state(self):
        pass

    @abstractmethod
    def run(self):
        pass

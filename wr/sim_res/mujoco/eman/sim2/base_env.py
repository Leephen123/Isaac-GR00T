# -*- coding: utf-8 -*-

import time 
import numpy as np
# from isaacgym.torch_utils import *
# import torch

import sys, os 

from abc import abstractmethod

import copy
import json
import threading
from datetime import datetime

# from tqdm import tqdm
from sim_res.mujoco.eman.base.config import Config

class BaseEnv():
    def __init__(self, config: Config, hands: str):
        self.config = config
        self.hands = hands
        self.remote_controller = None
    
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
    def get_raw_obs(self):
        pass
    
    @abstractmethod
    def step(self, target_joint):
        pass

    @abstractmethod
    def visualize(self, visualization):
        pass
    


    
    


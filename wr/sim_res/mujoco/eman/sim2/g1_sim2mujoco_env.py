# -*- coding: utf-8 -*-
import sys, os 

import time 
import numpy as np
import torch
import threading
from tqdm import tqdm

import mujoco, mujoco.viewer

from sim_res.mujoco.eman import EI_ROOT_DIR
from sim_res.mujoco.eman.base.config import Config
from sim_res.mujoco.eman.sim2.base_env import BaseEnv


class G1Sim2MujocoEnv(BaseEnv):

    JOINT_LOWER = np.array([-2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618, 
                         -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
                         -2.618, -0.52, -0.52, 
                         -3.0892, -1.5882, -2.618, -1.0472, -1.9722221, -1.6144296, -1.6144296, 
                         -3.0892, -2.2515, -2.618, -1.0472, -1.9722221, -1.6144296, -1.6144296,]).copy()
    JOINT_UPPER = np.array([2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618, 
                          2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618, 
                          2.618, 0.52, 0.52, 
                          2.6704, 2.2515, 2.618, 2.0944, 1.9722221, 1.6144296, 1.6144296, 
                          2.6704, 1.5882, 2.618, 2.0944, 1.9722221, 1.6144296, 1.6144296,])

    def __init__(self, config: Config, hands: str):
        super().__init__(config=config, hands=hands)
        
        assert self.hands in ["box", "dex3"], "Only support dex3 hands now"

        left_hand_num_dof = len(self.config.hands[self.hands]["kps"]["left"])
        right_hand_num_dof = len(self.config.hands[self.hands]["kps"]["right"])
        self.joint_concat_index = [*range(6)] + [*range(6, 12)] + [*range(12, 15)] + \
            [*range(15, 22)] + [*range(29, 29 + left_hand_num_dof)] + \
            [*range(22, 29)] + [*range(29 + left_hand_num_dof, 29 + left_hand_num_dof + right_hand_num_dof)]
        self.joint_split_index = {
            "skeleton": [*range(6)] + [*range(6, 12)] + [*range(12, 15)] + [*range(15, 22)] + [*range(22 + left_hand_num_dof, 22 + left_hand_num_dof + 7)],
            "hands": [*range(22, 22 + left_hand_num_dof)] + [*range(22 + left_hand_num_dof + 7, 22 + left_hand_num_dof + 7 + right_hand_num_dof)],
        }

        self._lock = threading.Lock()
        self._lock_read = threading.Lock()
        self._lock_write = threading.Lock()

        self.start_draw = False
        self._lock_draw = threading.Lock()

        self.write_start_pose = False
        self.start_pose = None

        step_thread = threading.Thread(target=self.step_thread)
        step_thread.daemon = True
        step_thread.start()
        time.sleep(1)
    
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

        # xml = "{EI_ROOT_DIR}/resources/robots/g1/scene_29dof.xml".format(EI_ROOT_DIR=EI_ROOT_DIR)
        xml = self.config.xml.format(EI_ROOT_DIR=EI_ROOT_DIR)
        if self.hands == "dex3":
            xml = "{EI_ROOT_DIR}/resources/robots/g1/scene_29dof_dex3.xml".format(EI_ROOT_DIR=EI_ROOT_DIR)

        model = mujoco.MjModel.from_xml_path(xml)
        model.opt.timestep = 1. / max_fps
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        viewer = mujoco.viewer.launch_passive(model, data) #mujoco_viewer.MujocoViewer(model, data)

        kp = self.resolve_compatibility("concat", state=np.array(
            self.config.kps + self.config.hands[self.hands]["kps"]["left"] + self.config.hands[self.hands]["kps"]["right"]), type=np.float32)
        kd = self.resolve_compatibility("concat", state=np.array(
            self.config.kds + self.config.hands[self.hands]["kds"]["left"] + self.config.hands[self.hands]["kds"]["right"]), type=np.float32)
        
        effort_limit = np.inf

        with self._lock:
            self.action = data.qpos.astype(np.float32).copy()[7:].copy()

        from loop_rate_limiters import RateLimiter
        rate_limiter = RateLimiter(frequency=max_fps, warn=False)
        last_render_time = time.time()

        for _ in range(50):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))

        while True:
            with self._lock_write:
                if self.write_start_pose:
                    start_pos = self.start_pose.copy()
                    data.qpos[:3] = start_pos["root_xyz"].copy()
                    data.qpos[3:7] = start_pos["root_rot"].copy()
                    data.qpos[7:] = start_pos["joint_pos"].copy()

                    data.qvel = 0.
                    data.ctrl = 0.
                    mujoco.mj_step(data.model, data)
                    # viewer.render()
                    with self._lock:
                        self.action = data.qpos.astype(np.float32).copy()[7:].copy()
                    self.write_start_pose = False

            with self._lock:
                action = self.action.copy()

            joint_pos = data.qpos.astype(np.float32).copy()[7:]
            joint_vel = data.qvel.astype(np.float32).copy()[6:]

            torque = (action - joint_pos) * kp - joint_vel * kd
            torque = np.clip(torque, -effort_limit, effort_limit)
            data.ctrl = torque.copy()
            mujoco.mj_step(data.model, data)

            rate_limiter.sleep()

            joint_pos = data.qpos.astype(np.float32).copy()[7:]
            joint_vel = data.qvel.astype(np.float32).copy()[6:]
            joint_torques = data.ctrl.astype(np.float32).copy()
            root_rot = data.qpos.astype(np.float32).copy()[3:7] #[[1, 2, 3, 0]]
            root_xyz = data.qpos.astype(np.float32).copy()[0:3]
            root_ang_vel = data.qvel.astype(np.float32).copy()[3:6]

            joint_pos, hands_joint_pos = self.resolve_compatibility("split", state=joint_pos)
            joint_vel, hands_joint_vel = self.resolve_compatibility("split", state=joint_vel)
            joint_torques, hands_joint_torques = self.resolve_compatibility("split", state=joint_torques)

            with self._lock_read:
                self.raw_obs = {"time": time.time(),
                    "joint_pos": joint_pos, "joint_vel": joint_vel, "joint_torques": joint_torques,
                    "root_rot": root_rot, "root_xyz": root_xyz, "root_ang_vel": root_ang_vel, 
                    "hands_joint_pos": hands_joint_pos, "hands_joint_vel": hands_joint_vel, "hands_joint_torques": hands_joint_torques}

            with self._lock_draw:
                if self.start_draw:
                    draw_points = self.draw_points.copy()
                    draw_points = np.array(draw_points, dtype=np.float32)
                    # draw_points = draw_points[0] # T, J, 3
                    for i in range(draw_points.shape[0]):
                        viewer.user_scn.geoms[i].pos = draw_points[i]
                    # self.start_draw = False
            
            if time.time() - last_render_time > self.config.DT:
                viewer.sync() #viewer.render()
                last_render_time = time.time()
        viewer.close()

    def zero_torque_state(self):
        return

    def move_to_default_pos(self):
        start_root_xyz = np.array([0., 0., 0.80], dtype=np.float32)
        start_root_rot = np.array([1., 0., 0., 0.], dtype=np.float32)
        target_joint_pos = np.array(self.config.default_joint_pos, dtype=np.float32).copy()
        target_joint_pos = np.concatenate([target_joint_pos, [1.]*len(self.joint_split_index["hands"])], -1)
        target_joint_pos = self.resolve_compatibility("concat", state=target_joint_pos)

        skeleton_target_joint_pos = self.resolve_compatibility("split", state=target_joint_pos)[0]
        assert (skeleton_target_joint_pos >= self.JOINT_LOWER - 1e-3).all()
        assert (skeleton_target_joint_pos <= self.JOINT_UPPER + 1e-3).all()

        for _ in tqdm(range(30)):
            with self._lock_write:
                self.write_start_pose = True
                self.start_pose = {"root_xyz": start_root_xyz, "root_rot": start_root_rot, "joint_pos": target_joint_pos}.copy()
            time.sleep(0.01)
        return

    def default_pos_state(self):
        return
    
    def get_raw_obs(self):
        with self._lock_read:
            raw_obs = self.raw_obs.copy()
        return raw_obs
    
    def step(self, step_action: dict):
        if "hands" in step_action and len(step_action["hands"]) == len(self.joint_split_index["hands"]):
            hands = step_action["hands"]
        else:
            hands = np.zeros(len(self.joint_split_index["hands"]), dtype=np.float32)
        action = np.concatenate([step_action["action"], hands], axis=-1)
        action = self.resolve_compatibility("concat", state=action)
        with self._lock:
            self.action = action.copy()
        return
    
    def visualize(self, visualization):
        if (not visualization) or (visualization.get("points", None) is None):
            return
        with self._lock_draw:
            self.start_draw = True
            self.draw_points = visualization["points"]
        return


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])

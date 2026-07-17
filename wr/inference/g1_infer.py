import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from loop_rate_limiters import RateLimiter
import os
import sys,cv2
from time import sleep

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

sys.path.append(os.getcwd())
from data_res.dds import (
    BodyPoseConfig,
    BodyPoseSubscriber,
    BodyPoseSubscriberV2_15,
    MocapConfig,
    MocapUE5G115MsgPublisher,
    _MOCAP_NUM_JOINTS,
    _MOCAP_POS_DIM,
    _MOCAP_QUAT_DIM,
)
from data_res.log import get_logger
from data_res.transforms import compute_absolute, compute_relative, normalize_quaternion, interpolate_pose7, rotation_6d_to_quaternion, restore_mocap_from_root_relative
from data_res.utils import (build_observation_from_msg, build_observation_from_msgv2_mpz, build_observation_from_msgv2_mpz_normal, get_camera_name, load_mocap_data, SELECT_11_INDICES)
from wr.data_res.camera_old import VideoCapture


gr00t_path = root_path / "serve_res/mpz/"
sys.path.append(str(gr00t_path))
from serve_res.mpz.gr00t.policy import server_client

logger = get_logger(__name__)


def append_state_history(state_history, observation):
    """
    observation["state"][key]: (1,1,D)
    state_history[key]: list[(D,)]
    """
    if not isinstance(observation, dict) or "state" not in observation:
        return

    for key, value in observation["state"].items():
        value = np.asarray(value, dtype=np.float32)

        if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != 1:
            continue

        state_history.setdefault(key, []).append(value[0, 0].copy())


def apply_state_history(
    observation,
    state_history,
    delta_indices,
    state_base_delta=2,
):
    """
    observation["state"][key]: (1,1,D)
    state_history[key]: list[(D,)]

    delta_indices:
        例如 list(range(-498, -98, 8)) + list(range(-98, 1, 2))

    state_base_delta:
        state_history 每存一帧，对应原始数据几个 step。
        当前为了匹配 range(..., 2)，应该是 2。
    """
    if not isinstance(observation, dict) or "state" not in observation:
        return observation

    new_state = {}

    for key, value in observation["state"].items():
        value = np.asarray(value, dtype=np.float32)

        if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != 1:
            new_state[key] = value
            continue

        cur = value[0, 0].copy()
        hist = state_history.get(key, [])

        if len(hist) == 0:
            hist = [cur]

        sampled = []
        last_idx = len(hist) - 1

        for delta in delta_indices:
            if delta % state_base_delta != 0:
                raise ValueError(
                    f"delta={delta} is not divisible by state_base_delta={state_base_delta}"
                )

            offset = delta // state_base_delta
            src_idx = last_idx + offset

            # 历史不够时，用最早帧 padding
            if src_idx < 0:
                src_idx = 0
            elif src_idx >= len(hist):
                src_idx = len(hist) - 1

            sampled.append(hist[src_idx])

        new_state[key] = np.stack(sampled, axis=0)[None].astype(np.float32)

    observation = dict(observation)
    observation["state"] = new_state
    return observation


def get_mixed_state_delta_indices():
    return list(range(-498, -98, 8)) + list(range(-98, 1, 2))


def trim_state_history(state_history, max_len=50):
    for key in list(state_history.keys()):
        if len(state_history[key]) > max_len:
            state_history[key] = state_history[key][-max_len:]


BODY_LIST = [
    "root", 
    "left_hip_yaw_joint", 
    "left_knee_joint", 
    "left_ankle_pitch_joint", 
    "left_ankle_roll_joint",
    "right_hip_yaw_joint", 
    "right_knee_joint", 
    "right_ankle_pitch_joint", 
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", 
    "left_elbow_joint", 
    "left_wrist_yaw_joint",
    "right_shoulder_yaw_joint", 
    "right_elbow_joint", 
    "right_wrist_yaw_joint"
]



@dataclass
class ClientConfig:
    host: str = "192.168.123.165"
    port: int = 9007
    timeout_ms: int = 15000
    task_description: str = "Clean the items on the desk and put the doll on the left sofa"
    api_token: str = None
    mocap_data_path : Path = Path("/home/unitree/wr_new/standardized_data/0518_episode_5.npy")
    infer_fps: float = 100.0
    send_fps: float = 90.0

    mocap_cfg: MocapConfig = field(
        default_factory=lambda: MocapConfig(domain_id=1, topic_name="MocapUE5G115Topic", depth=4)
    )
    body_pose_cfg: BodyPoseConfig = field(
        default_factory=lambda: BodyPoseConfig(domain_id=1, topic_name="WR/BodyPose_WR", depth=4)
    )
    camera_config: Path = Path("config/camera.yaml")

    state_history_len: int = 100          # 送给模型的长度
    state_buffer_len: int = 300           # 内部缓存长度，至少要覆盖 -498
    state_sample_every: int = 3           # 继续近似 50Hz 基础采样
    state_base_delta: int = 2             # 缓存每一帧对应原始数据 2 step


def split_xyz_wxyz(frame):
    frame = np.asarray(frame)
    xyz = frame[:, 0:3]
    wxyz = frame[:, 3:7]
    wxyz = normalize_quaternion(wxyz)
    return xyz, wxyz


class MocapDataQueue:
    def __init__(self, maxlen=300):
        self._lock = threading.Lock()
        self._queue = deque(maxlen=maxlen)
        self._last_xyz = np.zeros((_MOCAP_NUM_JOINTS, _MOCAP_POS_DIM), dtype=np.float32)
        self._last_wxyz = np.zeros((_MOCAP_NUM_JOINTS, _MOCAP_QUAT_DIM), dtype=np.float32)
        self._last_wxyz[:, 0] = 1.0

    def put(self, xyz, wxyz):
        xyz = np.asarray(xyz, dtype=np.float32).reshape(_MOCAP_NUM_JOINTS, _MOCAP_POS_DIM).copy()
        wxyz = np.asarray(wxyz, dtype=np.float32).reshape(_MOCAP_NUM_JOINTS, _MOCAP_QUAT_DIM).copy()
        with self._lock:
            self._queue.append((xyz, wxyz))
            self._last_xyz = xyz
            self._last_wxyz = wxyz

    def get_next_or_last(self):
        with self._lock:
            if len(self._queue) > 0:
                xyz, wxyz = self._queue.popleft()
                self._last_xyz = xyz
                self._last_wxyz = wxyz
                # print("mocap test")
                return xyz, wxyz
            # print("mocap last")
            return self._last_xyz, self._last_wxyz

    def clear(self, reset_last=False):
        with self._lock:
            self._queue.clear()
            if reset_last:
                self._last_xyz = np.zeros((_MOCAP_NUM_JOINTS, _MOCAP_POS_DIM), dtype=np.float32)
                self._last_wxyz = np.zeros((_MOCAP_NUM_JOINTS, _MOCAP_QUAT_DIM), dtype=np.float32)
                self._last_wxyz[:, 0] = 1.0

    def size(self):
        with self._lock:
            return len(self._queue)

    def empty(self):
        return self.size() == 0

    def is_full(self):
        with self._lock:
            return len(self._queue) >= self._queue.maxlen


class MocapSenderThread(threading.Thread):
    def __init__(self, mocap_queue: MocapDataQueue, fps: float, mocap_cfg: MocapConfig):
        super().__init__(daemon=True)
        self.mocap_queue = mocap_queue
        self.fps = fps
        self.publisher = MocapUE5G115MsgPublisher(mocap_cfg)
        self.running = True
        self.rate = RateLimiter(frequency=self.fps)

    def run(self):
        while self.running:
            xyz, wxyz = self.mocap_queue.get_next_or_last()
            self.publisher.send_msg(fps=self.fps, xyz=xyz, wxyz=wxyz)
            self.rate.sleep()

    def stop(self):
        self.running = False


if __name__ == "__main__":
    config = tyro.cli(ClientConfig)
    # body_pose = BodyPoseSubscriberV2_15(config.body_pose_cfg)
    body_pose = BodyPoseSubscriber(config.body_pose_cfg)
    mocap_queue = MocapDataQueue(maxlen=300)
    print("[INFO] init camera")
    camera_name_list = get_camera_name(config.camera_config)
    camera_caps = {name: VideoCapture(name) for name in camera_name_list}
    # camera_caps = {"front_head": cv2.VideoCapture('/home/mpz/wr_folder/0529/wr/data/ori/test1/videos/front_head.mp4')}
    sleep(2)


    client = server_client.PolicyClient(
        host=config.host,
        port=config.port,
        timeout_ms=config.timeout_ms,
        api_token=config.api_token,
    )
    print(f"waiting client.ping")

    if client.ping():
        print("Server is alive!")
    else:
        print("Failed to connect to the server.")
        sys.exit(1)
    replay_data = load_mocap_data(config.mocap_data_path)
    _MOCAP_POSE_DIM = _MOCAP_POS_DIM + _MOCAP_QUAT_DIM
    assert replay_data.ndim == 3, (
        f"Expected (T, {_MOCAP_NUM_JOINTS}, {_MOCAP_POSE_DIM}), got {replay_data.shape}"
    )
    assert replay_data.shape[1] == _MOCAP_NUM_JOINTS, (
        f"Expected {_MOCAP_NUM_JOINTS} joints, got {replay_data.shape}"
    )
    assert replay_data.shape[2] == _MOCAP_POSE_DIM, (
        f"Expected pose dim {_MOCAP_POSE_DIM}, got {replay_data.shape}"
    )
    logger.info(f"The replay episode length is: {replay_data.shape[0]}")
    
    
    # mocap_queue.put(replay_data[0, :, 0:3], replay_data[0, :, 3:7])
    # sender_thread = MocapSenderThread(
    #     mocap_queue=mocap_queue,
    #     fps=config.send_fps,
    #     mocap_cfg=config.mocap_cfg,
    # )
    # sender_thread.start()
    # logger.info("Sender thread started.")
    

    # compute the init action to relative
    while True:
        root_pose = body_pose.get_root_pose()
        if root_pose is not None:
            print(f"root pose received!")
            break
    root_pose_init = root_pose.copy()
    print(f"root_pose: {root_pose.shape}")
    # exit()

    replay_data_init = replay_data[0:1, :, :].copy()
    num_frames, num_joints, poses = replay_data_init.shape
    # print(f"replay_data_init: {replay_data_init.shape}")
    root_pose_tiled = np.tile(root_pose, (num_frames * num_joints, 1))
    replay_data_init_flat = replay_data_init.reshape(num_frames * num_joints, -1)
    # print(f"root_pose_tiled: {root_pose_tiled.shape}, replay_data_init_flat: {replay_data_init_flat.shape}")
    replay_data_init = compute_absolute(root_pose_tiled, replay_data_init_flat)
    replay_data_init = replay_data_init.reshape(num_frames, num_joints, poses)[0]

    # init the mocap queue with the first frame to avoid large jitter
    mocap_queue.put(replay_data_init[:, 0:3], replay_data_init[:, 3:7])
    sender_thread = MocapSenderThread(
        mocap_queue=mocap_queue,
        fps=config.send_fps,
        mocap_cfg=config.mocap_cfg,
    )
    sender_thread.start()
    logger.info("Sender thread started.")
    # exit()
    state_history = {}
    global_state_sample_every = 0
    state_delta_indices = get_mixed_state_delta_indices()
    assert len(state_delta_indices) == config.state_history_len
    root_rel_cum = None

    sleep(5)

    try:
        # infer_rate = RateLimiter(frequency=config.infer_fps)

        while True:
            if not mocap_queue.empty():
                # print(f"Pose queue is not empty (size={mocap_queue.size()}), waiting pytho...")
                sleep(0.01)
                continue
            while True:
                msg = body_pose.get_msg()
                if msg is not None:
                    print(f"Msg received!")
                    break
                sleep(0.01)
            
            # while True:
            #     current_15_pose7 = body_pose.get_15_pose7()[None]
            #     if current_15_pose7 is not None:
            #         print(f"current_15_pose7 received!")
            #         break
            #     sleep(0.01)
            # print(f"current_15_pose7: {current_15_pose7}")
            # # exit()
            # # current_pose_relative = compute_relative(root_pose_init.copy(), current_pose.copy())
            # num_frames, num_joints, pose_dim = current_15_pose7.shape

            # reference_tiled = np.tile(root_pose_init[0:1, :].copy(), (num_frames * num_joints, 1))
            # data_flat = current_15_pose7.copy().reshape(num_frames * num_joints, pose_dim)
            # print(f"compute_relative: root_pose_init[0:1, :]{root_pose_init[0:1, :]}")
            # relative_flat = compute_relative(reference_tiled, data_flat)
            # current_15_pose7_relative = relative_flat.reshape(num_frames, num_joints, pose_dim)
            # print(f"current_15_pose7_relative: {current_15_pose7_relative}")

            frame = {}
            for name, camera_cap in camera_caps.items():
                for _ in range(10):
                    _, frame[name] = camera_cap.read()
                
                # for _ in range(16):
                #     ret_frame, frame[name] = camera_cap.read()
                #     if not ret_frame:
                #         camera_caps = {"front_head": cv2.VideoCapture('/media/mpz/d5f7a2a2-7dfb-4053-8e51-ee6943e25306/Downloads/front_head.mp4')}
                #         ret_frame, frame[name] = camera_cap.read()
                #         break
                        
            
            

            # observation = build_observation_from_msgv2_mpz(current_15_pose7_relative, config.task_description, frame, debug=True)
            observation = build_observation_from_msgv2_mpz_normal(msg, config.task_description, frame.copy(), debug=True)
            
            
            # # 先把当前 state 加入历史
            # append_state_history(state_history, observation)
            # trim_state_history(state_history, config.state_buffer_len)

            # # 再把当前 observation 的 state 改成 (1,50,D)
            # observation = apply_state_history(
            #     observation,
            #     state_history,
            #     target_len=config.state_history_len,
            # )
            
            append_state_history(state_history, observation)
            trim_state_history(state_history, config.state_buffer_len)

            # state_delta_indices = get_mixed_state_delta_indices()

            observation = apply_state_history(
                observation,
                state_history,
                delta_indices=state_delta_indices,
                state_base_delta=config.state_base_delta,
            )
            # assert observation["state"]["body_joint"].shape[1] == config.state_history_len
            # assert observation["state"]["imu"].shape[1] == config.state_history_len
            
            
            # print(f"observation video: {observation['video']['front'].shape}")
            # print(f"observation body_joint: {observation['state']['body_joint'].shape}")
            # print(f"observation imu: {observation['state']['imu'].shape}")
            
            action = client.get_action(observation)[0]
            # print(f"action: {action['mocap'].shape}") # action: (1, 16, 15, 7)
            
            # action_concat = action['mocap'][0]
            # action_concat = np.stack(
            #     [np.asarray(action[key][0]) for key in BODY_LIST], axis=1
            # )
            # print(f"action_concat: {action_concat.shape}")
            
            joint_names = [
                "pelvis",
                "left_knee",
                "left_foot",
                "right_knee",
                "right_foot",
                "left_shoulder",
                "left_elbow",
                "left_wrist",
                "right_shoulder",
                "right_elbow",
                "right_wrist",
            ]

            pose11_list = []
            for name in joint_names:
                xyz = action[f"{name}_xyz"]              # (B, H, 3)
                rot6d = action[f"{name}_rot6d_3x2"]      # (B, H, 6)
                pose9 = np.concatenate([xyz, rot6d], axis=-1)
                pose11_list.append(pose9)

            pose11_9 = np.stack(pose11_list, axis=2)  # (B, H, 11, 9)
            
            
            print(f"using restore_mocap_from_root_relative")
            B, H, J, D = pose11_9.shape
            root_delta = action["root_delta"][0]  # (H, 3)
            print(f"root_delta: {root_delta.shape}")
            pose11_9_data = pose11_9[0].reshape((H, J*D))  # (H, 99)
            deltaXYZ_pose11_9 = np.concatenate([root_delta, pose11_9_data], axis=1)  # (H, 102)
            action_11x9, root_rel_cum = restore_mocap_from_root_relative(deltaXYZ_pose11_9, root_rel_cum)  # (H, 11, 9)
            print(f"action_11x9: {action_11x9.shape}")
            
            
            num_frames = action_11x9.shape[0]
            action_15x9 = np.zeros((num_frames, 15, 9), dtype = np.float32)
            action_15x9[..., 0:3] = 0.0
            action_15x9[..., 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
            action_15x9[:, SELECT_11_INDICES, :] = action_11x9

            print(action_15x9.shape)
            
            action_concat = rotation_6d_to_quaternion(action_15x9)
            
            
            action_concat = action_concat[3:63]
            print(f"action_concat.shape: {action_concat.shape}")
            # print(f"current_15_pose7: {current_15_pose7.shape}")
            # current_root_pose7 = current_15_pose7[0, 0:1].copy()
            current_root_pose7 = root_pose_init.copy()
            
            num_frames, num_joints, poses = action_concat.shape
            current_root_pose7_tiled = np.tile(current_root_pose7, (num_frames * num_joints, 1))
            action_concat_flat = action_concat.reshape(num_frames * num_joints, -1)
            print(f"compute_relative: current_root_pose7{current_root_pose7}")
            action_concat_flat = compute_absolute(current_root_pose7_tiled, action_concat_flat)
            action_concat_abs = action_concat_flat.reshape(num_frames, num_joints, poses)
            print(f"action_concat_abs shape: {action_concat_abs.shape}")
            action_concat_abs = interpolate_pose7(action_concat_abs, 2)
            print(f"action_concat_abs shape after interpolation: {action_concat_abs.shape}")
            # exit()
            for t in range(action_concat_abs.shape[0]):
                mocap_frame = action_concat_abs[t]
                xyz, wxyz = split_xyz_wxyz(mocap_frame)
                mocap_queue.put(xyz, wxyz)
                logger.debug(f"put mocap frame {t} to queue")

                global_state_sample_every+=1


                while True:
                    if not mocap_queue.empty():
                        sleep(0.01)
                        continue

                    # # if (t + 1) % config.state_sample_every == 0:
                    # if global_state_sample_every>=config.state_sample_every:
                    #     hist_msg = body_pose.get_msg()
                    #     if hist_msg is not None:
                    #         # hist_obs = build_observation_from_msg(
                    #         #     hist_msg,
                    #         #     config.task_description,
                    #         #     frame=None,
                    #         #     use_stickman=False,
                    #         #     backend_tag=config.backend_tag,
                    #         # )
                    #         hist_obs = build_observation_from_msgv2_mpz(hist_msg, config.task_description, frame.copy(), debug=True)
                    #         append_state_history(state_history, hist_obs)
                    #         trim_state_history(state_history, config.state_buffer_len)
                    #     global_state_sample_every = 0
                    
                    if global_state_sample_every >= config.state_sample_every:
                        hist_msg = body_pose.get_msg()
                        if hist_msg is not None:
                            hist_obs = build_observation_from_msgv2_mpz_normal(
                                hist_msg,
                                config.task_description,
                                frame.copy(),
                                debug=True,
                            )
                            append_state_history(state_history, hist_obs)
                            trim_state_history(state_history, config.state_buffer_len)

                        global_state_sample_every = 0
                    
                    break
            # sleep(3)
            # exit(0)
            logger.info("replay data per frame shape: %s", action_concat_abs[0].shape)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        sender_thread.stop()
        sender_thread.join(timeout=2.0)
        logger.info("Sender thread stopped.")

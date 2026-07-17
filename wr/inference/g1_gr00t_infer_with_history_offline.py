import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from loop_rate_limiters import RateLimiter
import os
import sys
from time import sleep

sys.path.append(os.getcwd())
from data_res.dds import (
    BodyPoseConfig,
    BodyPoseSubscriber,
    BODY_POSE_ROT6D_Q_SIZE,
    MocapConfig,
    MocapUE5G115MsgPublisher,
    MOCAP_NUM_JOINTS,
    MOCAP_POS_DIM,
    MOCAP_QUAT_DIM,
)
from data_res.log import get_logger
from data_res.transforms import (compute_absolute, normalize_quaternion, rotation_6d_to_quaternion, interpolate_pose7)
from data_res.utils import (build_observation_from_msg_with_history, get_camera_name, load_mocap_imu_data, SELECT_11_INDICES)
from wr.data_res.camera_old import VideoCapture
from serve_res.gr00t import server_client

logger = get_logger(__name__)


@dataclass
class ClientConfig:
    host: str = "192.168.123.163"
    port: int = 9002
    timeout_ms: int = 15000
    api_token: str = None
    task_description: str = "Clean the items on the desk and put the doll on the left sofa"
    replay_data_path: Path = Path("/home/unitree/wr_0527/data/episode_13/data_standard.json")
    infer_fps: float = 100.0
    send_fps: float = 100.0
    body_joint_history_fps: float = 100.0
    num_body_joint_history: int = 10
    camera_fps: float = 20.0
    use_stickman: bool = True
    use_interpolate: bool = True
    num_interp : int = 20
    mocap_cfg: MocapConfig = field(
        default_factory=lambda: MocapConfig(domain_id=1, topic_name="MocapUE5G115Topic", depth=4)
    )
    body_pose_cfg: BodyPoseConfig = field(
        default_factory=lambda: BodyPoseConfig(domain_id=1, topic_name="WR/BodyPose_WR", depth=4)
    )
    camera_config: Path = Path("config/camera.yaml")


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
        self._last_xyz = np.zeros((MOCAP_NUM_JOINTS, MOCAP_POS_DIM), dtype=np.float32)
        self._last_wxyz = np.zeros((MOCAP_NUM_JOINTS, MOCAP_QUAT_DIM), dtype=np.float32)
        self._last_wxyz[:, 0] = 1.0

    def put(self, xyz, wxyz):
        xyz = np.asarray(xyz, dtype=np.float32).reshape(MOCAP_NUM_JOINTS, MOCAP_POS_DIM).copy()
        wxyz = np.asarray(wxyz, dtype=np.float32).reshape(MOCAP_NUM_JOINTS, MOCAP_QUAT_DIM).copy()
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
                return xyz, wxyz
            return self._last_xyz, self._last_wxyz

    def clear(self, reset_last=False):
        with self._lock:
            self._queue.clear()
            if reset_last:
                self._last_xyz = np.zeros((MOCAP_NUM_JOINTS, MOCAP_POS_DIM), dtype=np.float32)
                self._last_wxyz = np.zeros((MOCAP_NUM_JOINTS, MOCAP_QUAT_DIM), dtype=np.float32)
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
    body_pose = BodyPoseSubscriber(
        config.body_pose_cfg,
        history_fps=config.body_joint_history_fps,
        history_maxlen=config.num_body_joint_history,
        auto_start_history=True,
    )
    mocap_queue = MocapDataQueue(maxlen=300)
    print("[INFO] init camera")
    camera_name_list = get_camera_name(config.camera_config)
    camera_caps = {name: VideoCapture(name) for name in camera_name_list}
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

    # compute the init action to relative
    replay_data, _ = load_mocap_imu_data(config.replay_data_path)
    while True:
        root_pose = body_pose.get_root_pose()
        if root_pose is not None:
            print(f"root pose received!")
            break

    replay_data_init = replay_data[0:1, :, :].copy()
    num_frames, num_joints, poses = replay_data_init.shape
    root_pose_tiled = np.tile(root_pose, (num_frames * num_joints, 1))
    replay_data_init_flat = replay_data_init.reshape(num_frames * num_joints, -1)
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
    last_action = None

    try:
        while True:
            init_pose = body_pose.get_root_pose()
            if init_pose is not None:
                print(f"root pose received!")
                break
            sleep(0.01)

        while True:
            if not mocap_queue.empty():
                sleep(0.01)
                continue

            while True:
                history_body_joint = body_pose.get_history_body_joint(
                    num_history=config.num_body_joint_history
                )
                if history_body_joint.shape[0] > 0:
                    print(f"Msg received!")
                    break
                sleep(0.01)

            frame = {}
            for name, camera_cap in camera_caps.items():
                _, frame[name] = camera_cap.read()

            observation = build_observation_from_msg_with_history(history_body_joint, config.task_description, frame, config.use_stickman)
            
            action_11x9 = client.get_action(observation)[0]['mocap'][0]
            action_11x9 =  action_11x9.reshape(-1, 99)
            num_frames, _ = action_11x9.shape

            action_15x9 = np.zeros((num_frames, 15, 9), dtype = np.float32)
            action_15x9[..., 0:3] = 0.0
            action_15x9[..., 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
            action_15x9[:, SELECT_11_INDICES, :] = action_11x9
            action_15x7 = rotation_6d_to_quaternion(action_15x9)
            num_frames, num_joints, num_poses = action_15x7.shape
            action_15x7_flat = action_15x7.reshape(num_frames * num_joints, num_poses)
            init_pose_x7_flat = np.tile(init_pose.copy(), (num_frames * num_joints, 1))
            abs_action = compute_absolute(init_pose_x7_flat, action_15x7_flat).reshape(num_frames, num_joints, num_poses)
            if last_action is not None:
                abs_action = np.concatenate((last_action, abs_action), axis=0)
            if config.use_interpolate:
                abs_action = interpolate_pose7(abs_action, config.num_interp)
            
            for t in range(abs_action.shape[0]):
                mocap_frame = abs_action[t]
                xyz, wxyz = split_xyz_wxyz(mocap_frame)
                mocap_queue.put(xyz, wxyz)
                logger.debug(f"put mocap frame {t} to queue")
            last_action = abs_action[-1:]
            logger.info("replay data per frame shape: %s", abs_action[0].shape)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        body_pose.stop_collecting_body_joint()
        sender_thread.stop()
        sender_thread.join(timeout=2.0)
        logger.info("Sender thread stopped.")

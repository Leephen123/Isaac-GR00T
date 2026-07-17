import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from loop_rate_limiters import RateLimiter
import requests
import cv2
import json
import os
import sys
from time import sleep

sys.path.append(os.getcwd())
from data_res.dds import (
    BodyPoseConfig,
    BodyPoseSubscriber,
    MocapConfig,
    MocapUE5G115MsgPublisher,
    MOCAP_NUM_JOINTS,
    MOCAP_POS_DIM,
    MOCAP_QUAT_DIM,
)
from data_res.log import get_logger
from data_res.transforms import compute_absolute, normalize_quaternion
from data_res.utils import (build_observation_from_msg, get_camera_name)
from wr.data_res.camera_old import VideoCapture

logger = get_logger(__name__)


@dataclass
class ClientConfig:
    host: str = "172.16.78.10"
    port: int = 36367
    timeout_ms: int = 15000
    task_description: str = ""
    api_token: str = None

    infer_fps: float = 100.0
    send_fps: float = 100.0
    camera_fps: float = 20.0
    mocap_cfg: MocapConfig = field(
        default_factory=lambda: MocapConfig(domain_id=1, topic_name="MocapUE5G115Topic", depth=4)
    )
    body_pose_cfg: BodyPoseConfig = field(
        default_factory=lambda: BodyPoseConfig(domain_id=1, topic_name="WR/BodyPose", depth=4)
    )
    camera_config: Path = Path("config/camera.yaml")


def split_xyz_wxyz(frame):
    frame = np.asarray(frame)
    xyz = frame[:, 0:3]
    wxyz = frame[:, 3:7]
    wxyz = normalize_quaternion(wxyz)
    return xyz, wxyz


def get_action_via_http_files(
    msg: dict,
    task_description: str,
    frame: dict,
    host: str,
    port: int,
    timeout_ms: int = 15000,
    action_dim_1: int = 15,
    action_dim_2: int = 7,
    action_dtype=np.float64,
):

    url = f"http://{host}:{port}/predict"

    xyz = msg.get("robot_xpos", None)
    wxyz = msg.get("robot_xquat", None)
    state = np.concatenate([xyz, wxyz], axis=-1).astype(np.float32)  # 15 * 7

    front_head = frame.get("front", None)
    # 没有腕部相机时给空图占位
    left_hand = np.zeros((240, 320, 3), dtype=np.uint8)
    right_hand = np.zeros((240, 320, 3), dtype=np.uint8)

    # msg 15 * 7
    files = {
        "json": json.dumps({"instruction": task_description}),
        "front_head": ("front_head", front_head.tobytes(), "application/octet-stream"),
        "left_hand": ("left_hand", left_hand.tobytes(), "application/octet-stream"),
        "right_hand": ("right_hand", right_hand.tobytes(), "application/octet-stream"),
        "state": ("state", state.tobytes(), "application/octet-stream"),
    }

    resp = requests.post(url, files=files, timeout=timeout_ms / 1000.0)
    resp.raise_for_status()

    # 兼容 JSON 返回
    content_type = resp.headers.get("Content-Type", "")
    if "application/json" in content_type:
        result = resp.json()
        if "action" not in result:
            raise ValueError(f"Server response missing 'action': {result}")
        return np.asarray(result["action"], dtype=np.float64)

    # 兼容 bytes 返回
    action_flat = np.frombuffer(resp.content, dtype=action_dtype)
    if action_flat.size == 0:
        raise ValueError("Server returned empty action bytes")
    if action_flat.size % (action_dim_1 * action_dim_2) != 0:
        raise ValueError(
            f"Invalid action bytes length: {action_flat.size}, action_dim_1={action_dim_1}, action_dim_2={action_dim_2}"
        )
    # action chunk 50 * 15 * 7
    return action_flat.reshape(-1, action_dim_1, action_dim_2).astype(np.float32)


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
    body_pose = BodyPoseSubscriber(config.body_pose_cfg)
    mocap_queue = MocapDataQueue(maxlen=300)
    print("[INFO] init camera")
    camera_name_list = get_camera_name(config.camera_config)
    camera_caps = {name: VideoCapture(name) for name in camera_name_list}
    sleep(2)

    sender_thread = MocapSenderThread(
        mocap_queue=mocap_queue,
        fps=config.send_fps,
        mocap_cfg=config.mocap_cfg,
    )
    sender_thread.start()
    logger.info("Sender thread started.")

    try:
        infer_rate = RateLimiter(frequency=config.infer_fps)

        while True:
            init_pose = body_pose.get_root_pose()
            if init_pose is not None:
                print(f"root pose received!")
                break
            sleep(0.01)

        while True:
            while True:
                msg = body_pose.get_msg()
                if msg is not None:
                    print(f"Msg received!")
                    break
                sleep(0.01)

            frame = {}
            for name, camera_cap in camera_caps.items():
                frame[name] = camera_cap.read()

            # step = {
            #     "body_joint": np.asarray(msg.q, dtype=np.float32),
            #     "imu": np.asarray(msg.wxyz, dtype=np.float32),
            #     "front": frame.get("front", None),
            # }
            action = get_action_via_http_files(
                msg=msg,
                task_description=config.task_description,
                frame=frame,
                host=config.host,
                port=config.port,
                timeout_ms=config.timeout_ms,
            )
            # -------------------------------------------------------------------------
            # # 兼容 server 返回的不同维度，尽量保持与原来 client.get_action(...)[0] 一致的语义
            # if isinstance(action_raw, np.ndarray) and action_raw.ndim == 3 and action_raw.shape[0] >= 1:
            #     action = action_raw[0]
            # else:
            #     action = action_raw
            abs_action = compute_absolute(init_pose, action)
            # -------------------------------------------------------------------------

            for t in range(abs_action.shape[0]):
                mocap_frame = abs_action[t]
                xyz, wxyz = split_xyz_wxyz(mocap_frame)
                mocap_queue.put(xyz, wxyz)
                logger.debug(f"put mocap frame {t} to queue")
                infer_rate.sleep()
            logger.info("replay data per frame shape: %s", abs_action[0].shape)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        sender_thread.stop()
        sender_thread.join(timeout=2.0)
        logger.info("Sender thread stopped.")

import os, sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro, time
from loop_rate_limiters import RateLimiter

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
from data_res.transforms import compute_absolute, interpolate_pose7, normalize_quaternion
from data_res.utils import load_mocap_data

logger = get_logger(__name__)


@dataclass
class ClientConfig:
    replay_data_path: Path = Path(
        "/media/mpz/d5f7a2a2-7dfb-4053-8e51-ee6943e25306/wr/wr1/standalized_data/right_left_clip.npy"
    )
    send_fps: float = 100.0
    use_interpolate: bool = False
    use_relative: bool = True

    mocap_cfg: MocapConfig = field(
        default_factory=lambda: MocapConfig(domain_id=1, topic_name="MocapUE5G115Topicvla", depth=4)
    )
    body_pose_cfg: BodyPoseConfig = field(
        default_factory=lambda: BodyPoseConfig(domain_id=1, topic_name="WR/BodyPose_mpz", depth=4)
    )


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
    body_pose = BodyPoseSubscriber(config.body_pose_cfg)
    mocap_queue = MocapDataQueue(maxlen=300)

    # assert config.send_fps > config.infer_fps, (
    #     f"send_fps ({config.send_fps}) must be greater than infer_fps ({config.infer_fps})"
    # )

    replay_data = load_mocap_data(config.replay_data_path)
    mocap_dim = MOCAP_POS_DIM + MOCAP_QUAT_DIM
    assert replay_data.ndim == 3, (
        f"Expected (T, {MOCAP_NUM_JOINTS}, {mocap_dim}), got {replay_data.shape}"
    )
    assert replay_data.shape[1] == MOCAP_NUM_JOINTS, (
        f"Expected {MOCAP_NUM_JOINTS} joints, got {replay_data.shape}"
    )
    assert replay_data.shape[2] == mocap_dim, (
        f"Expected pose dim {mocap_dim}, got {replay_data.shape}"
    )
    logger.info(f"The replay episode length is: {replay_data.shape[0]}")

    if config.use_interpolate:
        replay_data = interpolate_pose7(replay_data, num_interp=12)
        logger.info(f"The replay episode length after interp is: {replay_data.shape[0]}")

    # init the mocap queue with the first frame to avoid large jitter
    mocap_queue.put(replay_data[0, :, 0:3], replay_data[0, :, 3:7])
    sender_thread = MocapSenderThread(
        mocap_queue=mocap_queue,
        fps=config.send_fps,
        mocap_cfg=config.mocap_cfg,
    )
    sender_thread.start()
    logger.info("Sender thread started.")

    try:
        # infer_rate = RateLimiter(frequency=config.infer_fps)
        while True:
            if not mocap_queue.empty():
                time.sleep(0.01)
                continue

            if config.use_relative:
                get_root_pose_time = time.time()
                while True:
                    root_pose = body_pose.get_root_pose()
                    if root_pose is not None:
                        break
                    if time.time() - get_root_pose_time > 10.0:
                        raise ValueError("No G1 root pose received after 5 seconds")
                if root_pose is None:
                    raise ValueError("No G1 root pose received")
                replay_data_cp = replay_data.copy()
                num_frames, num_joints, poses = replay_data_cp.shape
                root_pose_tiled = np.tile(root_pose, (num_frames * num_joints, 1))
                replay_data_flat = replay_data_cp.reshape(num_frames * num_joints, -1)
                replay_data_cp = compute_absolute(root_pose_tiled, replay_data_flat)
                replay_data_cp = replay_data_cp.reshape(num_frames, num_joints, poses)

            for t in range(replay_data_cp.shape[0]):
                mocap_frame = replay_data_cp[t]
                xyz, wxyz = split_xyz_wxyz(mocap_frame)

                while mocap_queue.is_full():
                    time.sleep(0.001)

                mocap_queue.put(xyz, wxyz)
                logger.debug(f"put mocap frame {t} to queue")
                # infer_rate.sleep()

            logger.info("replay data per frame shape: %s", replay_data_cp[0].shape)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        sender_thread.stop()
        sender_thread.join(timeout=2.0)
        logger.info("Sender thread stopped.")

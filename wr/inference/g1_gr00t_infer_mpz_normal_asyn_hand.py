import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep
from typing import Any, Callable

import numpy as np
import tyro,json
from loop_rate_limiters import RateLimiter
import os
import sys, cv2

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
    MocapUEHandSubscriber,
    MocapUEHandPublisher,
)
from data_res.log import get_logger
from data_res.transforms import (
    compute_absolute,
    compute_relative,
    normalize_quaternion,
    interpolate_pose7,
    rotation_6d_to_quaternion,
    restore_mocap_from_root_relative,
    standardize_imu,
)
from data_res.utils import (
    build_observation_from_msg,
    build_observation_from_msgv2_mpz,
    build_observation_from_msgv2_mpz_normal,
    build_observation_from_msgv2_mpz_normal_hand,
    get_camera_name,
    load_mocap_data,
    SELECT_11_INDICES,
)
from data_res.camera import VideoCapture


gr00t_path = root_path / "serve_res/mpz/"
sys.path.append(str(gr00t_path))
from serve_res.mpz.gr00t.policy import server_client

logger = get_logger(__name__)


JOINT_NAMES_11 = [
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


def append_state_history(state_history, observation):
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
    "right_wrist_yaw_joint",
]


@dataclass
class ClientConfig:
    host: str = "192.168.123.165"
    port: int = 9007
    timeout_ms: int = 15000
    task_description: str = "Clean the items on the desk and put the doll on the left sofa"
    api_token: str = None
    mocap_data_path: Path = Path("data/housework_0.npz") #/home/unitree/wr/standalized_data/housework_0.npy
    infer_fps: float = 100.0
    send_fps: float = 30 # 70.0

    mocap_cfg: MocapConfig = field(
        default_factory=lambda: MocapConfig(domain_id=1, topic_name="MocapUE5G115Topic", depth=4)
    )
    body_pose_cfg: BodyPoseConfig = field(
        default_factory=lambda: BodyPoseConfig(domain_id=1, topic_name="WR/BodyPose", depth=4)
    )
    camera_config: Path = Path("config/camera.yaml")

    state_history_len: int = 100
    state_buffer_len: int = 300
    state_sample_every: int = 3
    state_base_delta: int = 2

    raw_action_start: int = 10
    raw_action_end: int = 63 #63
    interp_num: int = 2
    index_root_rel_cum: int = -4

    # 新增：chunk 之间 bridge interpolation
    enable_chunk_bridge_interp: bool = True #True
    chunk_bridge_interp_num: int = 5 # 10

    async_start_ratio: float = 0.98
    camera_flush_infer: int = 5
    fallback_sync_on_async_error: bool = True


def split_xyz_wxyz(frame):
    frame = np.asarray(frame)
    xyz = frame[:, 0:3]
    wxyz = frame[:, 3:7]
    wxyz = normalize_quaternion(wxyz)
    return xyz, wxyz


def bridge_action_chunks(
    prev_last_pose7: np.ndarray | None,
    cur_chunk_pose7: np.ndarray,
    enable: bool = True,
    num_interp: int = 3,
) -> np.ndarray:
    """
    在上一段最后一帧和当前段第一帧之间插入过渡帧。

    prev_last_pose7:
        (15,7) 上一个 chunk 最后一帧 absolute pose7

    cur_chunk_pose7:
        (T,15,7) 当前 chunk absolute pose7

    返回:
        (T + num_interp, 15, 7) 或原始 cur_chunk_pose7
    """
    cur_chunk_pose7 = np.asarray(cur_chunk_pose7, dtype=np.float32)

    if not enable:
        return cur_chunk_pose7

    if prev_last_pose7 is None:
        return cur_chunk_pose7

    if num_interp <= 0:
        return cur_chunk_pose7

    if cur_chunk_pose7.ndim != 3 or cur_chunk_pose7.shape[1:] != (15, 7):
        raise ValueError(f"Expected cur_chunk_pose7 shape (T,15,7), got {cur_chunk_pose7.shape}")

    prev_last_pose7 = np.asarray(prev_last_pose7, dtype=np.float32)

    if prev_last_pose7.shape != (15, 7):
        raise ValueError(f"Expected prev_last_pose7 shape (15,7), got {prev_last_pose7.shape}")

    if cur_chunk_pose7.shape[0] == 0:
        return cur_chunk_pose7

    # 只对 prev_last -> cur_first 做插值，不改变当前 chunk 内部点间隔
    bridge_pair = np.stack(
        [
            prev_last_pose7,
            cur_chunk_pose7[0],
        ],
        axis=0,
    )  # (2,15,7)

    bridge = interpolate_pose7(bridge_pair, num_interp=num_interp)
    # bridge: [prev_last, inserted..., cur_first]
    # 去掉 prev_last 和 cur_first，只插入中间帧
    inserted = bridge[1:-1]

    if inserted.shape[0] == 0:
        return cur_chunk_pose7

    out = np.concatenate([inserted, cur_chunk_pose7], axis=0).astype(np.float32)

    print(
        f"[CHUNK BRIDGE] enabled | "
        f"inserted={inserted.shape[0]} | "
        f"before={cur_chunk_pose7.shape[0]} | "
        f"after={out.shape[0]}"
    )

    return out


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
                return xyz, wxyz
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


class AsyncInferenceWorker:
    def __init__(self, infer_fn: Callable[..., Any], config: ClientConfig):
        self._infer_fn = infer_fn
        self._config = config
        self._lock = threading.Lock()
        self._busy = False
        self._done = threading.Event()
        self._result = None
        self._error = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def submit(self, **kwargs: Any) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self._done.clear()
            self._result = None
            self._error = None

        threading.Thread(target=self._run, kwargs=kwargs, daemon=True).start()
        return True

    def _make_async_client(self):
        return server_client.PolicyClient(
            host=self._config.host,
            port=self._config.port,
            timeout_ms=self._config.timeout_ms,
            api_token=self._config.api_token,
        )

    def _run(self, **kwargs: Any) -> None:
        try:
            async_client = self._make_async_client()
            result = self._infer_fn(policy_client=async_client, **kwargs)
            with self._lock:
                self._result = result
        except BaseException as exc:
            with self._lock:
                self._error = exc
        finally:
            with self._lock:
                self._busy = False
            self._done.set()

    def poll_done(self) -> bool:
        with self._lock:
            return (not self._busy) and self._done.is_set() and (
                self._result is not None or self._error is not None
            )

    def get_result(self):
        with self._lock:
            error = self._error
            result = self._result
            self._done.clear()
            self._result = None
            self._error = None

        if error is not None:
            raise error
        if result is None:
            raise RuntimeError("inference finished without result")
        return result


def read_camera_frames(camera_caps, flush_count=70):
    frame = {}
    for name, camera_cap in camera_caps.items():
        last_img = None
        ok = False
        for _ in range(flush_count):
            ok, last_img = camera_cap.read()
        if not ok or last_img is None:
            raise RuntimeError(f"Failed to read camera frame from {name}")
        frame[name] = last_img
    return frame


def action_to_abs_pose7(action, root_rel_cum, root_pose_init, config: ClientConfig):
    pose11_list = []
    for name in JOINT_NAMES_11:
        xyz = action[f"{name}_xyz"]
        rot6d = action[f"{name}_rot6d_3x2"]
        pose9 = np.concatenate([xyz, rot6d], axis=-1)
        pose11_list.append(pose9)

    pose11_9 = np.stack(pose11_list, axis=2)

    print("using restore_mocap_from_root_relative")
    B, H, J, D = pose11_9.shape
    root_delta = action["root_delta"][0]
    print(f"root_delta: {root_delta.shape}")

    pose11_9_data = pose11_9[0].reshape((H, J * D))
    deltaXYZ_pose11_9 = np.concatenate([root_delta, pose11_9_data], axis=1)

    action_11x9, next_root_rel_cum = restore_mocap_from_root_relative(
        deltaXYZ_pose11_9,
        root_rel_cum,
        config.index_root_rel_cum,
    )
    print(f"action_11x9: {action_11x9.shape}")

    num_frames = action_11x9.shape[0]
    action_15x9 = np.zeros((num_frames, 15, 9), dtype=np.float32)
    action_15x9[..., 0:3] = 0.0
    action_15x9[..., 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    action_15x9[:, SELECT_11_INDICES, :] = action_11x9

    action_concat = rotation_6d_to_quaternion(action_15x9)

    action_concat = action_concat[config.raw_action_start:config.raw_action_end]
    print(f"action_concat.shape: {action_concat.shape}")

    current_root_pose7 = root_pose_init.copy()
    num_frames, num_joints, poses = action_concat.shape
    current_root_pose7_tiled = np.tile(current_root_pose7, (num_frames * num_joints, 1))
    action_concat_flat = action_concat.reshape(num_frames * num_joints, -1)

    print(f"compute_absolute: current_root_pose7{current_root_pose7}")
    action_concat_flat = compute_absolute(current_root_pose7_tiled, action_concat_flat)
    action_concat_abs = action_concat_flat.reshape(num_frames, num_joints, poses)
    print(f"action_concat_abs shape: {action_concat_abs.shape}")

    action_concat_abs = interpolate_pose7(action_concat_abs, config.interp_num)
    print(f"action_concat_abs shape after interpolation: {action_concat_abs.shape}")
    
    hand_joint = action["hand_joint"][0]

    print(f"action_concat_abs: {action_concat_abs.shape}, hand_joint: {hand_joint.shape}")

    return action_concat_abs, next_root_rel_cum, hand_joint


def build_observation_with_history(
    msg,
    frame,
    config: ClientConfig,
    state_history,
    state_history_lock,
    state_delta_indices,
    debug=False,
    msg_hand=None,
):
    observation = build_observation_from_msgv2_mpz_normal_hand(
        msg,
        config.task_description,
        frame.copy(),
        debug=debug,
        msg_hand=msg_hand,
    )

    with state_history_lock:
        append_state_history(state_history, observation)
        trim_state_history(state_history, config.state_buffer_len)
        observation = apply_state_history(
            observation,
            state_history,
            delta_indices=state_delta_indices,
            state_base_delta=config.state_base_delta,
        )
    return observation


def _waiting_robot_move_done(mocap_queue):
    while True:
        if not mocap_queue.empty():
            sleep(0.01)
            continue
        return True


def main():
    config = tyro.cli(ClientConfig)

    body_pose = BodyPoseSubscriberV2_15(config.body_pose_cfg)
    mocap_queue = MocapDataQueue(maxlen=300)
    
    hand_subscriber = MocapUEHandSubscriber()
    hand_publisher = MocapUEHandPublisher()
    
    

    camera_name_list = get_camera_name(config.camera_config)
    camera_caps = {name: VideoCapture(name) for name in camera_name_list}
    sleep(2)

    main_client = server_client.PolicyClient(
        host=config.host,
        port=config.port,
        timeout_ms=config.timeout_ms,
        api_token=config.api_token,
    )
    print("waiting client.ping")

    if main_client.ping():
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

    while True:
        root_pose = body_pose.get_root_pose()
        if root_pose is not None:
            print("root pose received!")
            break
        sleep(0.01)


    # 关键：compute_absolute 只会用 reference_pose 的 yaw 来旋转 action。
    # 所以这里把 root_pose_init 的 yaw 去掉，避免 action 被当前 G1 yaw 旋偏。
    # print(f"root_pose raw: {root_pose}")
    # root_pose[0:3] = 0.0
    # root_pose[3:7] = standardize_imu(root_pose[3:7][None, :])[0].astype(np.float32)
    # print(f"root_pose_init yaw_removed: {root_pose}")

    root_pose_init = root_pose.copy()
    print(f"root_pose: {root_pose.shape}")


    # replay_data_init = replay_data[0:1, :, :].copy()
    # num_frames, num_joints, poses = replay_data_init.shape
    # root_pose_tiled = np.tile(root_pose, (num_frames * num_joints, 1))
    # replay_data_init_flat = replay_data_init.reshape(num_frames * num_joints, -1)
    # replay_data_init = compute_absolute(root_pose_tiled, replay_data_init_flat)
    # replay_data_init = replay_data_init.reshape(num_frames, num_joints, poses)[0]


    replay_data_init = replay_data[0, :, :].copy()

    print(f"replay_data_init: {replay_data_init.shape}")


    mocap_queue.put(replay_data_init[:, 0:3], replay_data_init[:, 3:7])
    sender_thread = MocapSenderThread(
        mocap_queue=mocap_queue,
        fps=config.send_fps,
        mocap_cfg=config.mocap_cfg,
    )
    sender_thread.start()
    logger.info("Sender thread started.")

    sleep(5)
    # exit(0)

    state_history = {}
    state_history_lock = threading.Lock()
    global_state_sample_every = 0
    state_delta_indices = get_mixed_state_delta_indices()
    assert len(state_delta_indices) == config.state_history_len
    root_rel_cum = None
    last_chunk_last_pose7 = None

    def infer_once(policy_client, msg, frame, root_rel_cum_snapshot, debug, msg_hand):
        observation = build_observation_with_history(
            msg=msg,
            frame=frame,
            config=config,
            state_history=state_history,
            state_history_lock=state_history_lock,
            state_delta_indices=state_delta_indices,
            debug=debug,
            msg_hand=msg_hand,
        )
        action = policy_client.get_action(observation)[0]       
        
        return action_to_abs_pose7(
            action=action,
            root_rel_cum=root_rel_cum_snapshot,
            root_pose_init=root_pose_init,
            config=config,
        )

    inference_worker = AsyncInferenceWorker(infer_once, config)

    try:
        prefetched_result = None

        while True:
            # if not mocap_queue.empty():
            #     sleep(0.01)
            #     continue
            
            _waiting_robot_move_done(mocap_queue)

            if prefetched_result is not None:
                action_concat_abs, root_rel_cum = prefetched_result
                prefetched_result = None
                logger.info("use prefetched async action chunk")
            else:
                if inference_worker.busy:
                    logger.info("waiting async inference result because queue is empty")
                    while inference_worker.busy:
                        sleep(0.001)

                if inference_worker.poll_done():
                    try:
                        action_concat_abs, root_rel_cum = inference_worker.get_result()
                        logger.info("use late async action chunk")
                    except Exception as exc:
                        logger.error("async inference failed: %s", exc, exc_info=True)
                        if not config.fallback_sync_on_async_error:
                            raise
                        action_concat_abs = None
                else:
                    action_concat_abs = None

                if action_concat_abs is None:
                    while True:
                        msg = body_pose.get_msg()
                        if msg is not None:
                            print("Msg received!")
                            break
                        sleep(0.01)
                    
                    msg_hand = hand_subscriber.get_state()

                    frame = read_camera_frames(camera_caps, flush_count=config.camera_flush_infer)
                    action_concat_abs, root_rel_cum, hand_joint = infer_once(
                        policy_client=main_client,
                        msg=msg,
                        frame=frame,
                        root_rel_cum_snapshot=root_rel_cum,
                        debug=True,
                        msg_hand=msg_hand,
                    )
                    logger.info("use sync action chunk")

            # 新增：chunk bridge interpolation
            action_concat_abs = bridge_action_chunks(
                prev_last_pose7=last_chunk_last_pose7,
                cur_chunk_pose7=action_concat_abs,
                enable=config.enable_chunk_bridge_interp,
                num_interp=config.chunk_bridge_interp_num,
            )

            async_submitted_for_this_chunk = False
            async_start_t = max(
                0,
                min(
                    action_concat_abs.shape[0] - 1,
                    int(action_concat_abs.shape[0] * config.async_start_ratio),
                ),
            )

            for t in range(action_concat_abs.shape[0]):

                mocap_frame = action_concat_abs[t]
                xyz, wxyz = split_xyz_wxyz(mocap_frame)
                mocap_queue.put(xyz, wxyz)
                logger.debug(f"put mocap frame {t} to queue")
                # hand_joint
                hand_publisher.send(hand_joint[t])

                global_state_sample_every += 1

                if (not async_submitted_for_this_chunk) and t >= async_start_t:
                    try:
                        # waiting for the previous actions to finish
                        _waiting_robot_move_done(mocap_queue)
                        
                        async_msg = body_pose.get_msg()
                        if async_msg is not None:
                            msg_hand = hand_subscriber.get_state()
                            async_frame = read_camera_frames(
                                camera_caps,
                                flush_count=config.camera_flush_infer,
                            )
                            
                            ok = inference_worker.submit(
                                msg=async_msg,
                                frame=async_frame,
                                root_rel_cum_snapshot=root_rel_cum,
                                debug=True,
                                msg_hand=msg_hand,
                            )
                            if ok:
                                async_submitted_for_this_chunk = True
                                logger.info(
                                    "async inference submitted at t=%d/%d, queue=%d",
                                    t,
                                    action_concat_abs.shape[0],
                                    mocap_queue.size(),
                                )
                    except Exception as exc:
                        logger.warning("submit async inference failed: %s", exc)
                        async_submitted_for_this_chunk = True

                while True:

                    if global_state_sample_every >= config.state_sample_every:
                        _waiting_robot_move_done(mocap_queue)
                        hist_msg = body_pose.get_msg()
                        if hist_msg is not None:
                            msg_hand = hand_subscriber.get_state()
                            hist_obs = build_observation_from_msgv2_mpz_normal_hand(
                                hist_msg,
                                config.task_description,
                                frame.copy() if "frame" in locals() else {},
                                debug=False,
                                msg_hand=msg_hand,
                            )
                            with state_history_lock:
                                append_state_history(state_history, hist_obs)
                                trim_state_history(state_history, config.state_buffer_len)

                        global_state_sample_every = 0

                    break

                if inference_worker.poll_done() and prefetched_result is None:
                    try:
                        prefetched_result = inference_worker.get_result()
                        logger.info("async inference finished and cached")
                    except Exception as exc:
                        logger.error("async inference failed: %s", exc, exc_info=True)
                        prefetched_result = None

            

            if action_concat_abs.shape[0] > 0:
                last_chunk_last_pose7 = action_concat_abs[-1].copy()

            logger.info("replay data per frame shape: %s", action_concat_abs[0].shape)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        sender_thread.stop()
        sender_thread.join(timeout=2.0)
        logger.info("Sender thread stopped.")


if __name__ == "__main__":
    main()
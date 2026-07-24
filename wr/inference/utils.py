import threading
from collections import deque
from time import sleep
from typing import Any, Callable

import numpy as np
from loop_rate_limiters import RateLimiter

from data_res.camera import VideoCapture
from data_res.dds import (
    BodyPoseSubscriber,
    MocapConfig,
    MocapUE5G115MsgPublisher,
    MOCAP_NUM_JOINTS,
    MOCAP_POS_DIM,
    MOCAP_QUAT_DIM,
    WR_GAE_BodyPose_Msg,
)
from data_res.log import get_logger
from data_res.transforms import (
    compute_absolute,
    compute_imu_relative,
    normalize_quaternion,
    quaternion_to_rotation_6d,
    rotation_6d_to_quaternion,
    restore_mocap_from_root_relative,
)
from data_res.utils import SELECT_11_INDICES

logger = get_logger(__name__)

DEFAULT_JOINT_ROT6D = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
NUM_MOCAP_JOINTS = 15
SIM_INIT_POSE = np.array([
            [
                -1.7715167999267578,
                -1.3067988157272339,
                0.7911615967750549,
                0.9507724046707153,
                0.01602913625538349,
                0.015250850468873978,
                -0.3090991973876953
            ],
            [
                -1.7058221101760864,
                -1.207708477973938,
                0.6605340838432312,
                0.9485098719596863,
                -0.006524891126900911,
                -0.06370166689157486,
                -0.31020721793174744
            ],
            [
                -1.668833613395691,
                -1.221899151802063,
                0.357001394033432,
                0.9324308037757874,
                0.07568785548210144,
                0.11128878593444824,
                -0.3353489935398102
            ],
            [
                -1.7186251878738403,
                -1.1533647775650024,
                0.05090729892253876,
                0.9683676362037659,
                0.0032607174944132566,
                0.0033355539198964834,
                -0.24948415160179138
            ],
            [
                -1.6310758590698242,
                -1.2016810178756714,
                0.050098590552806854,
                0.9683676362037659,
                0.0032607174944132566,
                0.0033355539198964834,
                -0.24948415160179138
            ],
            [
                -1.8422971963882446,
                -1.395257592201233,
                0.6556500196456909,
                0.9729068279266357,
                -0.013440640643239021,
                -0.06626231223344803,
                -0.22109051048755646
            ],
            [
                -1.8055568933486938,
                -1.414125680923462,
                0.3523419499397278,
                0.9764501452445984,
                0.06982644647359848,
                0.14209787547588348,
                -0.14655248820781708
            ],
            [
                -1.887191653251648,
                -1.357588529586792,
                0.050664860755205154,
                0.9583539962768555,
                -0.06671961396932602,
                -0.02429119683802128,
                -0.27661529183387756
            ],
            [
                -1.8026129007339478,
                -1.4102834463119507,
                0.05901190638542175,
                0.9583539962768555,
                -0.06671961396932602,
                -0.02429119683802128,
                -0.27661529183387756
            ],
            [
                -1.6882505416870117,
                -1.2097095251083374,
                1.0712023973464966,
                0.9440533518791199,
                0.08883167058229446,
                0.013056223280727863,
                -0.317335307598114
            ],
            [
                -1.6660805940628052,
                -1.1819530725479126,
                0.8901724815368652,
                0.9211023449897766,
                -0.06941866129636765,
                0.010061516426503658,
                -0.3829494118690491
            ],
            [
                -1.5353940725326538,
                -1.3118245601654053,
                0.8863720297813416,
                0.921332597732544,
                -0.06876746565103531,
                0.011659027077257633,
                -0.38246750831604004
            ],
            [
                -1.8926033973693848,
                -1.3909610509872437,
                1.0599069595336914,
                0.9041131138801575,
                -0.054554104804992676,
                0.0782046765089035,
                -0.41651809215545654
            ],
            [
                -1.9214211702346802,
                -1.413161277770996,
                0.879046618938446,
                0.9254816174507141,
                -0.05324266105890274,
                0.0015506735071539879,
                -0.3750287592411041
            ],
            [
                -1.7909960746765137,
                -1.5433179140090942,
                0.8761227130889893,
                0.9253843426704407,
                -0.052540648728609085,
                0.0031976578757166862,
                -0.3753573000431061
            ]
        ])


def _chunk_start_root_rel(
    pre_action_root_restored: np.ndarray,
    inference_start_step: int
) -> np.ndarray:
    """
    input: 上次推理restored后的action chunk的root xyz [T, 3]
    output: 下次推理时刻的initial root xyz [3,]
    """
    chunk_start_root_rel = pre_action_root_restored[inference_start_step + 1]
    return chunk_start_root_rel

def msg_to_6d_rot_joint(msg: WR_GAE_BodyPose_Msg_V2) -> dict:
    """Body joint vector (35,) = [imu_rot6d(6), joint_q(29)] from a BodyPose DDS message."""
    q_np = np.array(msg.q.copy(), dtype=np.float32)
    imu_np = np.array(msg.wxyz.copy(), dtype=np.float32)
    imu_np = compute_imu_relative(imu_np[None, :], imu_np[None, :])
    imu_np = quaternion_to_rotation_6d(imu_np)[0]
    return np.concatenate([imu_np, q_np], axis=0)


# def msg_to_6d_rot_joint(msg: WR_GAE_BodyPose_Msg) -> dict:
#     """Body joint vector (35,) = [imu_rot6d(6), joint_q(29)] from a BodyPose DDS message."""
#     q_np = np.array(msg.q.copy(), dtype=np.float32)
#     imu_np = np.array(msg.wxyz.copy(), dtype=np.float32)
#     imu_np = compute_imu_relative(imu_np[None, :], imu_np[None, :])
#     imu_np = quaternion_to_rotation_6d(imu_np)[0]
#     return np.concatenate([q_np, imu_np], axis=0)

def action_to_absolute_pose7_zero_fill_asyn(
    action: np.ndarray,
    init_pose: np.ndarray,
    pre_action_root_restored: np.ndarray = None,
    inference_start_step: int = None
) -> np.ndarray:
    """
    action: 模型侧返回回来的actions chunk
    init_pose: 机器人启动时的姿态
    两种情况:
    1. (T, 99) -> (T, 11, 9) -> zero-fill to (T, 15, 9) -> 6D to quat -> world-frame pose7. 
    2. (T, 102) -> 基于last_init_root_xyz恢复actions至局部坐标系下的绝对值 -> zero-fill to (T, 15, 9) -> 6D to quat -> world-frame pose7. 
    """
    assert action.shape[-1] == 99 or action.shape[-1] == 102, f"Expected action shape (T, 99) or (T, 102), got {action.shape}"
    if action.shape[-1] == 102:
        action = np.asarray(action, dtype=np.float32)
        
        num_frames = action.shape[0]
        action_with_velocity = action
        velocity = action_with_velocity[:,:3]
        action_mocap_flat = action_with_velocity[:,3:]
        action_mocap_xyz = action_mocap_flat[:, :33].reshape(-1, 11, 3)
        action_mocap_6d = action_mocap_flat[:,33:].reshape(-1, 11, 6)
        action_mocap = np.concatenate([action_mocap_xyz,action_mocap_6d],axis=-1).reshape(-1, 99)
        action_flat = np.concatenate([velocity, action_mocap], axis=-1)     # (B, 102) root_velocity + mocap 11x9
        # mocap_data.shape = (11, 9), next_init_root_xyz.shape = (3,)
        if pre_action_root_restored == None :
            last_init_root_xyz = None
        else:
            if inference_start_step == None:
                raise ValueError(f"inference_start_step should be provided")
            last_init_root_xyz = _chunk_start_root_rel(pre_action_root_restored, inference_start_step)
        mocap_data, _ = restore_mocap_from_root_relative(action_flat, last_init_root_xyz)
        action_11x9 = mocap_data.copy()

    elif action.shape[-1] == 99:
        # 99维度没有速度，所以直接6D转变成四元数，然后计算绝对值，即xyz变成绝对值，再xyz恢复yaw角，四元数恢复yaw角。
        num_frames = action.shape[0]
        action_11x9 = action.reshape(num_frames, 11, 9)
    else:
        raise ValueError(f"Expected action shape (T, 99) or (T, 102), got {action.shape}")

    # 开始6D转变成四元数，然后计算绝对值，即xyz变成绝对值，再xyz恢复yaw角，四元数恢复yaw角。
    action_15x9 = np.zeros((num_frames, 15, 9), dtype=np.float32)
    action_15x9[:, SELECT_11_INDICES, :] = action_11x9
    # rotation_6d_to_quaternion requires valid 6D; keep zero-fill semantics for xyz and
    # use identity 6D only where missing joints would otherwise be all-zero 6D.
    zero_rot6d = np.all(np.isclose(action_15x9[..., 3:9], 0.0), axis=-1)
    action_15x9[..., 3:9][zero_rot6d] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    action_15x7 = rotation_6d_to_quaternion(action_15x9)
    num_joints = action_15x7.shape[1]
    action_flat = action_15x7.reshape(num_frames * num_joints, 7)
    init_pose_flat = np.tile(init_pose, (num_frames * num_joints, 1))
    abs_flat = compute_absolute(init_pose_flat, action_flat)

    if action.shape[-1] == 102:
        return abs_flat.reshape(num_frames, num_joints, 7), mocap_data
    elif action.shape[-1] == 99:
        # 因为训练的时候state 帧率是100Hz，所以需要降采样到50Hz
        abs_flat = abs_flat.reshape(num_frames, num_joints, 7)
        abs_flat = abs_flat[::2]
        return abs_flat, mocap_data[:, :3]
    else:
        raise ValueError(f"Expected action shape (T, 99) or (T, 102), got {action.shape}")

def action_to_absolute_pose7_zero_fill(
    action: np.ndarray,
    init_pose: np.ndarray,
    last_init_root_xyz: np.ndarray,
) -> np.ndarray:
    """
    action: 模型侧返回回来的actions chunk
    init_pose: 机器人启动时的姿态
    两种情况:
    1. (T, 99) -> (T, 11, 9) -> zero-fill to (T, 15, 9) -> 6D to quat -> world-frame pose7. 
    2. (T, 102) -> 基于last_init_root_xyz恢复actions至局部坐标系下的绝对值 -> zero-fill to (T, 15, 9) -> 6D to quat -> world-frame pose7. 
    """
    assert action.shape[-1] == 99 or action.shape[-1] == 102, f"Expected action shape (T, 99) or (T, 102), got {action.shape}"
    if action.shape[-1] == 102:
        action = np.asarray(action, dtype=np.float32)
        
        num_frames = action.shape[0]
        action_with_velocity = action
        velocity = action_with_velocity[:,:3]
        action_mocap_flat = action_with_velocity[:,3:]
        action_mocap_xyz = action_mocap_flat[:, :33].reshape(-1, 11, 3)
        action_mocap_6d = action_mocap_flat[:,33:].reshape(-1, 11, 6)
        action_mocap = np.concatenate([action_mocap_xyz,action_mocap_6d],axis=-1).reshape(-1, 99)
        action_flat = np.concatenate([velocity, action_mocap], axis=-1)     # (B, 102) root_velocity + mocap 11x9
        # mocap_data.shape = (11, 9), next_init_root_xyz.shape = (3,)
        mocap_data, next_init_root_xyz = restore_mocap_from_root_relative(action_flat, last_init_root_xyz)
        action_11x9 = mocap_data

    elif action.shape[-1] == 99:
        # 99维度没有速度，所以直接6D转变成四元数，然后计算绝对值，即xyz变成绝对值，再xyz恢复yaw角，四元数恢复yaw角。
        num_frames = action.shape[0]
        action_11x9 = action.reshape(num_frames, 11, 9)
        next_init_root_xyz = None

    else:
        raise ValueError(f"Expected action shape (T, 99) or (T, 102), got {action.shape}")

    # 开始6D转变成四元数，然后计算绝对值，即xyz变成绝对值，再xyz恢复yaw角，四元数恢复yaw角。
    action_15x9 = np.zeros((num_frames, 15, 9), dtype=np.float32)
    action_15x9[:, SELECT_11_INDICES, :] = action_11x9
    # rotation_6d_to_quaternion requires valid 6D; keep zero-fill semantics for xyz and
    # use identity 6D only where missing joints would otherwise be all-zero 6D.
    zero_rot6d = np.all(np.isclose(action_15x9[..., 3:9], 0.0), axis=-1)
    action_15x9[..., 3:9][zero_rot6d] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    action_15x7 = rotation_6d_to_quaternion(action_15x9)
    num_joints = action_15x7.shape[1]
    action_flat = action_15x7.reshape(num_frames * num_joints, 7)
    init_pose_flat = np.tile(init_pose, (num_frames * num_joints, 1))
    abs_flat = compute_absolute(init_pose_flat, action_flat)

    if action.shape[-1] == 102:
        return abs_flat.reshape(num_frames, num_joints, 7), next_init_root_xyz
    elif action.shape[-1] == 99:
        # 因为训练的时候state 帧率是100Hz，所以需要降采样到50Hz
        abs_flat = abs_flat.reshape(num_frames, num_joints, 7)
        abs_flat = abs_flat[::2]
        return abs_flat, next_init_root_xyz
    else:
        raise ValueError(f"Expected action shape (T, 99) or (T, 102), got {action.shape}")


class BodyJointHistory:
    """Thread-safe ring buffer of body_joint (35,) vectors; pad with earliest frame when short."""

    def __init__(self, maxlen: int):
        self._lock = threading.Lock()
        self._buf: deque[np.ndarray] = deque(maxlen=maxlen)

    def push(self, body_joint: np.ndarray) -> None:
        with self._lock:
            self._buf.append(np.asarray(body_joint, dtype=np.float32))

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def stack(self, target_len: int | None = None, step: int = 1) -> np.ndarray:
        """
        Return (target_len, D) with oldest-to-newest order.
        Samples ..., buf[-1-step], buf[-1] when step=K (newest always last row).
        """
        with self._lock:
            if not self._buf:
                raise RuntimeError("body joint history is empty")
            frames = list(self._buf)

        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")

        t = target_len if target_len is not None else len(frames)
        n = len(frames)
        earliest = frames[0]

        selected = []
        for i in range(t):
            offset = (t - 1 - i) * step
            idx = n - 1 - offset
            selected.append(frames[idx] if idx >= 0 else earliest)
        return np.stack(selected, axis=0)


class MocapDataQueue:
    def __init__(self, maxlen=300):
        self._lock = threading.Lock()
        self._queue = deque(maxlen=maxlen)
        self._frames_sent = 0
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

    def put_pose7(self, mocap_frame: np.ndarray) -> None:
        frame = np.asarray(mocap_frame)
        xyz = frame[:, 0:3]
        wxyz = normalize_quaternion(frame[:, 3:7])
        self.put(xyz, wxyz)

    def get_next_or_last(self):
        with self._lock:
            if len(self._queue) > 0:
                xyz, wxyz = self._queue.popleft()
                self._last_xyz = xyz
                self._last_wxyz = wxyz
                self._frames_sent += 1
                return xyz, wxyz
            return self._last_xyz, self._last_wxyz

    @property
    def frames_sent(self) -> int:
        with self._lock:
            return self._frames_sent

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


class BodyJointCollectorThread(threading.Thread):
    def __init__(
        self,
        body_pose: BodyPoseSubscriber,
        history: BodyJointHistory,
        fps: float,
    ):
        super().__init__(daemon=True)
        self.body_pose = body_pose
        self.history = history
        self.running = True
        self.rate = RateLimiter(frequency=fps)

    def run(self):
        while self.running:
            msg = self.body_pose.get_msg()
            if msg is not None:
                self.history.push(msg_to_6d_rot_joint(msg))
            self.rate.sleep()

    def stop(self):
        self.running = False


def body_joint_history_maxlen(
    state_history_len: int,
    state_history_step: int,
    body_joint_history_fps: float,
    body_joint_history_buffer_sec: float,
) -> int:
    span = 1 + (state_history_len - 1) * state_history_step
    fps_buffer = int(body_joint_history_fps * body_joint_history_buffer_sec)
    return max(span, fps_buffer)


def wait_for_root_pose(body_pose: BodyPoseSubscriber) -> np.ndarray:
    """阻塞直到 DDS 上报根位姿 (7,) [x,y,z,qw,qx,qy,qz]。"""
    while True:
        root_pose = body_pose.get_root_pose()
        if root_pose is not None:
            logger.info("root pose ready")
            return root_pose
        sleep(0.01)


def seed_mocap_queue_from_replay(
    mocap_queue: MocapDataQueue,
    replay_data: np.ndarray,
    root_pose: np.ndarray,
) -> None:
    """用 replay 首帧（相对根）结合当前 root，生成绝对 pose 并预填 mocap 队列，减轻起步抖动。"""
    replay_frame = replay_data[0:1, :, :].copy()
    num_frames, num_joints, poses = replay_frame.shape
    root_tiled = np.tile(root_pose, (num_frames * num_joints, 1))
    flat = replay_frame.reshape(num_frames * num_joints, -1)
    abs_frame = compute_absolute(root_tiled, flat).reshape(num_frames, num_joints, poses)[0]
    mocap_queue.put_pose7(abs_frame)


def wait_for_body_joint_history(
    history: BodyJointHistory,
    state_history_len: int,
    state_history_step: int,
) -> None:
    """等待采集线程攒够 stack(K 步采样) 所需的最少帧数。"""
    min_frames = 1 + (state_history_len - 1) * state_history_step
    while len(history) < min_frames:
        sleep(0.01)
    logger.info("body joint history ready: %d / %d frames", len(history), min_frames)


def capture_camera_frames(camera_caps: dict[str, VideoCapture]) -> dict[str, np.ndarray]:
    """读取各相机当前帧；BGR 转 RGB。连读数次用于清空相机缓冲。"""
    frames = {}
    for name, cap in camera_caps.items():
        for _ in range(5):
            _, frames[name] = cap.read()
        frames[name] = frames[name][..., ::-1]
    return frames


class AsyncInferenceWorker:
    """Single in-flight background inference; main thread submit / poll_done / get_result."""

    def __init__(self, infer_fn: Callable[..., np.ndarray]):
        self._infer_fn = infer_fn
        self._lock = threading.Lock()
        self._busy = False
        self._done = threading.Event()
        self._result: np.ndarray | None = None
        self._error: BaseException | None = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def submit(self, **kwargs: Any) -> None:
        with self._lock:
            if self._busy:
                raise RuntimeError("inference already in flight")
            self._busy = True
            self._done.clear()
            self._result = None
            self._error = None
        threading.Thread(target=self._run, kwargs=kwargs, daemon=True).start()

    def _run(self, **kwargs: Any) -> None:
        try:
            result = self._infer_fn(**kwargs)
            with self._lock:
                self._result = result
        except BaseException as exc:
            with self._lock:
                self._error = exc
        finally:
            self._done.set()
            with self._lock:
                self._busy = False
    
    def poll_done(self) -> bool:
        with self._lock:
            return (not self._busy) and self._done.is_set() and (
                self._result is not None or self._error is not None
            )

    def get_result(self) -> np.ndarray:
        with self._lock:
            if not self._done.is_set():
                raise RuntimeError("inference not done yet")

            error = self._error
            result = self._result

            # 关键：消费后清空，避免重复 poll_done=True
            self._done.clear()
            self._result = None
            self._error = None

        if error is not None:
            raise error
        if result is None:
            raise RuntimeError("inference finished without result")
        return result


def enqueue_abs_action_chunk(
    mocap_queue: MocapDataQueue,
    abs_action: np.ndarray,
    start: int,
    end: int,
) -> int:
    """Enqueue abs_action[start:end]; returns number of frames enqueued."""
    n = 0
    
    while True:
        if not mocap_queue.empty():
            sleep(0.01)
            continue
        break
    
    for t in range(start, end):
        if t >= abs_action.shape[0]:
            break
        mocap_queue.put_pose7(abs_action[t])
        n += 1
    return n



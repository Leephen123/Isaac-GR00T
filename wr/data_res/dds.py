from collections import deque
import threading
import time
from dataclasses import dataclass
from typing import Final
from loop_rate_limiters import RateLimiter
import numpy as np
from data_res.transforms import compute_imu_relative, quaternion_to_rotation_6d
import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotations
import cyclonedds.idl.types as types
from cyclonedds.core import Qos, Policy
from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import DataWriter, Publisher
from cyclonedds.sub import DataReader, Subscriber
from cyclonedds.topic import Topic


MOCAP_NUM_JOINTS: Final = 15
MOCAP_POS_DIM: Final = 3  # xyz
MOCAP_QUAT_DIM: Final = 4  # wxyz
MOCAP_XYZ_SIZE: Final = MOCAP_NUM_JOINTS * MOCAP_POS_DIM
MOCAP_WXYZ_SIZE: Final = MOCAP_NUM_JOINTS * MOCAP_QUAT_DIM

BODY_POSE_XYZ_SIZE: Final = 45  # 15 joints * 3
BODY_POSE_WXYZ_SIZE: Final = 4  # root quaternion only
BODY_POSE_Q_SIZE: Final = 29  # joint angles
BODY_POSE_ROT6D_Q_SIZE: Final = 35  # root rotation 6d + joint angles


@dataclass
@annotations.final
@annotations.autoid("sequential")
class MocapUE5G115Msg(idl.IdlStruct, typename="MocapUE5G115Msg"):
    fps: types.float32
    timestamp: types.int64
    xyz: types.array[types.float32, MOCAP_XYZ_SIZE]
    wxyz: types.array[types.float32, MOCAP_WXYZ_SIZE]
    fingers: types.array[types.float32, MOCAP_NUM_JOINTS]


@dataclass
class MocapConfig:
    domain_id: int = 1
    """CycloneDDS domain ID."""
    topic_name: str = "MocapUE5G115Topic"
    """DDS topic name."""
    depth: int = 4
    """QoS KeepLast history depth."""


@dataclass
@annotations.final
@annotations.autoid("sequential")
class WR_GAE_BodyPose_Msg(idl.IdlStruct, typename="WR_GAE_BodyPose_Msg"):
    fps: types.float32
    xyz: types.array[types.float32, BODY_POSE_XYZ_SIZE]
    wxyz: types.array[types.float32, BODY_POSE_WXYZ_SIZE]
    q: types.array[types.float32, BODY_POSE_Q_SIZE]
    timestamp: types.int64


@dataclass
class BodyPoseConfig:
    domain_id: int = 1
    """CycloneDDS domain ID."""
    topic_name: str = "WR/BodyPose"
    """DDS topic name."""
    depth: int = 4
    """QoS KeepLast history depth."""


class MocapUE5G115MsgSubscriber:
    def __init__(self):
        self.participant = DomainParticipant(domain_id=1)
        self.qos = Qos(
            Policy.History.KeepLast(depth=4),
        )
        print("types module path:", types.__file__)
        self.topic = Topic(self.participant, "MocapUE5G115TopicResponse", MocapUE5G115Msg, qos=self.qos)
        self.subscriber = Subscriber(self.participant)
        self.reader = DataReader(self.subscriber, self.topic)

    def get_msg(self, timeout_sec=1.0):
        """
        阻塞等待 DDS 数据，避免 reader.read() 为空时 [-1] 报错。
        timeout_sec 超时后返回 None。
        """
        t0 = time.time()

        while True:
            samples = self.reader.read()

            if len(samples) > 0:
                return samples[-1]

            if time.time() - t0 > timeout_sec:
                return None

            time.sleep(0.001)

class MocapUE5G115MsgPublisher:
    def __init__(self, cfg: MocapConfig, fps):
        self.participant = DomainParticipant(domain_id=cfg.domain_id)
        self.qos = Qos(Policy.History.KeepLast(depth=cfg.depth))
        self.topic = Topic(self.participant, cfg.topic_name, MocapUE5G115Msg, qos=self.qos)
        self.publisher = Publisher(self.participant)
        self.writer = DataWriter(self.publisher, self.topic)
        self.fps = fps

    def send_msg(self, xyz, wxyz, fingers=None):
        xyz = np.asarray(xyz, dtype=np.float32)
        wxyz = np.asarray(wxyz, dtype=np.float32)
        assert xyz.shape == (MOCAP_NUM_JOINTS, MOCAP_POS_DIM), (
            f"xyz shape must be ({MOCAP_NUM_JOINTS}, {MOCAP_POS_DIM}), got {xyz.shape}"
        )
        assert wxyz.shape == (MOCAP_NUM_JOINTS, MOCAP_QUAT_DIM), (
            f"wxyz shape must be ({MOCAP_NUM_JOINTS}, {MOCAP_QUAT_DIM}), got {wxyz.shape}"
        )

        if fingers is None:
            fingers = np.zeros(MOCAP_NUM_JOINTS, dtype=np.float32)
        else:
            fingers = np.asarray(fingers, dtype=np.float32)
            assert fingers.shape == (MOCAP_NUM_JOINTS,), (
                f"fingers shape must be ({MOCAP_NUM_JOINTS},), got {fingers.shape}"
            )

        msg = MocapUE5G115Msg(
            fps=np.float32(self.fps),
            timestamp=np.int64(time.time_ns()),
            xyz=xyz.reshape(-1).tolist(),
            wxyz=wxyz.reshape(-1).tolist(),
            fingers=fingers.tolist(),
        )
        self.writer.write(msg)


class BodyPoseSubscriber:
    def __init__(
        self,
        cfg: BodyPoseConfig,
        state_queue: deque = None,
        history_fps: float | None = None,
        history_maxlen: int | None = None,
        auto_start_history: bool = False,
    ):
        self.participant = DomainParticipant(domain_id=cfg.domain_id)
        self.qos = Qos(Policy.History.KeepLast(depth=cfg.depth))
        self.topic = Topic(self.participant, cfg.topic_name, WR_GAE_BodyPose_Msg, qos=self.qos)
        self.subscriber = Subscriber(self.participant)
        self.reader = DataReader(self.subscriber, self.topic)
        self._reader_lock = threading.Lock()
        self.history_fps = history_fps
        self._body_joint_history = state_queue if state_queue is not None else deque(maxlen=history_maxlen)
        self._body_joint_history_lock = threading.Lock()
        self._history_stop_event = threading.Event()
        self._history_thread: threading.Thread | None = None

        if auto_start_history:
            self.start_collecting_body_joint(history_fps=history_fps)

    def msg_to_root_pose(self, msg: WR_GAE_BodyPose_Msg) -> np.ndarray:
        root_xyz = np.asarray(msg.xyz[0:3], dtype=np.float32)
        root_wxyz = np.asarray(msg.wxyz, dtype=np.float32)
        return np.concatenate([root_xyz, root_wxyz], axis=0)

    def msg_to_6d_rot_joint(self, msg: WR_GAE_BodyPose_Msg) -> np.ndarray:
        q_np = np.array(msg.q.copy(), dtype=np.float32)
        imu_np = np.array(msg.wxyz.copy(), dtype=np.float32)
        imu_np = compute_imu_relative(imu_np[None, :], imu_np[None, :])
        imu_np = quaternion_to_rotation_6d(imu_np)[0]
        return np.concatenate([imu_np, q_np], axis=0).astype(np.float32)

    def get_msg(self) -> WR_GAE_BodyPose_Msg | None:
        with self._reader_lock:
            samples = self.reader.read()
        if not samples:
            return None
        return samples[-1]

    def get_root_pose(self) -> np.ndarray | None:
        """Return root pose as (7,) float32 array [x, y, z, qw, qx, qy, qz], or None."""
        msg = self.get_msg()
        if msg is None:
            return None
        return self.msg_to_root_pose(msg)

    def get_body_joint(self) -> np.ndarray | None:
        """Return latest root rotation 6d + body joint as (35,) float32 array, or None."""
        msg = self.get_msg()
        if msg is None:
            return None
        return self.msg_to_6d_rot_joint(msg)

    def start_collecting_body_joint(self, history_fps: float | None = None):
        if history_fps is not None:
            self.history_fps = history_fps
        if self.history_fps is None or self.history_fps <= 0:
            raise ValueError(f"history_fps must be positive, got {self.history_fps}")
        if self._history_thread is not None and self._history_thread.is_alive():
            return

        self._history_stop_event.clear()
        self._history_thread = threading.Thread(target=self._collect_body_joint_loop, daemon=True)
        self._history_thread.start()

    def stop_collecting_body_joint(self, timeout: float | None = 2.0):
        self._history_stop_event.set()
        if self._history_thread is not None:
            self._history_thread.join(timeout=timeout)

    def _collect_body_joint_loop(self):
        history_timer = RateLimiter(frequency=self.history_fps)
        while not self._history_stop_event.is_set():
            body_joint = self.get_body_joint()
            if body_joint is not None:
                with self._body_joint_history_lock:
                    self._body_joint_history.append(body_joint)
            history_timer.sleep()

    def _copy_latest_body_joint_history(self, num_history: int | None = None) -> list[np.ndarray]:
        if num_history is not None and num_history <= 0:
            raise ValueError(f"num_history must be positive, got {num_history}")

        if num_history is None:
            history_source = self._body_joint_history
        else:
            history_source = list(self._body_joint_history)
            if len(history_source) > num_history:
                history_source = history_source[-num_history:]
            elif history_source:
                num_padding = num_history - len(history_source)
                history_source = [history_source[0]] * num_padding + history_source
        return [body_joint.copy() for body_joint in history_source]

    def get_history_body_joint(self, num_history: int | None = None, clear: bool = False) -> np.ndarray:
        """Return the latest num_history body joint history frames as (N, 35) float32."""
        with self._body_joint_history_lock:
            history = self._copy_latest_body_joint_history(num_history=num_history)
            if clear:
                self._body_joint_history.clear()
        return np.asarray(history, dtype=np.float32).reshape(-1, BODY_POSE_ROT6D_Q_SIZE)

    def clear_history_body_joint(self):
        with self._body_joint_history_lock:
            self._body_joint_history.clear()

    def get_history_pose(self, num_history: int | None = None, clear: bool = False):
        return self.get_history_body_joint(num_history=num_history, clear=clear)
    

# ============================================================
# Hand Message
# ============================================================
@dataclass
@annotations.final
@annotations.autoid("sequential")
class MocapUEHandData_Msg(idl.IdlStruct, typename="MocapUEHandData_Msg"):
    handid: types.int32
    dof: types.int32
    timestamp: types.int64
    joints: types.array[types.float32, 60]
    enable: types.int32



# ============================================================
#  新增：Hand Subscriber（
# ============================================================
class MocapUEHandSubscriber:
    """
    DDS Hand:
    - cmd: MocapHandsCmd
    - state: RobotHandState
    """

    def __init__(self, domain_id: int = 1, depth: int = 4):

        # =====================
        # ONLY ONE participant
        # =====================
        self.participant = DomainParticipant(domain_id=domain_id)

        # ✔ ONE subscriber（关键修复）
        self.subscriber = Subscriber(self.participant)

        self.qos = Qos(
            Policy.History.KeepLast(depth=depth)
        )

        # =====================
        # CMD
        # =====================
        self.topic_cmd = Topic(
            self.participant,
            "MocapHandsCmd",
            MocapUEHandData_Msg,
            qos=self.qos
        )

        self.reader_cmd = DataReader(
            self.subscriber,
            self.topic_cmd
        )

        # =====================
        # STATE
        # =====================
        self.topic_state = Topic(
            self.participant,
            "RobotHandState",
            MocapUEHandData_Msg,
            qos=self.qos
        )

        self.reader_state = DataReader(
            self.subscriber,
            self.topic_state
        )

    # =========================
    # safe helper
    # =========================
    def _safe_last(self, reader):
        samples = reader.read()
        if not samples:
            return None
        return samples[-1]

    # =========================
    # CMD
    # =========================
    def get_cmd(self):
        msg = self._safe_last(self.reader_cmd)
        if msg is None:
            return None
        return np.array(msg.joints[:msg.dof], dtype=np.float32)

    # =========================
    # STATE
    # =========================
    def get_state(self):
        msg = self._safe_last(self.reader_state)
        if msg is None:
            return None

        return np.array(msg.joints[:msg.dof], dtype=np.float32)


    # =========================
    # unified
    # =========================
    def get(self):
        return {
            "cmd": self.get_cmd(),
            "state": self.get_state()
        }


# ============================================================
# Hand Publisher
# ============================================================
class MocapUEHandPublisher:
    """
    DDS Hand Publisher

    Topic:
        MocapHandsCmd

    输入:
        joints: np.ndarray(shape=(12,))
                数值范围 0~1000

    发送:
        joints / 1000.0
        handid = 4
        dof = 12
        enable = 1
    """

    def __init__(
        self,
        domain_id: int = 1,
        topic_name: str = "MocapUEHand",
        depth: int = 4,
    ):
        self.participant = DomainParticipant(domain_id=domain_id)

        self.publisher = Publisher(self.participant)

        self.qos = Qos(
            Policy.History.KeepLast(depth=depth)
        )

        self.topic = Topic(
            self.participant,
            topic_name,
            MocapUEHandData_Msg,
            qos=self.qos,
        )

        self.writer = DataWriter(
            self.publisher,
            self.topic,
        )

    def send(self, joints: np.ndarray):
        """
        joints:
            shape=(12,)
            value range: 0~1000
        """

        joints = np.asarray(joints, dtype=np.float32)

        if joints.shape != (12,):
            raise ValueError(
                f"Expected joints shape (12,), got {joints.shape}"
            )
        print(f"hand joints: {joints}")

        # 缩放到 0~1
        joints_scaled = joints / 1000.0

        joints_scaled = 1 - joints_scaled

        # DDS 固定长度 60
        joints_msg = np.zeros(60, dtype=np.float32)
        joints_msg[:12] = joints_scaled

        msg = MocapUEHandData_Msg(
            handid=4,
            dof=12,
            timestamp=int(time.time_ns()),
            joints=joints_msg.tolist(),
            enable=1,
        )

        self.writer.write(msg)



class BodyPoseSubscriberV3:
    def __init__(self, cfg: BodyPoseConfig, mocap_cfg: MocapConfig = None, topic_name="MocapUE5G115TopicResponse", tag="v2"):
        # self.participant = DomainParticipant(domain_id=mocap_cfg.domain_id)
        # self.qos = Qos(Policy.History.KeepLast(depth=mocap_cfg.depth))
        # self.topic = Topic(self.participant, topic_name, MocapUE5G115Msg, qos=self.qos)
        # self.subscriber = Subscriber(self.participant)
        # self.reader = DataReader(self.subscriber, self.topic)
        # print(f"cfg.domain_id: {cfg.domain_id}, cfg.depth: {cfg.depth}, topic_name: {topic_name}")
        # mocap_cfg.topic_name = topic_name
        self.mocap_subscriber = MocapUE5G115MsgSubscriber()

        self.cfg = cfg
        self.tag = tag
        
        if self.tag == "v1":
            self.body_pose = BodyPoseSubscriber(cfg)
        elif self.tag == "v2":
            self.body_pose = BodyPoseSubscriberV2_15(cfg)
        
    # def msg_to_root_pose(self, msg: WR_GAE_BodyPose_Msg) -> np.ndarray:
    #     if self.tag == "v1":
    #         return self.body_pose.msg_to_root_pose(msg)
    #     elif self.tag == "v2":
    #         return self.body_pose.msg_to_root_pose(msg)
    
    def get_msg_self(self):
        # while True:
        #     # samples = self.mocap_subscriber.get_msg()
        #     samples = self.mocap_subscriber.get_msg()
        #     # if not samples:
        #     #     return None
        #     if not samples and samples is not None:

        #         # print(f"1111111111...........")
        #         break
        #     else: 
        #         print(f"waiting22222...........samples: {samples}")
        #         pass
        # # print(f"samples: {samples}")
        samples = self.mocap_subscriber.get_msg()

        return samples

    def get_msg(self) -> WR_GAE_BodyPose_Msg | None:
        if self.tag == "v1":
            msg = self.body_pose.get_msg()
            msg_self = self.get_msg_self()
            # msg.wxyz = msg_self.wxyz[:4]
            return msg
        elif self.tag == "v2":
            msg = self.body_pose.get_msg()
            msg_self = self.get_msg_self()
            # msg.robot_rootquat = msg_self.wxyz[:4]
            return msg

    def get_root_pose(self) -> np.ndarray | None:
        if self.tag == "v1":
            root_pose = self.body_pose.get_root_pose()
            root_pose[3:] = self.get_msg_self().wxyz[:4]
            return root_pose
        elif self.tag == "v2":
            root_pose = self.body_pose.get_root_pose()
            # print(f"root_pose111111: {root_pose}")
            # print(f"self.get_msg_self().wxyz: {self.get_msg_self().wxyz}")
            root_pose[3:] = self.get_msg_self().wxyz[:4]
            return root_pose

    def get_15_pose7(self) -> np.ndarray | None:
        """Return the latest 15-point pose when using the V2 body-pose topic."""
        if self.tag != "v2":
            return None
        return self.body_pose.get_15_pose7()


# ============================================================
# WR_GAE BodyPose V2: 15-point pose7 extractor
# root xyz uses waist_yaw xyz, root wxyz uses robot_rootquat
# ============================================================

_BODY_POSE_V2_MAX_ROBOT_JOINTS = 50
_BODY_POSE_V2_MAX_HAND_JOINTS = 60

_BODY_POSE_V2_ROBOT_QPOS_SIZE = 50
_BODY_POSE_V2_ROBOT_XQUAT_SIZE = 50 * 4
_BODY_POSE_V2_ROBOT_XPOS_SIZE = 50 * 3
_BODY_POSE_V2_ROOTQUAT_SIZE = 4
_BODY_POSE_V2_HAND_QPOS_SIZE = 60
_BODY_POSE_V2_RESERVE_SIZE = 100


@dataclass
@annotations.final
@annotations.autoid("sequential")
class WR_GAE_BodyPose_Msg_V2(
    idl.IdlStruct,
    typename="WR_GAE_BodyPose_Msg",
):
    fps: types.float32

    robot_id: types.int32
    robot_joint_num: types.int32

    robot_qpos: types.array[
        types.float32,
        _BODY_POSE_V2_ROBOT_QPOS_SIZE,
    ]

    # flattened: (50, 4), wxyz
    robot_xquat: types.array[
        types.float32,
        _BODY_POSE_V2_ROBOT_XQUAT_SIZE,
    ]

    # flattened: (50, 3), xyz
    robot_xpos: types.array[
        types.float32,
        _BODY_POSE_V2_ROBOT_XPOS_SIZE,
    ]

    # root quaternion: [qw, qx, qy, qz]
    robot_rootquat: types.array[
        types.float32,
        _BODY_POSE_V2_ROOTQUAT_SIZE,
    ]

    hand_id: types.int32
    hand_joint_num: types.int32

    hand_qpos: types.array[
        types.float32,
        _BODY_POSE_V2_HAND_QPOS_SIZE,
    ]

    reserve: types.array[
        types.float32,
        _BODY_POSE_V2_RESERVE_SIZE,
    ]

    timestamp: types.int64


G1_29_BODY_NAMES = [
    "left_hip_pitch",       # 0
    "left_hip_roll",        # 1
    "left_hip_yaw",         # 2
    "left_knee",            # 3
    "left_ankle_pitch",     # 4
    "left_ankle_roll",      # 5

    "right_hip_pitch",      # 6
    "right_hip_roll",       # 7
    "right_hip_yaw",        # 8
    "right_knee",           # 9
    "right_ankle_pitch",    # 10
    "right_ankle_roll",     # 11

    "waist_yaw",            # 12
    "waist_roll",           # 13
    "waist_pitch",          # 14

    "left_shoulder_pitch",  # 15
    "left_shoulder_roll",   # 16
    "left_shoulder_yaw",    # 17
    "left_elbow",           # 18
    "left_wrist_roll",      # 19
    "left_wrist_pitch",     # 20
    "left_wrist_yaw",       # 21

    "right_shoulder_pitch", # 22
    "right_shoulder_roll",  # 23
    "right_shoulder_yaw",   # 24
    "right_elbow",          # 25
    "right_wrist_roll",     # 26
    "right_wrist_pitch",    # 27
    "right_wrist_yaw",      # 28
]


# 15 points:
# 第 0 个点是 root/pelvis:
#   xyz  用 waist_yaw(index=12) 近似
#   wxyz 用 msg.robot_rootquat
SELECT_15_INDICES = [
    12,

    1,
    3,
    5,
    5,

    7,
    9,
    11,
    11,

    16,
    18,
    21,

    23,
    25,
    28,
]

SELECT_15_NAMES = [
    "root_pelvis_approx_waist_yaw",

    "left_hip_roll",
    "left_knee",
    "left_ankle_roll",
    "left_ankle_roll_dup",

    "right_hip_roll",
    "right_knee",
    "right_ankle_roll",
    "right_ankle_roll_dup",

    "left_shoulder_roll",
    "left_elbow",
    "left_wrist_yaw",

    "right_shoulder_roll",
    "right_elbow",
    "right_wrist_yaw",
]


class BodyPoseSubscriberV2_15:
    def __init__(self, cfg: BodyPoseConfig):
        self.participant = DomainParticipant(domain_id=cfg.domain_id)
        self.qos = Qos(Policy.History.KeepLast(depth=cfg.depth))

        self.topic = Topic(
            self.participant,
            cfg.topic_name,
            WR_GAE_BodyPose_Msg_V2,
            qos=self.qos,
        )

        self.subscriber = Subscriber(self.participant)
        self.reader = DataReader(self.subscriber, self.topic)

    def get_msg(self) -> WR_GAE_BodyPose_Msg_V2 | None:
        samples = self.reader.read()
        if not samples:
            return None
        return samples[-1]

    def _get_xyz_wxyz_all(self, msg: WR_GAE_BodyPose_Msg_V2):
        n = int(msg.robot_joint_num)

        if n < 29:
            raise ValueError(
                f"robot_joint_num={n}, expected at least 29. "
                f"Valid robot_xpos/robot_xquat data may be incomplete."
            )

        xyz_all = np.asarray(
            msg.robot_xpos[:29 * 3],
            dtype=np.float32,
        ).reshape(29, 3)

        wxyz_all = np.asarray(
            msg.robot_xquat[:29 * 4],
            dtype=np.float32,
        ).reshape(29, 4)

        return xyz_all, wxyz_all

    def get_15_xyz_wxyz(self):
        """
        Return:
            xyz:  (15, 3)
            wxyz: (15, 4)

        第 0 个点:
            xyz  = waist_yaw xyz, index 12
            wxyz = robot_rootquat
        """
        msg = self.get_msg()

        if msg is None:
            return None, None

        xyz_all, wxyz_all = self._get_xyz_wxyz_all(msg)
        # print(f"xyz_all: {xyz_all}")

        selected = np.asarray(SELECT_15_INDICES, dtype=np.int64)

        if selected.max() >= 29 or selected.min() < 0:
            raise ValueError(
                f"Invalid SELECT_15_INDICES={SELECT_15_INDICES}"
            )

        xyz = xyz_all[selected].copy()
        wxyz = wxyz_all[selected].copy()

        # root/pelvis 特殊处理
        xyz[0] = xyz_all[12]
        wxyz[0] = np.asarray(msg.robot_rootquat, dtype=np.float32)

        return xyz, wxyz

    def get_15_pose7(self):
        """
        Return:
            pose15: (15, 7)
            [x, y, z, qw, qx, qy, qz]
        """
        xyz, wxyz = self.get_15_xyz_wxyz()

        if xyz is None:
            return None

        return np.concatenate([xyz, wxyz], axis=-1)

    def get_root_pose(self):
        """
        Return:
            root pose: (7,)
            [x, y, z, qw, qx, qy, qz]

        xyz  = waist_yaw xyz
        wxyz = robot_rootquat
        """
        pose15 = self.get_15_pose7()

        if pose15 is None:
            return None

        return pose15[0]

    def get_robot_qpos(self):
        msg = self.get_msg()

        if msg is None:
            return None

        n = int(msg.robot_joint_num)
        n = max(0, min(n, _BODY_POSE_V2_MAX_ROBOT_JOINTS))

        return np.asarray(
            msg.robot_qpos[:n],
            dtype=np.float32,
        )

    def get_hand_qpos(self):
        msg = self.get_msg()

        if msg is None:
            return None

        n = int(msg.hand_joint_num)
        n = max(0, min(n, _BODY_POSE_V2_MAX_HAND_JOINTS))

        return np.asarray(
            msg.hand_qpos[:n],
            dtype=np.float32,
        )

    def print_15_names(self):
        for i, name in enumerate(SELECT_15_NAMES):
            src_idx = SELECT_15_INDICES[i]
            src_name = G1_29_BODY_NAMES[src_idx]
            print(f"{i:02d}: {name}  <-  index {src_idx}: {src_name}")

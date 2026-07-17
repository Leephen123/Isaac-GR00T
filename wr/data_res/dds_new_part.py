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




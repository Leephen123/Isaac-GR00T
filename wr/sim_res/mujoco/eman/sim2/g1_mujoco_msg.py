
from enum import auto
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

import numpy as np

@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorCmd_(idl.IdlStruct, typename="unitree_hg.msg.dds_.MotorCmd_"):
    mode: types.uint8
    q: types.float32
    dq: types.float32
    tau: types.float32
    kp: types.float32
    kd: types.float32
    reserve: types.uint32

@dataclass
@annotate.final
@annotate.autoid("sequential")
class LowCmd_(idl.IdlStruct, typename="unitree_hg.msg.dds_.LowCmd_"):
    mode_pr: types.uint8
    mode_machine: types.uint8
    motor_cmd: types.array[MotorCmd_, 35]
    reserve: types.array[types.uint32, 4]
    crc: types.uint32

@dataclass
@annotate.final
@annotate.autoid("sequential")
class IMUState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.IMUState_"):
    quaternion: types.array[types.float32, 4]
    gyroscope: types.array[types.float32, 3]
    accelerometer: types.array[types.float32, 3]
    rpy: types.array[types.float32, 3]
    temperature: types.int16

@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.MotorState_"):
    mode: types.uint8
    q: types.float32
    dq: types.float32
    ddq: types.float32
    tau_est: types.float32
    temperature: types.array[types.int16, 2]
    vol: types.float32
    sensor: types.array[types.uint32, 2]
    motorstate: types.uint32
    reserve: types.array[types.uint32, 4]

@dataclass
@annotate.final
@annotate.autoid("sequential")
class LowState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.LowState_"):
    version: types.array[types.uint32, 2]
    mode_pr: types.uint8
    mode_machine: types.uint8
    tick: types.uint32
    imu_state: IMUState_
    motor_state: types.array[MotorState_, 35]
    wireless_remote: types.array[types.uint8, 40]
    reserve: types.array[types.uint32, 4]
    crc: types.uint32


def unitree_hg_msg_dds__MotorCmd_():
    return MotorCmd_(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

def unitree_hg_msg_dds__LowCmd_():
    return LowCmd_(0, 0, [unitree_hg_msg_dds__MotorCmd_() for i in range(35)], [0, 0, 0, 0], 0)

def unitree_hg_msg_dds__IMUState_():
    return IMUState_([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0)

def unitree_hg_msg_dds__MotorState_():
    return MotorState_(0, 0.0, 0.0, 0.0, 0.0, [0, 0], 0.0, [0, 0], 0,  [0, 0, 0, 0])

def unitree_hg_msg_dds__LowState_():
    return LowState_([0, 0], 0, 0, 0, unitree_hg_msg_dds__IMUState_(),
                [unitree_hg_msg_dds__MotorState_() for i in range(35)],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0], 0)


import time
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.sub import Subscriber, DataReader
from cyclonedds.util import duration
from cyclonedds.core import Qos, Policy

from sim_res.mujoco.eman.sim2.topic import TOPIC_MUJOCO_LOW_CMD, TOPIC_MUJOCO_LOW_STATE

class MujocoLowCmdPublisher:
    def __init__(self):
        self.low_cmd_msg = unitree_hg_msg_dds__LowCmd_()

        self.low_cmd_participant = DomainParticipant(domain_id=0)
        self.low_cmd_qos = Qos(
            Policy.History.KeepLast(depth=1),
        )
        self.low_cmd_topic = Topic(self.low_cmd_participant, TOPIC_MUJOCO_LOW_CMD, LowCmd_, qos=self.low_cmd_qos)
        self.low_cmd_publisher = Publisher(self.low_cmd_participant)
        self.low_cmd_writer = DataWriter(self.low_cmd_publisher, self.low_cmd_topic)

    def publish(self):
        self.low_cmd_writer.write(self.low_cmd_msg)


class MujocoLowCmdSubscriber:
    def __init__(self):
        self.low_cmd_participant = DomainParticipant(domain_id=0)
        self.low_cmd_qos = Qos(
            Policy.History.KeepLast(depth=1),
        )
        self.low_cmd_topic = Topic(self.low_cmd_participant, TOPIC_MUJOCO_LOW_CMD, LowCmd_, qos=self.low_cmd_qos)
        self.low_cmd_subscriber = Subscriber(self.low_cmd_participant)
        self.low_cmd_reader = DataReader(self.low_cmd_subscriber, self.low_cmd_topic)

        self.last_read_time = time.time()

    def read(self) -> Optional[LowCmd_]:
        try:
            msg = self.low_cmd_reader.read()
            if len(msg) > 0:
                return msg[-1]
            return unitree_hg_msg_dds__LowCmd_()
        except Exception:
            return unitree_hg_msg_dds__LowCmd_()


class MujocoLowStatePublisher:
    def __init__(self):
        self.low_state_msg = unitree_hg_msg_dds__LowState_()
        self.low_state_participant = DomainParticipant(domain_id=0)
        self.low_state_qos = Qos(
            Policy.History.KeepLast(depth=1),
        )
        self.low_state_topic = Topic(self.low_state_participant, TOPIC_MUJOCO_LOW_STATE, LowState_, qos=self.low_state_qos)
        self.low_state_publisher = Publisher(self.low_state_participant)
        self.low_state_writer = DataWriter(self.low_state_publisher, self.low_state_topic)

    def publish(self):
        self.low_state_writer.write(self.low_state_msg)


class MujocoLowStateSubscriber:
    def __init__(self):
        self.low_state_participant = DomainParticipant(domain_id=0)
        self.low_state_qos = Qos(
            Policy.History.KeepLast(depth=1),
        )
        self.low_state_topic = Topic(self.low_state_participant, TOPIC_MUJOCO_LOW_STATE, LowState_, qos=self.low_state_qos)
        self.low_state_subscriber = Subscriber(self.low_state_participant)
        self.low_state_reader = DataReader(self.low_state_subscriber, self.low_state_topic)

    def read(self) -> Optional[LowState_]:
        try:
            msg = self.low_state_reader.read()
            if len(msg) > 0:
                return msg[-1]
            return unitree_hg_msg_dds__LowState_()
        except Exception:
            return unitree_hg_msg_dds__LowState_()


@dataclass
@annotate.final
@annotate.autoid("sequential")
class HandCmd_(idl.IdlStruct, typename="unitree_hg.msg.dds_.HandCmd_"):
    motor_cmd: types.sequence[MotorCmd_]
    reserve: types.array[types.uint32, 4]

@dataclass
@annotate.final
@annotate.autoid("sequential")
class PressSensorState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.PressSensorState_"):
    pressure: types.array[types.float32, 12]
    temperature: types.array[types.float32, 12]
    lost: types.uint32
    reserve: types.uint32

@dataclass
@annotate.final
@annotate.autoid("sequential")
class HandState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.HandState_"):
    motor_state: types.sequence[MotorState_]
    press_sensor_state: types.sequence[PressSensorState_]
    imu_state: IMUState_
    power_v: types.float32
    power_a: types.float32
    system_v: types.float32
    device_v: types.float32
    error: types.array[types.uint32, 2]
    reserve: types.array[types.uint32, 2]

def unitree_hg_msg_dds__HandCmd_():
    return HandCmd_([unitree_hg_msg_dds__MotorCmd_() for i in range(7)], [0, 0, 0, 0])

def unitree_hg_msg_dds__PressSensorState_():
    return PressSensorState_([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0, 0)

def unitree_hg_msg_dds__HandState_():
    return HandState_([unitree_hg_msg_dds__MotorState_() for i in range(7)], 
                        [unitree_hg_msg_dds__PressSensorState_() for i in range(7)],
                         unitree_hg_msg_dds__IMUState_(), 
                         0.0, 0.0, 0.0, 0.0, [0, 0], [0, 0])

from sim_res.mujoco.eman.sim2.topic import TOPIC_MUJOCO_LEFTHAND_CMD, TOPIC_MUJOCO_LEFTHAND_STATE, TOPIC_MUJOCO_RIGHTHAND_CMD, TOPIC_MUJOCO_RIGHTHAND_STATE

class MujocoLeftHandCmdPublisher:
    def __init__(self):
        self.hand_cmd_msg = unitree_hg_msg_dds__HandCmd_()

        self.hand_cmd_participant = DomainParticipant(domain_id=0)
        self.hand_cmd_qos = Qos(
            Policy.History.KeepLast(depth=1),
        )
        self.hand_cmd_topic = Topic(self.hand_cmd_participant, TOPIC_MUJOCO_LEFTHAND_CMD, HandCmd_, qos=self.hand_cmd_qos)
        self.hand_cmd_publisher = Publisher(self.hand_cmd_participant)
        self.hand_cmd_writer = DataWriter(self.hand_cmd_publisher, self.hand_cmd_topic)

    def publish(self):
        self.hand_cmd_writer.write(self.hand_cmd_msg)

class MujocoLeftHandCmdSubscriber:
    def __init__(self):
        self.hand_cmd_participant = DomainParticipant(domain_id=0)
        self.hand_cmd_qos = Qos(
            Policy.History.KeepLast(depth=1),
        )
        self.hand_cmd_topic = Topic(self.hand_cmd_participant, TOPIC_MUJOCO_LEFTHAND_CMD, HandCmd_, qos=self.hand_cmd_qos)
        self.hand_cmd_subscriber = Subscriber(self.hand_cmd_participant)
        self.hand_cmd_reader = DataReader(self.hand_cmd_subscriber, self.hand_cmd_topic)

    def read(self) -> Optional[HandCmd_]:
        try:
            msg = self.hand_cmd_reader.read()
            if len(msg) > 0:
                return msg[-1]
            return unitree_hg_msg_dds__HandCmd_()
        except Exception:
            return unitree_hg_msg_dds__HandCmd_()

class MujocoLeftHandStatePublisher:
    def __init__(self):
        self.hand_state_msg = unitree_hg_msg_dds__HandState_()

        self.hand_state_participant = DomainParticipant(domain_id=0)
        self.hand_state_qos = Qos(
            Policy.History.KeepLast(depth=1),
        )
        self.hand_state_topic = Topic(self.hand_state_participant, TOPIC_MUJOCO_LEFTHAND_STATE, HandState_, qos=self.hand_state_qos)
        self.hand_state_publisher = Publisher(self.hand_state_participant)
        self.hand_state_writer = DataWriter(self.hand_state_publisher, self.hand_state_topic)

    def publish(self):
        self.hand_state_writer.write(self.hand_state_msg)

class MujocoLeftHandStateSubscriber:
    def __init__(self):
        self.hand_state_participant = DomainParticipant(domain_id=0)
        self.hand_state_qos = Qos(
            Policy.History.KeepLast(depth=1),
        )
        self.hand_state_topic = Topic(self.hand_state_participant, TOPIC_MUJOCO_LEFTHAND_STATE, HandState_, qos=self.hand_state_qos)
        self.hand_state_subscriber = Subscriber(self.hand_state_participant)
        self.hand_state_reader = DataReader(self.hand_state_subscriber, self.hand_state_topic)

    def read(self) -> Optional[HandState_]:
        try:
            msg = self.hand_state_reader.read()
            if len(msg) > 0:
                return msg[-1]
            return unitree_hg_msg_dds__HandState_()
        except Exception:
            return unitree_hg_msg_dds__HandState_()

class MujocoRightHandCmdPublisher:
    def __init__(self):
        self.hand_cmd_msg = unitree_hg_msg_dds__HandCmd_()

        self.hand_cmd_participant = DomainParticipant(domain_id=0)
        self.hand_cmd_qos = Qos(
            Policy.History.KeepLast(depth=1),
        )
        self.hand_cmd_topic = Topic(self.hand_cmd_participant, TOPIC_MUJOCO_RIGHTHAND_CMD, HandCmd_, qos=self.hand_cmd_qos)
        self.hand_cmd_publisher = Publisher(self.hand_cmd_participant)
        self.hand_cmd_writer = DataWriter(self.hand_cmd_publisher, self.hand_cmd_topic)

    def publish(self):
        self.hand_cmd_writer.write(self.hand_cmd_msg)

class MujocoRightHandCmdSubscriber:
    def __init__(self):
        self.hand_cmd_participant = DomainParticipant(domain_id=0)
        self.hand_cmd_qos = Qos(
            Policy.History.KeepLast(depth=1),
        )
        self.hand_cmd_topic = Topic(self.hand_cmd_participant, TOPIC_MUJOCO_RIGHTHAND_CMD, HandCmd_, qos=self.hand_cmd_qos)
        self.hand_cmd_subscriber = Subscriber(self.hand_cmd_participant)
        self.hand_cmd_reader = DataReader(self.hand_cmd_subscriber, self.hand_cmd_topic)

    def read(self) -> Optional[HandCmd_]:
        try:
            msg = self.hand_cmd_reader.read()
            if len(msg) > 0:
                return msg[-1]
            return unitree_hg_msg_dds__HandCmd_()
        except Exception:
            return unitree_hg_msg_dds__HandCmd_()

class MujocoRightHandStatePublisher:
    def __init__(self):
        self.hand_state_msg = unitree_hg_msg_dds__HandState_()

        self.hand_state_participant = DomainParticipant(domain_id=0)
        self.hand_state_qos = Qos(
            Policy.History.KeepLast(depth=1),
        )
        self.hand_state_topic = Topic(self.hand_state_participant, TOPIC_MUJOCO_RIGHTHAND_STATE, HandState_, qos=self.hand_state_qos)
        self.hand_state_publisher = Publisher(self.hand_state_participant)
        self.hand_state_writer = DataWriter(self.hand_state_publisher, self.hand_state_topic)

    def publish(self):
        self.hand_state_writer.write(self.hand_state_msg)

class MujocoRightHandStateSubscriber:
    def __init__(self):
        self.hand_state_participant = DomainParticipant(domain_id=0)
        self.hand_state_qos = Qos(
            Policy.History.KeepLast(depth=1),
        )
        self.hand_state_topic = Topic(self.hand_state_participant, TOPIC_MUJOCO_RIGHTHAND_STATE, HandState_, qos=self.hand_state_qos)
        self.hand_state_subscriber = Subscriber(self.hand_state_participant)
        self.hand_state_reader = DataReader(self.hand_state_subscriber, self.hand_state_topic)

    def read(self) -> Optional[HandState_]:
        try:
            msg = self.hand_state_reader.read()
            if len(msg) > 0:
                return msg[-1]
            return unitree_hg_msg_dds__HandState_()
        except Exception:
            return unitree_hg_msg_dds__HandState_()


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Keypoints(idl.IdlStruct, typename="mujoco.visualization.keypoints"):
    keypoints: types.array[types.float32, 90]

def mujoco_visualization_keypoints_():
    return Keypoints([0.] * 90)

from sim_res.mujoco.eman.sim2.topic import TOPIC_MUJOCO_VISUALIZATION_KEYPOINTS

class MujocoKeypointsPublisher:
    def __init__(self):
        self.keypoints_msg = mujoco_visualization_keypoints_()

        self.keypoints_participant = DomainParticipant(domain_id=1)
        self.keypoints_qos = Qos(
            Policy.History.KeepLast(depth=1),
        )
        self.keypoints_topic = Topic(self.keypoints_participant, TOPIC_MUJOCO_VISUALIZATION_KEYPOINTS, Keypoints, qos=self.keypoints_qos)
        self.keypoints_publisher = Publisher(self.keypoints_participant)
        self.keypoints_writer = DataWriter(self.keypoints_publisher, self.keypoints_topic)

    def publish(self, keypoints):
        keypoints = keypoints.reshape(-1, 3)
        zero_keypoints = np.zeros((30, 3), dtype=np.float32)
        zero_keypoints[:len(keypoints), :] = keypoints.copy()
        self.keypoints_msg.keypoints = zero_keypoints.flatten().tolist()
        self.keypoints_writer.write(self.keypoints_msg)
    

class MujocoKeypointsSubscriber:
    def __init__(self):
        self.keypoints_participant = DomainParticipant(domain_id=1)
        self.keypoints_qos = Qos(
            Policy.History.KeepLast(depth=1),
        )
        self.keypoints_topic = Topic(self.keypoints_participant, TOPIC_MUJOCO_VISUALIZATION_KEYPOINTS, Keypoints, qos=self.keypoints_qos)
        self.keypoints_subscriber = Subscriber(self.keypoints_participant)
        self.keypoints_reader = DataReader(self.keypoints_subscriber, self.keypoints_topic)

    def read(self) -> Optional[Keypoints]:
        try:
            msg = self.keypoints_reader.read()
            if len(msg) > 0:
                return msg[-1]
            return Keypoints([0.] * 90)
        except Exception:
            return Keypoints([0.] * 90)
        



from sim_res.mujoco.eman.sim2.topic import TOPIC_MUJOCO_VIDEO_FILENAME

@dataclass
@annotate.final
@annotate.autoid("sequential")
class VideoFilenameMsg(idl.IdlStruct, typename="mujoco.video.filename"):
    filename: str
    timestamp: types.int64


def mujoco_video_filename_msg_():
    return VideoFilenameMsg("", 0)


class MujocoVideoFilenamePublisher:
    def __init__(self):
        self.msg = mujoco_video_filename_msg_()

        self.participant = DomainParticipant(domain_id=1)
        self.qos = Qos(
            Policy.History.KeepLast(depth=1),
        )
        self.topic = Topic(
            self.participant,
            TOPIC_MUJOCO_VIDEO_FILENAME,
            VideoFilenameMsg,
            qos=self.qos,
        )
        self.publisher = Publisher(self.participant)
        self.writer = DataWriter(self.publisher, self.topic)

    def publish(self, filename: str, timestamp: int = 0):
        self.msg.filename = filename
        self.msg.timestamp = int(timestamp)
        self.writer.write(self.msg)


class MujocoVideoFilenameSubscriber:
    def __init__(self):
        self.participant = DomainParticipant(domain_id=1)
        self.qos = Qos(
            Policy.History.KeepLast(depth=1),
        )
        self.topic = Topic(
            self.participant,
            TOPIC_MUJOCO_VIDEO_FILENAME,
            VideoFilenameMsg,
            qos=self.qos,
        )
        self.subscriber = Subscriber(self.participant)
        self.reader = DataReader(self.subscriber, self.topic)

        self._last_filename = ""
        self._last_timestamp = 0

    def read(self) -> Optional[VideoFilenameMsg]:
        try:
            msg = self.reader.read()
            if len(msg) > 0:
                return msg[-1]
            return None
        except Exception:
            return None

    def read_new_filename(self) -> Optional[str]:
        """
        只在收到“新的文件名消息”时返回文件名；
        否则返回 None。
        """
        msg = self.read()
        if msg is None:
            return None

        filename = (msg.filename or "").strip()
        timestamp = int(msg.timestamp)

        # 空字符串视为“没有要求保存”
        if filename == "":
            return None

        # 用 timestamp + filename 做一次简单去重
        if filename == self._last_filename and timestamp == self._last_timestamp:
            return None

        self._last_filename = filename
        self._last_timestamp = timestamp
        return filename






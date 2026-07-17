
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
class MotorCmd_(idl.IdlStruct, typename="o1_agile.msg.dds_.MotorCmd_"):
    mode: types.uint32
    q: types.float32
    dq: types.float32
    tau: types.float32
    kp: types.float32
    kd: types.float32


@dataclass
@annotate.final
@annotate.autoid("sequential")
class LowCmd_(idl.IdlStruct, typename="o1_agile.msg.dds_.LowCmd_"):
    mode_pr: types.uint8
    mode_machine: types.uint8
    sequences: types.uint16
    motor_cmd: types.array[MotorCmd_, 32]
    crc: types.uint32

@dataclass
@annotate.final
@annotate.autoid("sequential")
class IMUState_(idl.IdlStruct, typename="o1_agile.msg.dds_.IMUState_"):
    accelerometer: types.array[types.float32, 3]
    gyroscope: types.array[types.float32, 3]
    quaternion: types.array[types.float32, 4]
    rpy: types.array[types.float32, 3]
    temperature: types.float32
    timestamp: types.uint64

@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorState_(idl.IdlStruct, typename="o1_agile.msg.dds_.MotorState_"):
    mode: types.uint8
    err_code: types.uint8
    resv: types.array[types.uint8, 2]
    q: types.float32
    dq: types.float32
    tau_est: types.float32
    temperature: types.int32

@dataclass
@annotate.final
@annotate.autoid("sequential")
class LowState_(idl.IdlStruct, typename="o1_agile.msg.dds_.LowState_"):
    mode_pr: types.uint8
    mode_machine: types.uint8
    sequences: types.uint16
    motor_state: types.array[MotorState_, 32]
    imu_state: IMUState_
    wireless_remote: types.array[types.uint8, 40]
    crc: types.uint32

def o1_agile_msg_dds__IMUState_():
    return IMUState_([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, 0)

def o1_agile_msg_dds__BmsState_():
    return BmsState_([0, 0, 0], 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0)

def o1_agile_msg_dds__MotorCmd_():
    return MotorCmd_(0, 0.0, 0.0, 0.0, 0.0, 0.0)

def o1_agile_msg_dds__LowCmd_():
    return LowCmd_(0, 0, 0, [o1_agile_msg_dds__MotorCmd_() for i in range(32)], 0)

def o1_agile_msg_dds__MotorState_():
    return MotorState_(0, 0, [0, 0], 0.0, 0.0, 0.0, 0)

def o1_agile_msg_dds__LowState_():
    return LowState_(0, 0, 0, [o1_agile_msg_dds__MotorState_() for i in range(32)], o1_agile_msg_dds__IMUState_(),
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0)


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
        self.low_cmd_msg = o1_agile_msg_dds__LowCmd_()

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
            return o1_agile_msg_dds__LowCmd_()
        except Exception:
            return o1_agile_msg_dds__LowCmd_()


class MujocoLowStatePublisher:
    def __init__(self):
        self.low_state_msg = o1_agile_msg_dds__LowState_()
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
            return o1_agile_msg_dds__LowState_()
        except Exception:
            return o1_agile_msg_dds__LowState_()


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
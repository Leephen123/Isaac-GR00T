import contextlib
import datetime
import json
import time
from functools import lru_cache
from pathlib import Path
import yaml
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .log import get_logger
from data_res.dds import (
    BODY_POSE_Q_SIZE,
    WR_GAE_BodyPose_Msg,
    MOCAP_NUM_JOINTS,
    MOCAP_POS_DIM,
    MOCAP_QUAT_DIM,
    BODY_POSE_WXYZ_SIZE,
)
from data_res.transforms import (
    quaternion_to_rotation_6d,
    compute_relative,
    compute_imu_relative,
)

logger = get_logger(__name__)


CAMERAS_MAP = {
    "front_head": "ego_view",
    "left_hand": "left_wrist_view",
    "right_hand": "right_wrist_view",
}

SELECT_11_INDICES = [
    0,  # pelvis/root,       original 29 index: root/floating base
    2,  # left_knee_link,    original 29 index: 3
    3,  # left_foot_link,    original 29 index: 5
    6,  # right_knee_link,   original 29 index: 9
    7,  # right_foot_link,   original 29 index: 11
    9,  # left_shoulder_link, original 29 index: 16
    10,  # left_elbow_link,    original 29 index: 18
    11,  # left_wrist/hand,    original 29 index: 21
    12,  # right_shoulder_link, original 29 index: 23
    13,  # right_elbow_link,    original 29 index: 25
    14,  # right_wrist/hand,    original 29 index: 28
]

UNSELECT_4_INDICES = [
    1,  # left_hip_roll,      original 29 index: 1
    4,  # left_foot_link,     original 29 index: 5, repeated foot point
    5,  # right_hip_roll,     original 29 index: 7
    8,  # right_foot_link,    original 29 index: 11, repeated foot point
]

I18N = {
    "en": {
        "episode": "Episode",
        "action": "Action Count",
        "image": "Image Count",
        "fps": "FPS Target",
        "ratio": "Image Ratio",
        "data": "Data Len",
        "status": "Status",
        "recording": "RECORDING",
        "idle": "IDLE",
        "time": "Time",
    },
    "zh": {
        "episode": "回合",
        "action": "动作数",
        "image": "图片数",
        "fps": "目标帧率",
        "ratio": "采样比例",
        "data": "数据量",
        "status": "状态",
        "recording": "采集中",
        "idle": "空闲",
        "time": "时间",
    },
}


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@lru_cache(maxsize=8)
def _load_font(font_size):
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    try:
        return ImageFont.truetype(font_path, font_size)
    except OSError:
        return ImageFont.load_default()


def put_text_cn(img, text, pos, font_size=24, color=(255, 255, 255)):
    """
    Draw Chinese text on image using PIL (supports CJK fonts).

    Args:
        img: OpenCV image (numpy array)
        text: Text to draw (supports Chinese)
        pos: Position tuple (x, y)
        font_size: Font size in pixels
        color: RGB color tuple

    Returns:
        Modified image with text drawn
    """
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = _load_font(font_size)
    draw.text(pos, text, font=font, fill=color)
    return np.array(img_pil)


def render_ui(
    lang,
    episode_count,
    state_count,
    img_count,
    state_fps,
    img_save_stride,
    data_len,
    real_fps,
    mode,
):
    ui = np.zeros((400, 800, 3), np.uint8)
    t = datetime.datetime.now().strftime("%H:%M:%S")
    L = I18N.get(lang, I18N["zh"])

    ui = put_text_cn(ui, f"{L['time']}: {t}", (20, 40))
    ui = put_text_cn(ui, f"{L['episode']}: {episode_count}", (20, 80))
    ui = put_text_cn(ui, f"{L['action']}: {state_count}", (20, 120))
    ui = put_text_cn(ui, f"{L['image']}: {img_count}", (20, 160))
    ui = put_text_cn(ui, f"{L['fps']}: {state_fps}", (20, 200))
    ui = put_text_cn(ui, f"{L['ratio']}: 1/{img_save_stride}", (20, 240))
    ui = put_text_cn(ui, f"{L['data']}: {data_len}", (20, 280))
    ui = put_text_cn(ui, f"Real FPS: {real_fps:.2f}", (20, 320))

    status = L["recording"] if mode == "COLLECTING" else L["idle"]
    ui = put_text_cn(ui, f"{L['status']}: {status}", (20, 350), color=(0, 0, 255))
    ui = put_text_cn(ui, f"Lang: {lang} (L切换)", (500, 40), color=(255, 255, 0))
    return ui


@contextlib.contextmanager
def profile_time(name, enabled=False):
    if not enabled:
        yield
        return

    start_time = None
    try:
        start_time = time.perf_counter_ns()
        yield
    finally:
        if start_time is not None:
            elapsed_time_ms = (time.perf_counter_ns() - start_time) / 1e6
            logger.debug(f"[PROFILE] {name}: {elapsed_time_ms:.2f} ms")


def load_mocap_imu_data(
    path: Path, npz_key: str = "all_data", downsample_rate: int = 1
) -> np.ndarray | None:
    assert downsample_rate > 0, "downsample_rate must be a positive integer"
    if path.suffix == ".npy":
        return np.load(path)

    if path.suffix == ".npz":
        with np.load(path) as data:
            return data[npz_key]

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        return (
            np.asarray(
                [
                    np.asarray(step["mocap"], dtype=np.float32)
                    for index, step in enumerate(records)
                    if index % downsample_rate == 0
                ]
            ),
            np.asarray(
                [
                    step["imu"]
                    for index, step in enumerate(records)
                    if index % downsample_rate == 0
                ],
                dtype=np.float32,
            ),
        )

    return None


def standardize_mocap(data: np.ndarray) -> np.ndarray:
    expected_joints = MOCAP_NUM_JOINTS
    expected_pose_dim = MOCAP_POS_DIM + MOCAP_QUAT_DIM
    if (
        data.ndim != 3
        or data.shape[1] != expected_joints
        or data.shape[2] != expected_pose_dim
    ):
        raise ValueError(
            f"Expected mocap data shape (T, {expected_joints}, {expected_pose_dim}), got {data.shape}"
        )

    num_frames, num_joints, pose_dim = data.shape
    reference = data[0, 0].copy()

    reference_tiled = np.tile(reference, (num_frames * num_joints, 1))
    data_flat = data.reshape(num_frames * num_joints, pose_dim)
    relative_flat = compute_relative(reference_tiled, data_flat)

    relative = relative_flat.reshape(num_frames, num_joints, pose_dim)
    return relative


def standardize_imu(data: np.ndarray) -> np.ndarray:
    expected_pose_dim = BODY_POSE_WXYZ_SIZE
    if data.ndim != 2 or data.shape[1] != expected_pose_dim:
        raise ValueError(
            f"Expected mocap data shape (T, {expected_pose_dim}), got {data.shape}"
        )

    # remove the yaw angle to align with mocap data
    relative = compute_imu_relative(data, data)
    return relative


def load_image_from_path_or_array(
    image_input: Path | np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    debug: bool = False,
    name: str | None = None,
) -> np.ndarray:
    if isinstance(image_input, Path):
        image = Image.open(image_input).convert("RGB")

    elif isinstance(image_input, np.ndarray):
        image = np.asarray(image_input)

        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected frame shape (H, W, 3), got {image.shape}")

        image = image.astype(np.uint8)

        image = image[:, :, ::-1]
        image = Image.fromarray(image).convert("RGB")

    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    if debug:
        os.makedirs("./image_debug", exist_ok=True)
        save_name = name if name is not None else "debug_image"
        save_path = os.path.join("./image_debug", f"{save_name}.png")
        image.save(save_path)
        print(f"image save in {save_path}")

    resized_image = image.resize(target_size, Image.BILINEAR)
    result = np.array(resized_image)

    return result[None, None, ...]


def _history_state_to_imu_joints(
    history_state, history_len: int | None = None
) -> np.ndarray:
    history_state = np.asarray(history_state, dtype=np.float32)
    body_joint_dim = BODY_POSE_Q_SIZE
    raw_imu_dim = BODY_POSE_WXYZ_SIZE
    rot6d_imu_dim = 6
    raw_expected_dim = body_joint_dim + raw_imu_dim
    rot6d_expected_dim = body_joint_dim + rot6d_imu_dim

    if history_state.ndim != 2 or history_state.shape[1] not in (
        raw_expected_dim,
        rot6d_expected_dim,
    ):
        raise ValueError(
            f"history_state shape error: expected (T, {raw_expected_dim}) "
            f"as [imu_wxyz({raw_imu_dim}) | body_joint({body_joint_dim})], "
            f"or (T, {rot6d_expected_dim}) as [imu_6d({rot6d_imu_dim}) | "
            f"body_joint({body_joint_dim})], got {history_state.shape}"
        )

    if history_len is not None:
        if history_len <= 0:
            raise ValueError(f"history_len must be positive, got {history_len}")
        if history_state.shape[0] > history_len:
            history_state = history_state[-history_len:]
        elif 0 < history_state.shape[0] < history_len:
            pad_count = history_len - history_state.shape[0]
            history_state = np.concatenate(
                [np.repeat(history_state[:1], pad_count, axis=0), history_state],
                axis=0,
            )

    if history_state.shape[1] == rot6d_expected_dim:
        return history_state.astype(np.float32, copy=False)

    imu = standardize_imu(history_state[:, :raw_imu_dim])
    imu_6d = quaternion_to_rotation_6d(imu)
    body_joint = history_state[:, raw_imu_dim:]
    return np.concatenate([imu_6d, body_joint], axis=1).astype(np.float32)


def format_history_state_for_observation(
    state: np.ndarray,
    state_horizon: int | None = None,
) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 2:
        raise ValueError(f"Expected state history with shape (T, D), got {state.shape}")

    if state_horizon is None:
        state_horizon = state.shape[0]
    if state_horizon <= 0:
        raise ValueError(f"state_horizon must be positive, got {state_horizon}")

    if state_horizon == 1:
        return state.reshape(1, 1, -1).astype(np.float32)

    if state.shape[0] > state_horizon:
        state = state[-state_horizon:]
    elif state.shape[0] < state_horizon:
        pad_count = state_horizon - state.shape[0]
        state = np.concatenate(
            [np.repeat(state[:1], pad_count, axis=0), state],
            axis=0,
        )
    return state[None, ...].astype(np.float32)


def _build_history_video(frame=None, debug: bool = False) -> dict[str, np.ndarray]:
    video = {
        "ego_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
        "left_wrist_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
        "right_wrist_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
    }

    if frame is None:
        return video

    for name, image in frame.items():
        video[CAMERAS_MAP[name]] = load_image_from_path_or_array(
            np.asarray(image), (256, 256), debug, name
        )
    return video


def build_observation_from_msg_with_history(
    history_state,
    task_description: str,
    frame=None,
    history_len: int = 50,
    state_horizon: int | None = None,
    debug: bool = False,
):
    state = _history_state_to_imu_joints(history_state, history_len=history_len)
    video = _build_history_video(frame=frame, debug=debug)

    observation = {
        "video": video,
        "state": {
            "imu_joints": format_history_state_for_observation(
                state, state_horizon=state_horizon
            )
        },
        "language": {"annotation.human.task_description": [[task_description]]},
    }

    stickman_np = np.zeros((1, 1, 900), dtype=np.float32)
    observation["stickman"] = {"annotation.human.stickman": stickman_np}
    return observation


def build_observation_from_msg_rtc(
    history_state,
    task_description: str,
    frame=None,
    delay_frames: int = 6,
    action_executed_steps: int | None = None,
    history_len: int = 50,
    state_horizon: int | None = None,
    debug: bool = False,
):
    state = _history_state_to_imu_joints(history_state, history_len=history_len)
    video = _build_history_video(frame=frame, debug=debug)
    delay_frames = max(int(delay_frames), 0)

    observation = {
        "video": video,
        "state": {
            "imu_joints": format_history_state_for_observation(
                state, state_horizon=state_horizon
            )
        },
        "language": {"annotation.human.task_description": [[task_description]]},
    }
    if action_executed_steps is not None:
        observation.update(
            {
                "delay_frames": np.asarray(delay_frames, dtype=np.int32),
                "action_executed_steps": np.asarray(action_executed_steps, dtype=np.int32),
            }
        )

    stickman_np = np.zeros((1, 1, 900), dtype=np.float32)
    observation["stickman"] = {"annotation.human.stickman": stickman_np}
    return observation


def build_observation_from_msg(
    msg: WR_GAE_BodyPose_Msg,
    task_description: str,
    frame=None,
    use_stickman: bool = False,
    debug: bool = True,
):
    q_np = np.array(msg.q, dtype=np.float32)
    imu_np = np.array(msg.wxyz, dtype=np.float32)
    imu_np = standardize_imu(imu_np[None, :])
    imu_np = quaternion_to_rotation_6d(imu_np)[0]

    if q_np.shape != (29,):
        raise ValueError(f"q shape error: expected (29,), got {q_np.shape}")
    if imu_np.shape != (6,):
        raise ValueError(f"wxyz shape error: expected (6,), got {imu_np.shape}")

    left_leg = q_np[0:6]
    right_leg = q_np[6:12]
    waist = q_np[12:15]
    left_arm = q_np[15:22]
    right_arm = q_np[22:29]

    base_rotation = imu_np
    state = {
        "base_rotation": base_rotation[None, None, :].astype(np.float32),
        "left_leg": left_leg[None, None, :].astype(np.float32),
        "right_leg": right_leg[None, None, :].astype(np.float32),
        "waist": waist[None, None, :].astype(np.float32),
        "left_arm": left_arm[None, None, :].astype(np.float32),
        "right_arm": right_arm[None, None, :].astype(np.float32),
    }

    video = {
        "ego_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
        "left_wrist_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
        "right_wrist_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
    }

    if frame is not None:
        for name, image in frame.items():
            video[CAMERAS_MAP[name]] = load_image_from_path_or_array(
                np.asarray(image), (256, 256), debug, name
            )

    observation = {
        "video": video,
        "state": state,
        "language": {"annotation.human.task_description": [[task_description]]},
    }

    if use_stickman:
        stickman_np = np.zeros((1, 1, 900), dtype=np.float32)
        observation["stickman"] = {"annotation.human.stickman": stickman_np}
    return observation


def get_camera_name(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if "cameras" not in config or not isinstance(config["cameras"], dict):
        raise ValueError(
            f"Camera config must contain a 'cameras' mapping: {config_path}"
        )

    camera_name_list = sorted(config["cameras"].keys())
    available = ", ".join(camera_name_list) if camera_name_list else "none"
    print(f"Available camera names: {available}")
    return camera_name_list

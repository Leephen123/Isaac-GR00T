import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import Array
from pathlib import Path
import yaml

# third-party
import cv2
import numpy as np
import tyro
import math
from loop_rate_limiters import RateLimiter

# local
from wr.data_res.camera_old import VideoCapture
from data_res.log import get_logger
from data_res.dds import MocapConfig, MocapUE5G115MsgSubscriber
from data_res.utils import NumpyEncoder, profile_time, render_ui

logger = get_logger(__name__)

# robot SDK (not pip-installed)
sys.path.insert(0, "/home/unitree/unitree_sdk2_python")
sys.path.insert(0, "/home/unitree/xr_teleoperate")

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from teleop.robot_control.robot_arm import G1_29_ArmController


@dataclass
class DataCollectorConfig:
    root_path: Path
    """Root directory for saving episodes."""
    task: str
    """Task description written into each data record."""
    state_fps: int = 120
    """Robot state data collection frequency (Hz)."""
    img_save_fps: int = 20
    """Image saving frequency (Hz)."""
    debug: bool = False
    """Print per-frame timing info for debugging."""
    profile: bool = False
    """Enable profile_time instrumentation for performance measurement."""
    lang: str = "zh"
    """UI display language, 'zh' or 'en'."""
    network_iface: str = "eth0"
    """Network interface for unitree SDK channel (e.g. 'eth0', 'enp3s0')."""
    mocap: MocapConfig = field(default_factory=MocapConfig)
    """Mocap DDS subscriber configuration."""
    camera_config: Path = Path("./config/camera.yaml")
    """Camera config, to get camera names list"""


class DataCollector:
    def __init__(self, cfg: DataCollectorConfig):
        self.state_fps = cfg.state_fps
        self.img_save_stride = cfg.state_fps // cfg.img_save_fps
        self.rate_limiter = RateLimiter(cfg.state_fps)
        self.debug = cfg.debug
        self.profile = cfg.profile
        self.lang = cfg.lang

        self.root_path = cfg.root_path
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.task = cfg.task
        self.camera_names = self._get_camera_name(cfg.camera_config)

        # for ui display and data collection
        self.episode_count = self._init_episode()
        self.data_buffer = []
        self.mode = "WAITING"
        self.state_count = 0
        self.img_count = 0
        self.latest_img = {}
        self.rel_img_path = {}
        self.camera_lock = threading.Lock()
        self.running = True
        self.real_fps = 0.0

        logger.info("1. Init robot")  # TODO: need define more clear interface.
        ChannelFactoryInitialize(0, cfg.network_iface)
        self.body_ctrl = G1_29_ArmController(False, False, False)
        self.hand_ctrl = Array("d", 14, lock=False)

        logger.info("2. Init mocap subscriber")
        try:
            self.mocap_subscriber = MocapUE5G115MsgSubscriber(cfg.mocap)
        except Exception as e:
            logger.warning(f"failed to init mocap subscriber: {e}")
            self.mocap_subscriber = None

        logger.info("3. Init camera")
        self.camera_caps = {name: VideoCapture(name) for name in self.camera_names}
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()

        self.executor = ThreadPoolExecutor(max_workers=4)

    def _get_camera_name(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        if "cameras" not in config or not isinstance(config["cameras"], dict):
            raise ValueError(f"Camera config must contain a 'cameras' mapping: {config_path}")

        camera_name_list = sorted(config["cameras"].keys())
        available = ", ".join(camera_name_list) if camera_name_list else "none"
        print(f"Available camera names: {available}")
        return camera_name_list

    def _toggle_lang(self):
        self.lang = "en" if self.lang == "zh" else "zh"
        logger.info(f"language -> {self.lang}")

    def _init_episode(self):
        max_id = -1
        for item in self.root_path.iterdir():
            if item.name.startswith("episode_"):
                try:
                    idx = int(item.name.split("_")[1])
                    max_id = max(max_id, idx)
                except (ValueError, IndexError):
                    pass
        return max_id + 1

    def _camera_loop(self):
        while self.running:
            for name, capture in self.camera_caps.items():
                ret, frame = capture.read()
                if ret:
                    with self.camera_lock:
                        self.latest_img[name] = frame.copy()

    def _collect_mocap_data(self):
        assert self.mocap_subscriber is not None
        data = self.mocap_subscriber.get_data()
        assert data is not None
        return data

    def _save_json(self):
        if not self.data_buffer:
            return

        path = self.episode_path / "data.json"
        with open(path, "w") as f:
            json.dump(self.data_buffer, f, cls=NumpyEncoder, indent=4)

    def _start(self):
        logger.info(f"Starting episode_{self.episode_count} ...")
        self.episode_path = self.root_path / f"episode_{self.episode_count}"
        self.episode_path.mkdir(parents=True, exist_ok=True)
        for name in self.camera_names:
            (self.episode_path / "images" / name).mkdir(parents=True, exist_ok=True)

        self.data_buffer = []
        self.state_count = 0
        self.img_count = 0
        self.img_time_last_s = None
        self.mode = "COLLECTING"

    def _stop(self):
        self.mode = "WAITING"
        self._save_json()
        self.episode_count = self._init_episode()
        logger.info(f"Saved episode_{self.episode_count - 1}")

    def _delete_last(self):
        last_episode = self.root_path / f"episode_{self.episode_count - 1}"
        if last_episode.exists():
            shutil.rmtree(last_episode)
        self.episode_count = self._init_episode()

    def _collect_data_only(self):
        with profile_time("collect_data: proprioception sensor data", enabled=self.profile):
            imu = np.array(self.body_ctrl.get_base_orientation_quat())
            body_joint = np.array(self.body_ctrl.get_current_motor_q())[:29]
            hand_joint = np.array(self.hand_ctrl)

        with profile_time("collect_data: image data", enabled=self.profile):
            with self.camera_lock:
                frame = {
                    name: latest_img.copy()
                    for name, latest_img in self.latest_img.items()
                    if latest_img is not None
                }
            if len(frame) == 0:
                logger.warning("frame is None, skipping this iteration")
                return

            if self.state_count % self.img_save_stride == 0:
                for name, img in frame.items():
                    img_dir = self.episode_path / "images" / name
                    img_path = img_dir / f"{self.img_count:05d}.png"
                    self.executor.submit(cv2.imwrite, str(img_path), img)

                    rel_img_path = str(img_path.relative_to(self.episode_path))
                    self.rel_img_path[name] = rel_img_path

                    if self.debug:
                        if self.img_time_last_s is not None:
                            logger.info(
                                f"[IMAGE INTERVAL ms] {(time.time() - self.img_time_last_s) * 1e3:.1f} ms"
                            )
                        self.img_time_last_s = time.time()
                self.img_count += 1

        with profile_time("collect_data: motion capture data", enabled=self.profile):
            mocap = self._collect_mocap_data()

        self.data_buffer.append(
            {
                **self.rel_img_path,
                "body_joint": body_joint,
                "hand_joint": hand_joint,
                "imu": imu,
                "mocap": mocap,
                "task": [self.task],
                "fps": self.real_fps,
                "timestamp": time.time(),
            }
        )
        self.state_count += 1

        if self.debug:
            if len(self.data_buffer) > 2:
                logger.info(
                    f"[ACTION INTERVAL ms] {(self.data_buffer[-1]['timestamp'] - self.data_buffer[-2]['timestamp']) * 1e3:.1f} ms"
                )

    def run(self):
        logger.info("running...")
        should_stop = threading.Event()

        def _ui_loop():
            while not should_stop.is_set():
                ui = render_ui(
                    lang=self.lang,
                    episode_count=self.episode_count,
                    state_count=self.state_count,
                    img_count=self.img_count,
                    state_fps=self.state_fps,
                    img_save_stride=self.img_save_stride,
                    data_len=len(self.data_buffer),
                    real_fps=self.real_fps,
                    mode=self.mode,
                )
                cv2.imshow("DataCollector", ui)

                with self.camera_lock:
                    frame = {
                        name: img.copy() for name, img in self.latest_img.items() if img is not None
                    }

                n_cams = len(frame)
                if n_cams > 0:
                    cell_w, cell_h = 320, 240
                    cols = n_cams
                    rows = 1

                    grid_canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

                    for i, (name, img) in enumerate(frame.items()):
                        r = 0
                        c = i

                        img_resized = cv2.resize(img, (cell_w, cell_h))

                        cv2.putText(
                            img_resized,
                            name,
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                        grid_canvas[
                            r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w
                        ] = img_resized

                    cv2.imshow("Cameras View", grid_canvas)

                key = cv2.pollKey()
                if key != -1:
                    key &= 0xFF

                # keyboard bindings: l: toggle language, 1: start/stop recording, 2: delete last episode, 3: exit
                if key == ord("l"):
                    self._toggle_lang()

                if key == ord("1"):
                    if self.mode == "WAITING":
                        self._start()
                    else:
                        self._stop()

                elif key == ord("2"):
                    if self.mode == "WAITING":
                        self._delete_last()

                elif key == ord("3"):
                    should_stop.set()

                time.sleep(0.03)

        def _sensor_loop():
            count = 0
            start_time = time.perf_counter()

            while not should_stop.is_set():
                with profile_time("sensor_loop: total", enabled=self.profile):
                    curr_time = time.perf_counter()
                    if count == 100:
                        self.real_fps = count / (curr_time - start_time)
                        count = 0
                        start_time = curr_time
                    count += 1

                    if self.mode == "COLLECTING":
                        with profile_time("collect_data: total time", enabled=self.profile):
                            self._collect_data_only()
                    self.rate_limiter.sleep()

        _ui_thread = threading.Thread(target=_ui_loop)
        _sensor_thread = threading.Thread(target=_sensor_loop)

        _ui_thread.start()
        _sensor_thread.start()

        should_stop.wait()
        _ui_thread.join()
        _sensor_thread.join()

        self.running = False
        self.cleanup()

    def cleanup(self):
        self.running = False
        self.executor.shutdown(wait=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = tyro.cli(DataCollectorConfig)
    DataCollector(cfg).run()

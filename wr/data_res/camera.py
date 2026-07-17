from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from time import monotonic, sleep

import cv2
import numpy as np
import yaml

from .log import get_logger

logger = get_logger(__name__)

_DEFAULT_FPS = 20


class CameraGrabber:
    """Background reader that always exposes the latest frame for each camera."""

    def __init__(
        self,
        camera_caps: dict[str, VideoCapture],
        read_retry_s: float = 0.005,
    ) -> None:
        if read_retry_s <= 0:
            raise ValueError(f"read_retry_s must be positive, got {read_retry_s}")
        self.camera_caps = camera_caps
        self.read_retry_s = read_retry_s
        self._latest: dict[str, np.ndarray | None] = {
            name: None for name in camera_caps
        }
        self._captured_at_s: dict[str, float | None] = {
            name: None for name in camera_caps
        }
        self._errors: dict[str, Exception] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self._threads:
            return
        with self._lock:
            for name in self.camera_caps:
                self._latest[name] = None
                self._captured_at_s[name] = None
            self._errors.clear()
        self._stop_event.clear()
        for name, camera_cap in self.camera_caps.items():
            thread = threading.Thread(
                target=self._grab_loop,
                args=(name, camera_cap),
                name=f"camera-grab-{name}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def _grab_loop(self, name: str, camera_cap: VideoCapture) -> None:
        while not self._stop_event.is_set():
            try:
                ret, frame = camera_cap.read()
            except Exception as exc:
                if self._stop_event.is_set():
                    return
                with self._lock:
                    self._errors[name] = exc
                logger.exception("Camera reader failed: %s", name)
                return
            if not ret or frame is None:
                sleep(self.read_retry_s)
                continue
            captured_at_s = monotonic()
            with self._lock:
                self._latest[name] = frame
                self._captured_at_s[name] = captured_at_s

    def wait_until_ready(self, timeout_s: float = 10.0) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")
        deadline = monotonic() + timeout_s
        missing = list(self.camera_caps)
        while monotonic() < deadline:
            with self._lock:
                self._raise_if_failed_locked()
                missing = [
                    name for name, frame in self._latest.items() if frame is None
                ]
            if not missing:
                return
            sleep(0.01)
        raise RuntimeError(f"Cameras produced no frame within {timeout_s}s: {missing}")

    def get_snapshot(self) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Return an atomic frame snapshot and per-camera monotonic timestamps."""
        with self._lock:
            self._raise_if_failed_locked()
            missing = [
                name
                for name, frame in self._latest.items()
                if frame is None or self._captured_at_s[name] is None
            ]
            if missing:
                raise RuntimeError(f"No frame available yet for cameras: {missing}")
            frames = {
                name: frame.copy()
                for name, frame in self._latest.items()
                if frame is not None
            }
            captured_at_s = {
                name: float(timestamp)
                for name, timestamp in self._captured_at_s.items()
                if timestamp is not None
            }
        return frames, captured_at_s

    def get_frames(self) -> dict[str, np.ndarray]:
        frames, _ = self.get_snapshot()
        return frames

    def _raise_if_failed_locked(self) -> None:
        if not self._errors:
            return
        details = ", ".join(
            f"{name}: {type(exc).__name__}: {exc}"
            for name, exc in sorted(self._errors.items())
        )
        raise RuntimeError(f"Camera reader thread failed: {details}")

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        alive_threads = [thread for thread in self._threads if thread.is_alive()]
        if alive_threads:
            logger.warning(
                "Camera reader threads are blocked; releasing devices: %s",
                [thread.name for thread in alive_threads],
            )
            for camera_cap in self.camera_caps.values():
                camera_cap.release()
            for thread in alive_threads:
                thread.join(timeout=1.0)
        self._threads = [thread for thread in alive_threads if thread.is_alive()]
        if self._threads:
            logger.error(
                "Camera reader threads did not stop: %s",
                [thread.name for thread in self._threads],
            )


class VideoCapture:
    """
    OpenCV V4L2 camera capture wrapper.
    """

    def __init__(self, camera_name):
        self.camera_name = camera_name
        self.cap = None
        camera_config = self.get_camera_config(camera_name)
        self.device = camera_config["path"]
        width = camera_config["width"]
        height = camera_config["height"]
        fps = camera_config.get("fps", _DEFAULT_FPS)
        white_balance_automatic = camera_config.get("white_balance_automatic", 1)

        self._set_v4l2_control("white_balance_automatic", white_balance_automatic)

        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.device}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self._warmup(10)
        logger.info(f"[{camera_name}] opened at {self.device} ({width}x{height}@{fps})")

    @staticmethod
    def load_camera_config():
        config_path = Path(__file__).parent.parent / "config" / "camera.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Camera config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        if "cameras" not in config or not isinstance(config["cameras"], dict):
            raise ValueError(
                f"Camera config must contain a 'cameras' mapping: {config_path}"
            )
        return config

    @staticmethod
    def get_camera_config(camera_name):
        cameras = VideoCapture.load_camera_config()["cameras"]
        if camera_name not in cameras:
            available = ", ".join(sorted(cameras)) or "none"
            raise KeyError(
                f"Camera '{camera_name}' is not configured. Available cameras: {available}"
            )

        camera_config = cameras[camera_name]
        missing_keys = {"path", "width", "height"} - camera_config.keys()

        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise KeyError(
                f"Camera '{camera_name}' config is missing required keys: {missing}"
            )

        return camera_config

    def _warmup(self, frames: int = 10):
        """Read a few frames immediately after opening the camera to let it settle."""
        for i in range(frames):
            ret, _ = self.cap.read()

    def _set_v4l2_control(self, name, value):
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", self.device, f"--set-ctrl={name}={value}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError as e:
            logger.warning(f"v4l2-ctl failed to start for {self.device}: {e}")
            return

        if result.returncode != 0:
            logger.warning(f"v4l2-ctl failed for {self.device}: set {name}={value}")

    def read(self):
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        # BGR (OpenCV default)
        return True, frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

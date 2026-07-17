import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from loop_rate_limiters import RateLimiter

from data_res.log import get_logger
from data_res.dds import MocapConfig, MocapUE5G115MsgSubscriber

logger = get_logger(__name__)


@dataclass
class RecorderConfig:
    output_path: Path = Path("received_mocap_data.npy")
    """Output .npy file path."""
    fps: int = 100
    """Sampling frequency (Hz)."""
    duration_s: float = 35.0
    """Recording duration in seconds."""
    mocap: MocapConfig = field(default_factory=MocapConfig)
    """Mocap DDS subscriber configuration."""


def record_mocap(cfg: RecorderConfig):
    subscriber = MocapUE5G115MsgSubscriber(cfg.mocap)
    rate_limiter = RateLimiter(frequency=cfg.fps)
    start_time = time.perf_counter()
    data = []

    logger.info(
        f"Recording mocap data for {cfg.duration_s:.1f}s at {cfg.fps} Hz "
        f"from topic '{cfg.mocap.topic_name}'..."
    )

    while time.perf_counter() - start_time < cfg.duration_s:
        data_frame = subscriber.get_data()
        if data_frame is not None:
            data.append(data_frame)
        rate_limiter.sleep()

    data = np.asarray(data, dtype=np.float32)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cfg.output_path, data)

    logger.info(f"Saved {len(data)} frames to {cfg.output_path}")


if __name__ == "__main__":
    record_mocap(tyro.cli(RecorderConfig))

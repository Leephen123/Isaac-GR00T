import copy
import logging
import sys

_RESET = "\033[0m"
_COLORS = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[35m",  # magenta
}

_CONFIGURED = False


class _ColorFormatter(logging.Formatter):
    def __init__(self, *args, enable_color: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_color = enable_color

    def format(self, record: logging.LogRecord) -> str:
        if not self.enable_color:
            return super().format(record)

        colored_record = copy.copy(record)
        color = _COLORS.get(record.levelno, "")
        colored_record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(colored_record)


def setup_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        _ColorFormatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
            enable_color=sys.stderr.isatty(),
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)

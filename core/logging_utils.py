import logging
import os
import platform
import resource
import sys
import time
from datetime import datetime


def _rss_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return usage / 1024**2
    return usage / 1024


def setup_logger(name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.now().strftime(f"{name}_%Y-%m-%d_%H-%M-%S.log")
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info(f"Инициализация логирования. Файл: {log_filepath}")
    logger.info(f"Python {platform.python_version()} | {platform.platform()}")
    return logger, log_filepath


class Heartbeat:
    def __init__(self, logger, total, label, every_seconds=15.0):
        self.logger = logger
        self.total = total
        self.label = label
        self.every_seconds = every_seconds
        self.start = time.time()
        self.last = self.start
        self.done = 0

    def update(self, n):
        self.done += n
        now = time.time()
        if now - self.last >= self.every_seconds or self.done >= self.total:
            elapsed = now - self.start
            rate = self.done / max(elapsed, 1e-9)
            eta = (self.total - self.done) / max(rate, 1e-9)
            pct = 100.0 * self.done / max(self.total, 1)
            self.logger.info(
                f"{self.label}: {self.done:,}/{self.total:,} ({pct:5.1f}%) | "
                f"{rate:,.0f}/s | прошло {elapsed / 60:.1f} мин | осталось ~{eta / 60:.1f} мин | "
                f"RSS {_rss_mb():.0f} MB"
            )
            self.last = now

    def finish(self):
        elapsed = time.time() - self.start
        self.logger.info(f"{self.label}: готово за {elapsed / 60:.2f} мин | RSS {_rss_mb():.0f} MB")


class StageTimer:
    def __init__(self, logger, label):
        self.logger = logger
        self.label = label

    def __enter__(self):
        self.start = time.time()
        self.logger.info(f">>> {self.label}")
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self.start
        if exc_type is None:
            self.logger.info(f"<<< {self.label} — {elapsed / 60:.2f} мин | RSS {_rss_mb():.0f} MB")
        else:
            self.logger.error(f"<<< {self.label} — УПАЛО через {elapsed / 60:.2f} мин: {exc!r}")
        return False

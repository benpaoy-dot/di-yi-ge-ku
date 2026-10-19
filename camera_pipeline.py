"""Bounded, latest-frame camera pipeline; workers never touch Tk widgets."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable

import cv2


# The order matters for "auto": try the largest common UVC modes first and
# verify the mode with an actual frame because many drivers accept set() while
# silently keeping a lower capture size.
CAMERA_RESOLUTION_PRESETS = (
    ("自动最高", "auto"),
    ("3840×2160 · 4K", "3840x2160"),
    ("2560×1440 · 2K", "2560x1440"),
    ("1920×1080 · 1080p", "1920x1080"),
    ("1440×1080 · 4:3", "1440x1080"),
    ("1280×960", "1280x960"),
    ("1280×720 · 720p", "1280x720"),
    ("640×480 · VGA", "640x480"),
)

_RESOLUTION_SIZES = {
    "3840x2160": (3840, 2160),
    "2560x1440": (2560, 1440),
    "1920x1080": (1920, 1080),
    "1440x1080": (1440, 1080),
    "1280x960": (1280, 960),
    "1280x720": (1280, 720),
    "640x480": (640, 480),
}


@dataclass(frozen=True)
class FramePacket:
    generation: int
    sequence: int
    timestamp: float
    frame: Any


@dataclass(frozen=True)
class RecognitionPacket:
    source: FramePacket
    detections: list
    latency_ms: float


def open_default_capture(index):
    """Open a camera using the Windows backends in a predictable order."""
    last_capture = None
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        try:
            capture = cv2.VideoCapture(index, backend)
        except Exception:
            continue
        if capture.isOpened():
            if last_capture is not None:
                last_capture.release()
            return capture
        if last_capture is not None:
            last_capture.release()
        last_capture = capture
    return last_capture


class CameraPipeline:
    def __init__(self, recognize: Callable, capture_factory=None, interval=0.12):
        self._recognize = recognize
        self._factory = capture_factory or open_default_capture
        self.interval = interval
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._generation = 0
        self._request = None
        self._frame = None
        self._result = None
        self._status = (0, 'offline', '')
        self._threads = []
        self._active_index = 0
        self._active_resolution = None

    def start(self):
        if self._threads:
            return
        self._threads = [
            threading.Thread(target=self._capture_loop, name='camera-capture', daemon=True),
            threading.Thread(target=self._recognition_loop, name='face-recognition', daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def request_camera(self, index, fallback=False, resolution="auto"):
        with self._condition:
            self._generation += 1
            indices = tuple(dict.fromkeys([index, 0, 1, 2])) if fallback else (index,)
            self._request = (self._generation, indices, resolution)
            self._frame = self._result = None
            self._active_resolution = None
            self._status = (self._generation, 'opening', f'摄像头 {index}')
            self._condition.notify_all()
            return self._generation

    def invalidate(self):
        """Invalidate recognition and pending frames after a people-cache change."""
        with self._condition:
            if self._request is None:
                return self._generation
            indices = ((self._active_index,) if self._status[1] in ('online', 'recognition_error')
                       else self._request[1])
            self._generation += 1
            self._request = (self._generation, indices, self._request[2])
            self._frame = self._result = None
            self._active_resolution = None
            self._status = (self._generation, 'opening', '')
            self._condition.notify_all()
            return self._generation

    def latest_frame(self):
        with self._condition:
            packet = self._frame
            if packet is None or time.perf_counter() - packet.timestamp > 1.0:
                return None
            return packet

    def latest_result(self):
        with self._condition:
            return self._result

    def status(self):
        with self._condition:
            return self._status

    def resolution(self):
        with self._condition:
            return self._active_resolution

    def _publish_status(self, generation, state, detail):
        with self._condition:
            if generation == self._generation and not self._stop.is_set():
                if state == 'recognition_error' and (self._frame is None or self._status[1] == 'failed'):
                    return
                self._status = (generation, state, detail)
                if state == 'online':
                    self._active_index = int(detail)
                if state == 'failed':
                    self._frame = self._result = None
                elif state == 'recognition_error':
                    self._result = None
                self._condition.notify_all()

    def _current(self, generation):
        with self._condition:
            return generation == self._generation and not self._stop.is_set()

    def _capture_loop(self):
        capture = None
        active_generation = -1
        sequence = 0
        try:
            while not self._stop.is_set():
                with self._condition:
                    request = self._request
                    if request is None or (request[0] == active_generation and capture is None):
                        self._condition.wait(timeout=0.1)
                        continue
                generation, indices, resolution = request
                if generation != active_generation:
                    if capture is not None:
                        capture.release()
                        capture = None
                    pending_frame = None
                    active_generation = generation
                    for index in indices:
                        if not self._current(generation):
                            break
                        candidate = None
                        try:
                            candidate = self._factory(index)
                            if candidate.isOpened():
                                configured = self._configure_capture(candidate, resolution)
                                if configured is not None:
                                    actual_resolution, pending_frame = configured
                                    capture = candidate
                                    with self._condition:
                                        if generation == self._generation:
                                            self._active_resolution = actual_resolution
                                    self._publish_status(generation, 'online', index)
                                    break
                        except Exception:
                            pass
                        if candidate is not None:
                            candidate.release()
                    if capture is None:
                        self._publish_status(generation, 'failed', '无法打开所选摄像头，请检查连接或选择其他编号')
                        continue
                if not self._current(generation):
                    continue
                if pending_frame is not None:
                    ok, frame = True, pending_frame
                    pending_frame = None
                else:
                    try:
                        ok, frame = capture.read()
                    except Exception:
                        ok, frame = False, None
                if not ok or frame is None:
                    capture.release()
                    capture = None
                    self._publish_status(generation, 'failed', '摄像头读取失败，请重新连接')
                    continue
                sequence += 1
                with self._condition:
                    if generation == self._generation and not self._stop.is_set():
                        self._frame = FramePacket(generation, sequence, time.perf_counter(), frame.copy())
                        self._condition.notify_all()
                self._stop.wait(0.001)
        finally:
            if capture is not None:
                capture.release()

    def _configure_capture(self, capture, resolution):
        """Request a mode and return the actual first frame size and frame."""
        if resolution == "auto":
            candidates = tuple(_RESOLUTION_SIZES.values())
        else:
            candidates = (_RESOLUTION_SIZES.get(resolution, (1280, 720)),)
        best = None
        for width, height in candidates:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, 30)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            try:
                ok, frame = capture.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None or getattr(frame, "ndim", 0) < 2:
                continue
            actual = (int(frame.shape[1]), int(frame.shape[0]))
            candidate = (actual[0] * actual[1], actual, frame)
            if resolution != "auto" or actual == (width, height):
                return actual, frame
            if best is None or candidate[0] > best[0]:
                best = candidate
        return (best[1], best[2]) if best is not None else None

    def _recognition_loop(self):
        last_key = None
        next_run = 0.0
        while not self._stop.is_set():
            with self._condition:
                source = self._frame
                key = (source.generation, source.sequence) if source else None
                delay = next_run - time.perf_counter()
                if source is None or key == last_key or delay > 0:
                    self._condition.wait(timeout=max(0.001, min(0.1, delay)) if delay > 0 else 0.1)
                    continue
                last_key = key
            started = time.perf_counter()
            try:
                detections = self._recognize(source.frame)
            except Exception as error:
                self._publish_status(source.generation, 'recognition_error', f'识别失败：{type(error).__name__}')
                next_run = time.perf_counter() + self.interval
                continue
            finished = time.perf_counter()
            with self._condition:
                if (source.generation == self._generation and not self._stop.is_set()
                        and self._frame is not None
                        and self._status[1] in ('online', 'recognition_error')):
                    self._result = RecognitionPacket(source, detections, (finished - started) * 1000)
                    if self._status[1] == 'recognition_error':
                        self._status = (source.generation, 'online', self._active_index)
            next_run = finished + self.interval

    def stop(self, timeout=0.5):
        self._stop.set()
        with self._condition:
            self._frame = self._result = None
            self._condition.notify_all()
        deadline = time.perf_counter() + timeout
        for thread in self._threads:
            thread.join(max(0, deadline - time.perf_counter()))
        # A hung OS camera call may outlive this bounded wait, but cannot access Tk
        # or publish anything; its owning daemon releases the device on return.

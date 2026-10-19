import threading
import time
import unittest

import cv2
import numpy as np

from camera_pipeline import CameraPipeline


def wait_until(predicate, timeout=2):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError('timed out waiting for pipeline state')


class FakeCapture:
    def __init__(self, index, opened=True):
        self.index = index
        self.opened = opened
        self.released = False
        self.counter = 0
        self.settings = {}

    def isOpened(self):
        return self.opened

    def set(self, *args):
        if len(args) == 2:
            self.settings[args[0]] = args[1]
        return True

    def read(self):
        time.sleep(0.005)
        self.counter += 1
        return True, np.full((8, 8, 3), self.index, np.uint8)

    def release(self):
        self.released = True


class CameraPipelineTests(unittest.TestCase):
    def test_selected_resolution_is_requested_and_actual_size_is_published(self):
        capture = FakeCapture(0)
        pipeline = CameraPipeline(lambda frame: [], lambda index: capture)
        generation = pipeline.request_camera(0, resolution="1920x1080")
        pipeline.start()
        try:
            wait_until(lambda: pipeline.status() == (generation, 'online', 0))
            self.assertEqual(capture.settings[cv2.CAP_PROP_FRAME_WIDTH], 1920)
            self.assertEqual(capture.settings[cv2.CAP_PROP_FRAME_HEIGHT], 1080)
            self.assertEqual(pipeline.resolution(), (8, 8))
        finally:
            pipeline.stop()

    def test_capture_keeps_running_while_recognition_is_blocked(self):
        entered, release = threading.Event(), threading.Event()
        def recognize(frame):
            entered.set()
            release.wait(2)
            return []
        capture = FakeCapture(0)
        pipeline = CameraPipeline(recognize, lambda index: capture)
        pipeline.request_camera(0)
        pipeline.start()
        try:
            self.assertTrue(entered.wait(1))
            sequence = pipeline.latest_frame().sequence
            wait_until(lambda: pipeline.latest_frame().sequence >= sequence + 5)
            self.assertIsNone(pipeline.latest_result())
        finally:
            release.set()
            pipeline.stop()
        self.assertTrue(capture.released)

    def test_switch_releases_old_device_and_discards_inflight_result(self):
        entered, release = threading.Event(), threading.Event()
        captures = []
        def factory(index):
            capture = FakeCapture(index)
            captures.append(capture)
            return capture
        def recognize(frame):
            if frame[0, 0, 0] == 0:
                entered.set()
                release.wait(2)
                return ['old identity']
            return ['new identity']
        pipeline = CameraPipeline(recognize, factory, interval=0.01)
        pipeline.request_camera(0)
        pipeline.start()
        try:
            self.assertTrue(entered.wait(1))
            generation = pipeline.request_camera(1)
            self.assertIsNone(pipeline.latest_frame())
            self.assertIsNone(pipeline.latest_result())
            wait_until(lambda: captures[0].released and pipeline.status() == (generation, 'online', 1))
            release.set()
            wait_until(lambda: pipeline.latest_result() is not None)
            self.assertEqual(pipeline.latest_result().source.generation, generation)
            self.assertEqual(pipeline.latest_result().detections, ['new identity'])
        finally:
            release.set()
            pipeline.stop()
        self.assertTrue(all(capture.released for capture in captures))

    def test_explicit_switch_failure_has_no_silent_fallback_or_old_frames(self):
        calls = []
        def factory(index):
            calls.append(index)
            return FakeCapture(index, opened=False)
        pipeline = CameraPipeline(lambda frame: [], factory)
        generation = pipeline.request_camera(7)
        pipeline.start()
        try:
            wait_until(lambda: pipeline.status()[1] == 'failed')
            self.assertEqual(calls, [7])
            self.assertEqual(pipeline.status()[0], generation)
            self.assertIsNone(pipeline.latest_frame())
            self.assertIsNone(pipeline.latest_result())
        finally:
            pipeline.stop()

    def test_startup_fallback_uses_other_camera(self):
        pipeline = CameraPipeline(lambda frame: [], lambda index: FakeCapture(index, index == 2))
        generation = pipeline.request_camera(0, fallback=True)
        pipeline.start()
        try:
            wait_until(lambda: pipeline.status() == (generation, 'online', 2))
        finally:
            pipeline.stop()

    def test_people_cache_invalidation_keeps_actual_startup_camera(self):
        calls = []
        def factory(index):
            calls.append(index)
            return FakeCapture(index, opened=index == 2)
        pipeline = CameraPipeline(lambda frame: [], factory)
        generation = pipeline.request_camera(0, fallback=True)
        pipeline.start()
        try:
            wait_until(lambda: pipeline.status() == (generation, 'online', 2))
            generation = pipeline.invalidate()
            wait_until(lambda: pipeline.status() == (generation, 'online', 2))
            self.assertEqual(calls, [0, 1, 2, 2])
        finally:
            pipeline.stop()

    def test_read_failure_releases_device_and_clears_result(self):
        capture = FakeCapture(0)
        capture.read = lambda: (False, None)
        pipeline = CameraPipeline(lambda frame: [], lambda index: capture)
        pipeline.request_camera(0)
        pipeline.start()
        try:
            wait_until(lambda: pipeline.status()[1] == 'failed')
            self.assertTrue(capture.released)
            self.assertIsNone(pipeline.latest_frame())
        finally:
            pipeline.stop()

    def test_read_failure_during_inference_cannot_restore_old_identity(self):
        entered, release, fail_read = threading.Event(), threading.Event(), threading.Event()
        capture = FakeCapture(0)
        original_read = capture.read
        def read():
            return (False, None) if fail_read.is_set() else original_read()
        capture.read = read
        def recognize(frame):
            entered.set()
            release.wait(2)
            return ['stale identity']
        pipeline = CameraPipeline(recognize, lambda index: capture)
        pipeline.request_camera(0)
        pipeline.start()
        try:
            self.assertTrue(entered.wait(1))
            fail_read.set()
            wait_until(lambda: pipeline.status()[1] == 'failed')
            release.set()
            wait_until(lambda: pipeline._threads[1].is_alive())
            time.sleep(0.05)
            self.assertIsNone(pipeline.latest_result())
            self.assertEqual(pipeline.status()[1], 'failed')
        finally:
            release.set()
            pipeline.stop()

    def test_shutdown_is_bounded_even_if_driver_is_stuck(self):
        entered, release = threading.Event(), threading.Event()
        capture = FakeCapture(0)
        def read():
            entered.set()
            release.wait(2)
            return True, np.zeros((8, 8, 3), np.uint8)
        capture.read = read
        pipeline = CameraPipeline(lambda frame: [], lambda index: capture)
        pipeline.request_camera(0)
        pipeline.start()
        self.assertTrue(entered.wait(1))
        started = time.perf_counter()
        pipeline.stop(timeout=0.05)
        self.assertLess(time.perf_counter() - started, 0.2)
        release.set()
        wait_until(lambda: capture.released)
        self.assertIsNone(pipeline.latest_frame())


if __name__ == '__main__':
    unittest.main()

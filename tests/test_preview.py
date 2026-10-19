import time
import queue
import unittest
from unittest.mock import Mock
from face_welcome_app import FaceWelcomeApp


class PreviewResponsivenessTests(unittest.TestCase):
    def test_preview_callback_does_not_wait_for_camera_or_inference(self):
        app = FaceWelcomeApp.__new__(FaceWelcomeApp)
        app.running = True
        app.root = Mock()
        app.capture = Mock()
        def slow_read():
            time.sleep(0.08)
            return True, None
        app.capture.read.side_effect = slow_read
        app.process_frame = lambda frame: time.sleep(0.08)
        app.draw_frame = Mock()
        app.last_frame = None
        app._recognition_source = None
        app._speech_events = queue.SimpleQueue()
        app._ensure_workers = Mock()
        app._pipeline = Mock()
        app._pipeline.latest_frame.return_value = None
        app._pipeline.latest_result.return_value = None
        app._pipeline.status.return_value = (0, 'opening', '')
        app._handle_camera_status = Mock()
        app._last_frame_sequence = -1
        app._last_result_sequence = -1
        started = time.perf_counter()
        app.update_frame()
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.05, 'camera read and inference must not block the UI callback')
        app.capture.read.assert_not_called()


if __name__ == '__main__':
    unittest.main()

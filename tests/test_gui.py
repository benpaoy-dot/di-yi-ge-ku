import gc
from pathlib import Path
import tempfile
import time
import tkinter as tk
import unittest

import numpy as np

from camera_pipeline import FramePacket, RecognitionPacket
from face_welcome_app import FaceWelcomeApp, Detection, Person


class GuiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = tk.Tk()
        self.root.withdraw()
        path = str(Path(self.temp.name) / 'check.db')
        class CheckApp(FaceWelcomeApp):
            def _init_database(self):
                self.db_path = path
                super()._init_database()
        self.app = CheckApp(self.root)
        for callback in self.root.tk.call('after', 'info'):
            self.root.after_cancel(callback)

    def tearDown(self):
        self.root.update_idletasks()
        self.app.close()
        gc.collect()
        self.temp.cleanup()

    def test_rgb_preview_and_people_manager_build_without_camera(self):
        self.root.deiconify()
        self.root.update()
        self.root.update_idletasks()
        self.app.draw_frame(np.zeros((480, 640, 3), np.uint8))
        self.assertIsNotNone(self.app.frame_photo)
        self.app.open_people_manager()
        self.root.update_idletasks()
        self.assertTrue(self.app.people_manager_window.winfo_exists())
        self.assertEqual(self.app._pipeline._threads, [])

    def test_reset_clears_old_identity_frames_and_confirmation(self):
        person = Person(99, 'Test', 'T99', 'zh-CN', '')
        self.app.current_person = person
        self.app.last_spoken_person_id = 99
        self.app.last_detections = [Detection((1, 1, 10, 10), 1.0, person, 1.0)]
        self.app.identity_history.append(99)
        self.app.last_frame = np.zeros((10, 10, 3), np.uint8)
        self.app._reset_camera_view()
        self.assertIsNone(self.app.current_person)
        self.assertIsNone(self.app.last_spoken_person_id)
        self.assertIsNone(self.app.last_frame)
        self.assertEqual(self.app.last_detections, [])
        self.assertEqual(list(self.app.identity_history), [])

    def test_stale_result_from_old_camera_is_not_applied(self):
        from unittest.mock import Mock
        self.app._camera_generation = 2
        self.app._camera_state = 'online'
        self.app._pipeline = Mock()
        frame = FramePacket(2, 12, time.perf_counter(), np.zeros((20, 20, 3), np.uint8))
        old = FramePacket(1, 10, time.perf_counter(), frame.frame)
        self.app._pipeline.status.return_value = (2, 'online', 0)
        self.app._pipeline.latest_frame.return_value = frame
        self.app._pipeline.latest_result.return_value = RecognitionPacket(old, [], 5.0)
        self.app.apply_detections = Mock()
        self.app.draw_frame = Mock()
        self.app.update_frame()
        self.app.apply_detections.assert_not_called()
        self.app._pipeline.latest_result.return_value = RecognitionPacket(frame, [], 5.0)
        self.app.update_frame()
        self.app.apply_detections.assert_called_once_with([], 5.0)

    def test_real_ui_previews_while_background_inference_is_blocked(self):
        import threading
        from camera_pipeline import CameraPipeline
        from test_camera_pipeline import FakeCapture
        entered, release = threading.Event(), threading.Event()
        def recognize(frame):
            entered.set()
            release.wait(2)
            return []
        self.app._pipeline = CameraPipeline(recognize, lambda index: FakeCapture(index))
        self.app.open_camera()
        try:
            self.assertTrue(entered.wait(1))
            for _ in range(8):
                self.app.update_frame()
                time.sleep(0.01)
            self.assertIsNotNone(self.app.last_frame)
            self.assertGreater(self.app._last_frame_sequence, 1)
            self.assertEqual(self.app._last_result_sequence, -1)
            self.assertEqual(self.app._camera_state, 'online')
        finally:
            release.set()
            self.app._pipeline.stop()

    def test_import_button_uses_safe_import_and_refreshes_people(self):
        from unittest.mock import patch
        from test_database_import import make_database
        source = Path(self.temp.name) / 'incoming.db'
        make_database(source, name='Imported Test', employee='T01', reverse=True)
        self.app.open_people_manager()
        def widgets(parent):
            for child in parent.winfo_children():
                yield child
                yield from widgets(child)
        button = next(widget for widget in widgets(self.app.people_manager_window)
                      if isinstance(widget, tk.Button) and widget.cget('text') == '导入已有人员数据库')
        with patch('face_welcome_app.filedialog.askopenfilename', return_value=str(source)), \
                patch('face_welcome_app.messagebox.askyesno', return_value=True), \
                patch('face_welcome_app.messagebox.showinfo') as info, \
                patch('face_welcome_app.messagebox.showerror') as error:
            button.invoke()
            error.assert_not_called()
            info.assert_called_once()
        self.assertEqual(len(self.app.people), 1)
        self.assertEqual(len(self.app.sample_index), 1)
        self.assertEqual(len(list((Path(self.temp.name) / 'database-backups').glob('*.db'))), 1)
        self.assertEqual(self.app._pipeline._threads, [])

    def test_legacy_cascades_load_from_chinese_environment_path(self):
        for classifier in (self.app.face_detector, self.app.eye_detector, self.app.glasses_eye_detector):
            self.assertFalse(classifier.empty(), 'cascade must load even under a Chinese path')

    def test_unchanged_people_cache_does_not_reopen_camera(self):
        from unittest.mock import Mock
        pipeline = self.app._pipeline
        pipeline.invalidate = Mock()
        self.app.refresh_people()
        pipeline.invalidate.assert_not_called()


if __name__ == '__main__':
    unittest.main()

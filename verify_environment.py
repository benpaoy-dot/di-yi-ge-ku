"""Verify the actual app backend without opening a camera or writing personal data."""
from __future__ import annotations

import gc
import importlib.metadata
from contextlib import closing
import sqlite3
import sys
import tempfile
from pathlib import Path


def main() -> int:
    import cv2
    import numpy as np
    import onnxruntime as ort
    import tkinter as tk
    from face_welcome_app import FaceWelcomeApp

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    for package in ("insightface", "onnxruntime", "opencv-python", "numpy"):
        print(f"{package}: {importlib.metadata.version(package)}")
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")

    # Surface import failures that the production fallback intentionally catches.
    from insightface.app import FaceAnalysis  # noqa: F401

    with tempfile.TemporaryDirectory(prefix="face-welcome-check-") as directory:
        class CheckApp(FaceWelcomeApp):
            def _init_database(self) -> None:
                self.db_path = str(Path(directory) / "check.db")
                super()._init_database()

        root = tk.Tk()
        root.withdraw()
        app = None
        try:
            app = CheckApp(root)
            # Cancel camera acquisition and all recurring callbacks before any update.
            for callback in root.tk.call("after", "info"):
                root.after_cancel(callback)
            if not app.insightface_ready or app.active_feature_type != "insightface":
                raise RuntimeError("Primary InsightFace backend failed; fallback is NOT a pass")
            models = app.insightface_app.models
            for task in ("detection", "recognition"):
                if task not in models:
                    raise RuntimeError(f"Missing InsightFace {task} model")
                providers = models[task].session.get_providers()
                if providers != ["CPUExecutionProvider"]:
                    raise RuntimeError(f"Unexpected {task} execution providers: {providers}")
                print(f"PASS: InsightFace {task} model, CPU inference")
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            if app.insightface_app.get(blank):
                raise RuntimeError("InsightFace falsely detected a face in a blank frame")
            print("PASS: SCRFD inference on a blank frame")
            embedding = models["recognition"].get_feat(np.zeros((112, 112, 3), dtype=np.uint8))
            if embedding.shape != (1, 512) or not np.all(np.isfinite(embedding)):
                raise RuntimeError(f"Invalid ArcFace feature output: {embedding.shape}")
            print("PASS: ArcFace inference produces a finite 512-dimensional vector")
            if not app.sface_model_ready:
                raise RuntimeError("YuNet / SFace fallback models failed to load")
            app.face_detector_yunet.setInputSize((640, 480))
            _, faces = app.face_detector_yunet.detect(blank)
            if faces is not None and len(faces):
                raise RuntimeError("YuNet falsely detected a face in a blank frame")
            sface = app.face_recognizer.feature(np.zeros((112, 112, 3), dtype=np.uint8))
            if sface.shape != (1, 128) or not np.all(np.isfinite(sface)):
                raise RuntimeError(f"Invalid SFace feature output: {sface.shape}")
            print("PASS: YuNet / SFace fallback loads and runs")
            with closing(sqlite3.connect(app.db_path)) as connection:
                tables = {row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )}
                if not {"people", "face_samples"}.issubset(tables):
                    raise RuntimeError("SQLite schema initialization failed")
            root.update_idletasks()
            print("PASS: Tkinter UI construction and temporary SQLite schema")
            print("PASS: active_feature_type=insightface (not an OpenCV fallback)")
            print("Camera and audible speech were not tested; no personal database was created.")
        finally:
            if app is not None:
                app.close()
            else:
                root.destroy()
            # The app uses transaction contexts, which do not close SQLite handles.
            # Collect their unreachable cycles before Windows removes the temp DB.
            gc.collect()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)

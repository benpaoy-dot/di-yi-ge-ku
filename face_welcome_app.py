from __future__ import annotations

import base64
import os
import sqlite3
import subprocess
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np


BG = "#091116"
PANEL = "#101a21"
LINE = "#253640"
MUTED = "#81929b"
TEXT = "#edf6f5"
ACCENT = "#63e0ca"
WARN = "#ffb260"
DANGER = "#ff7078"
LANGUAGE_OPTIONS = {
    "中文 · zh-CN": "zh-CN",
    "English · en-US": "en-US",
    "中英双语 · zh-en": "zh-en",
}


@dataclass
class Person:
    id: int
    name: str
    employee_id: str
    language: str
    welcome_text: str
    sample_count: int = 0


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    confidence: float
    person: Person | None = None
    match_score: float = 0.0
    landmarks: np.ndarray | None = None
    backend: str = "legacy"
    embedding: np.ndarray | None = None


class FaceWelcomeApp:
    """Camera welcome console with a local people database.

    InsightFace/SCRFD detects faces and ArcFace produces identity embeddings
    when its model package is available. OpenCV YuNet + SFace remains as a
    fallback, while the old grayscale feature is kept only for compatibility.
    """

    insightface_match_threshold = 0.45
    sface_match_threshold = 0.363
    legacy_match_threshold = 0.76
    identity_confirmation_frames = 2
    yunet_score_threshold = 0.75
    yunet_nms_threshold = 0.30
    sample_min_detection_confidence = 0.82
    sample_min_face_size = 72
    sample_min_sharpness = 35.0
    sample_min_contrast = 18.0
    sample_brightness_range = (38.0, 218.0)
    duplicate_sample_threshold = 0.992
    detection_scale_factor = 1.12
    detection_min_neighbors = 5
    detection_min_size = (55, 55)
    face_aspect_min = 0.72
    face_aspect_max = 1.32

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Face Welcome AI")
        self.root.geometry("1600x900")
        self.root.minsize(1100, 700)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_welcome.db")
        self._init_database()
        self.people: dict[int, Person] = {}
        self.sample_index: list[tuple[Person, np.ndarray, str]] = []
        self.face_detector_yunet = None
        self.face_recognizer = None
        self.insightface_app = None
        self.insightface_ready = False
        self.sface_model_ready = False
        self.modern_model_ready = False
        self.active_feature_type = "legacy"
        self._load_modern_models()
        self.refresh_people()

        self.capture: cv2.VideoCapture | None = None
        self.camera_index = 0
        self.running = True
        self.frame_photo: tk.PhotoImage | None = None
        self.face_detector = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
        )
        self.eye_detector = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
        )
        self.glasses_eye_detector = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_eye_tree_eyeglasses.xml")
        )
        self.last_detections: list[Detection] = []
        self.current_person: Person | None = None
        self.last_spoken_person_id: int | None = None
        self.last_speech_at = 0.0
        self.welcome_flash_until = 0.0
        self.speech_cooldown = 18.0
        self.identity_history: deque[int | None] = deque(maxlen=3)
        self.last_frame_at = time.perf_counter()
        self.fps = 0.0
        self.activity = deque(maxlen=5)
        self.people_manager_window: tk.Toplevel | None = None

        self._build_ui()
        self.root.after(100, self.open_camera)
        self.root.after(40, self.update_frame)
        self.root.after(1000, self.update_clock)

    def _init_database(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    employee_id TEXT NOT NULL DEFAULT '',
                    language TEXT NOT NULL DEFAULT 'zh-CN',
                    welcome_text TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS face_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    feature BLOB NOT NULL,
                    feature_type TEXT NOT NULL DEFAULT 'legacy',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
                )
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(face_samples)").fetchall()}
            if "feature_type" not in columns:
                connection.execute("ALTER TABLE face_samples ADD COLUMN feature_type TEXT NOT NULL DEFAULT 'legacy'")

    def _load_modern_models(self) -> None:
        """Load InsightFace first, then keep YuNet/SFace as a fallback."""
        # Do not download a 120 MB model package from inside the GUI startup.
        # A local buffalo_sc/buffalo_s directory is used when it has already been cached;
        # otherwise the app starts immediately with the SFace fallback.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        insightface_dirs = [
            os.path.join(base_dir, "models", "buffalo_sc"),
            os.path.join(base_dir, "models", "buffalo_s"),
            os.path.join(base_dir, "models", "insightface", "buffalo_s"),
            os.path.join(base_dir, "models", "insightface", "buffalo_sc"),
            os.path.expanduser("~/.insightface/models/buffalo_s"),
            os.path.expanduser("~/.insightface/models/buffalo_sc"),
        ]
        insightface_dir = next(
            (path for path in insightface_dirs if os.path.isdir(path) and any(name.endswith(".onnx") for name in os.listdir(path))),
            None,
        )
        if insightface_dir:
            try:
                from insightface.app import FaceAnalysis

                self.insightface_app = FaceAnalysis(
                    name=insightface_dir,
                    providers=["CPUExecutionProvider"],
                )
                self.insightface_app.prepare(ctx_id=-1, det_size=(640, 640))
                self.insightface_ready = True
            except Exception:
                self.insightface_app = None
                self.insightface_ready = False

        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        detector_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
        recognizer_path = os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")
        if os.path.isfile(detector_path) and os.path.isfile(recognizer_path):
            try:
                # OpenCV's Windows file loader can reject ONNX paths containing
                # non-ASCII characters. Read the files with Python and use the
                # buffer overload so the app also works in Chinese directories.
                with open(detector_path, "rb") as detector_file:
                    detector_model = detector_file.read()
                with open(recognizer_path, "rb") as recognizer_file:
                    recognizer_model = recognizer_file.read()
                self.face_detector_yunet = cv2.FaceDetectorYN_create(
                    "onnx",
                    detector_model,
                    b"",
                    (320, 320),
                    self.yunet_score_threshold,
                    self.yunet_nms_threshold,
                    5000,
                )
                self.face_recognizer = cv2.FaceRecognizerSF_create("onnx", recognizer_model, b"")
                self.sface_model_ready = True
            except (cv2.error, OSError, ValueError):
                self.face_detector_yunet = None
                self.face_recognizer = None

        self.modern_model_ready = self.insightface_ready or self.sface_model_ready
        if self.insightface_ready:
            self.active_feature_type = "insightface"
        elif self.sface_model_ready:
            self.active_feature_type = "sface"
        else:
            self.active_feature_type = "legacy"

    def refresh_people(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            sample_count_sql = (
                f"COALESCE(SUM(CASE WHEN s.feature_type = '{self.active_feature_type}' THEN 1 ELSE 0 END), 0)"
                if self.active_feature_type != "legacy"
                else "COUNT(s.id)"
            )
            rows = connection.execute(
                f"""
                SELECT p.id, p.name, p.employee_id, p.language, p.welcome_text,
                       {sample_count_sql} AS sample_count
                FROM people p
                LEFT JOIN face_samples s ON s.person_id = p.id
                GROUP BY p.id
                ORDER BY p.id
                """
            ).fetchall()
            self.people = {
                int(row["id"]): Person(
                    id=int(row["id"]),
                    name=row["name"],
                    employee_id=row["employee_id"],
                    language=row["language"],
                    welcome_text=row["welcome_text"],
                    sample_count=int(row["sample_count"]),
                )
                for row in rows
            }
            samples = connection.execute("SELECT person_id, feature, feature_type FROM face_samples ORDER BY id").fetchall()
        self.sample_index = []
        for person_id, blob, feature_type in samples:
            person = self.people.get(int(person_id))
            if person is None:
                continue
            feature = np.frombuffer(blob, dtype=np.float32)
            if feature.size:
                self.sample_index.append((person, feature, feature_type or "legacy"))
        if hasattr(self, "db_status_label"):
            count = sum(person.sample_count for person in self.people.values())
            self.db_status_label.configure(text=f"LOCAL DB  ·  {len(self.people)} 人  /  {count} 份样本")

    def _font(self, size: int, weight: str = "normal") -> tuple[str, int, str]:
        return ("Microsoft YaHei UI", size, weight)

    def _label(self, parent, text, size=10, color=TEXT, weight="normal", **kwargs):
        return tk.Label(parent, text=text, fg=color, bg=kwargs.pop("bg", parent.cget("bg")), font=self._font(size, weight), **kwargs)

    def _build_ui(self) -> None:
        root = self.root
        top = tk.Frame(root, bg=BG, height=82)
        top.pack(fill="x", padx=30, pady=(20, 0))
        top.pack_propagate(False)
        tk.Frame(top, bg=LINE, height=1).pack(side="bottom", fill="x")

        brand = tk.Frame(top, bg=BG)
        brand.pack(side="left", fill="y")
        tk.Label(brand, text="FW", fg=ACCENT, bg=BG, font=self._font(13), width=3, height=1, relief="solid", bd=1).pack(side="left", padx=(0, 14), pady=12)
        title_box = tk.Frame(brand, bg=BG)
        title_box.pack(side="left", pady=10)
        self._label(title_box, "FACE WELCOME AI", 18, TEXT, "bold", anchor="w").pack(anchor="w")
        self._label(title_box, "SMART RECEPTION CONSOLE", 9, MUTED, anchor="w").pack(anchor="w", pady=(3, 0))

        status = tk.Frame(top, bg=BG)
        status.pack(side="right", fill="y")
        self.camera_status = self._label(status, "●  CAMERA OFFLINE", 10, DANGER, anchor="e")
        self.camera_status.pack(side="left", padx=(0, 24), pady=30)
        self._label(status, "LOCAL MATCH ENGINE", 10, MUTED, anchor="e").pack(side="left", padx=(0, 24), pady=30)
        self.clock_label = self._label(status, "--:--:--", 10, TEXT, anchor="e")
        self.clock_label.pack(side="left", pady=30)

        content = tk.Frame(root, bg=BG)
        content.pack(fill="both", expand=True, padx=30, pady=(22, 30))
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=0)
        content.grid_rowconfigure(0, weight=1)

        camera_shell = tk.Frame(content, bg=LINE, padx=1, pady=1)
        camera_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 22))
        camera_shell.grid_rowconfigure(0, weight=1)
        camera_shell.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(camera_shell, bg="#15252a", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self.draw_frame(self.last_frame))
        self.last_frame = None

        side = tk.Frame(content, bg=BG, width=360)
        side.grid(row=0, column=1, sticky="nsew")
        side.grid_propagate(False)
        side.grid_rowconfigure(1, weight=1)
        self._build_identity_panel(side)
        self._build_activity_panel(side)
        footer = tk.Frame(side, bg=BG)
        footer.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        self.db_status_label = self._label(footer, "LOCAL DB", 9, MUTED)
        self.db_status_label.pack(side="left")
        engine_labels = {
            "insightface": "INSIGHTFACE ENGINE",
            "sface": "SFACE ENGINE",
            "legacy": "LEGACY FALLBACK",
        }
        engine_label = engine_labels[self.active_feature_type]
        self._label(footer, engine_label, 9, ACCENT if self.modern_model_ready else WARN).pack(side="right")
        self.refresh_people()

    def _panel(self, parent, row: int, sticky="nsew") -> tk.Frame:
        frame = tk.Frame(parent, bg=PANEL, highlightbackground=LINE, highlightthickness=1, padx=18, pady=16)
        frame.grid(row=row, column=0, sticky=sticky, pady=(0, 14))
        return frame

    def _build_identity_panel(self, parent) -> None:
        panel = self._panel(parent, 0)
        header = tk.Frame(panel, bg=PANEL)
        header.pack(fill="x", pady=(0, 18))
        self._label(header, "CURRENT IDENTITY", 9, MUTED).pack(side="left")
        self.verify_label = self._label(header, "SCANNING", 9, MUTED)
        self.verify_label.pack(side="right")

        identity = tk.Frame(panel, bg=PANEL)
        identity.pack(fill="x")
        self.avatar = tk.Label(identity, text="--", fg=ACCENT, bg="#15272b", font=self._font(21), width=3, height=1, relief="solid", bd=1)
        self.avatar.pack(side="left", padx=(0, 14))
        info = tk.Frame(identity, bg=PANEL)
        info.pack(side="left", fill="x", expand=True)
        self.identity_name = self._label(info, "等待人脸", 24, TEXT, "bold", anchor="w")
        self.identity_name.pack(anchor="w")
        self.identity_detail = self._label(info, "摄像头启动后将显示识别状态", 10, MUTED, anchor="w")
        self.identity_detail.pack(anchor="w", pady=(5, 0))

        self._label(panel, "匹配相似度", 9, MUTED, anchor="w").pack(fill="x", pady=(22, 6))
        self.match_value = self._label(panel, "--", 9, TEXT, anchor="e")
        self.match_value.place(relx=1.0, y=102, anchor="ne")
        self.match_bar = tk.Canvas(panel, height=4, bg="#24333a", highlightthickness=0)
        self.match_bar.pack(fill="x")
        self._label(panel, "识别耗时", 9, MUTED, anchor="w").pack(fill="x", pady=(14, 0))
        self.latency_value = self._label(panel, "--", 9, TEXT, anchor="e")
        self.latency_value.place(relx=1.0, y=145, anchor="ne")

        voice = tk.Frame(panel, bg=PANEL)
        voice.pack(fill="x", pady=(18, 0), ipady=11)
        self.voice_icon = tk.Label(voice, text=") ))", fg=ACCENT, bg=PANEL, font=self._font(9), width=4)
        self.voice_icon.pack(side="left", padx=(0, 8))
        self.wave_canvas = tk.Canvas(voice, height=24, bg=PANEL, highlightthickness=0)
        self.wave_canvas.pack(side="left", fill="x", expand=True)
        self.voice_text = self._label(voice, "等待播报", 9, MUTED)
        self.voice_text.pack(side="right", padx=(8, 0))
        self.root.after(120, self.animate_wave)

        button_row = tk.Frame(panel, bg=PANEL)
        button_row.pack(fill="x", pady=(16, 0))
        self.connect_button = tk.Button(button_row, text="重新连接摄像头", command=self.open_camera, fg=BG, bg=ACCENT, activebackground="#a0f8e7", relief="flat", bd=0, padx=12, pady=7, font=self._font(9, "bold"), cursor="hand2")
        self.connect_button.pack(side="left")
        self.speak_button = tk.Button(button_row, text="播放欢迎语", command=self.speak_current_person, fg=TEXT, bg="#1c2c33", activebackground="#2b414a", relief="flat", bd=0, padx=12, pady=7, font=self._font(9), cursor="hand2")
        self.speak_button.pack(side="right")
        tk.Button(panel, text="人员管理  /  录入样本", command=self.open_people_manager, fg=ACCENT, bg=PANEL, activebackground="#1c2c33", activeforeground=TEXT, relief="flat", bd=0, padx=0, pady=5, font=self._font(9), cursor="hand2").pack(anchor="w", pady=(5, 0))

    def _build_activity_panel(self, parent) -> None:
        panel = self._panel(parent, 1)
        panel.grid_rowconfigure(1, weight=1)
        self._label(panel, "RECENT ACTIVITY", 9, MUTED).pack(side="top", anchor="w")
        self.activity_frame = tk.Frame(panel, bg=PANEL)
        self.activity_frame.pack(fill="both", expand=True, pady=(12, 0))
        self.refresh_activity()

    def refresh_activity(self) -> None:
        if not hasattr(self, "activity_frame"):
            return
        for child in self.activity_frame.winfo_children():
            child.destroy()
        if not self.activity:
            self._label(self.activity_frame, "暂无识别记录", 10, MUTED, anchor="w").pack(fill="x", pady=(10, 0))
            return
        for name, state, stamp, color in self.activity:
            row = tk.Frame(self.activity_frame, bg=PANEL)
            row.pack(fill="x", pady=(0, 11))
            tk.Frame(row, bg=LINE, height=1).pack(fill="x", pady=(0, 10))
            left = tk.Frame(row, bg=PANEL)
            left.pack(side="left")
            self._label(left, name, 10, TEXT, anchor="w").pack(side="left")
            self._label(left, state, 8, color, anchor="w").pack(side="left", padx=(8, 0))
            self._label(row, stamp, 9, MUTED, anchor="e").pack(side="right")

    def update_clock(self) -> None:
        if not self.running:
            return
        self.clock_label.configure(text=time.strftime("%H:%M:%S"))
        self.root.after(1000, self.update_clock)

    def open_camera(self) -> None:
        self.close_capture()
        for index in [self.camera_index, 1, 2]:
            capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                capture.set(cv2.CAP_PROP_FPS, 30)
                self.capture = capture
                self.camera_index = index
                self.camera_status.configure(text=f"●  CAMERA {index:02d} ONLINE", fg=ACCENT)
                self.identity_detail.configure(text="人员库已加载 · 等待人脸")
                return
            capture.release()
        self.camera_status.configure(text="●  CAMERA OFFLINE", fg=DANGER)
        self.identity_detail.configure(text="未找到摄像头，请连接后点击重新连接")

    def close_capture(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def update_frame(self) -> None:
        if not self.running:
            return
        if self.capture is not None:
            ok, frame = self.capture.read()
            if ok:
                self.last_frame = frame
                self.process_frame(frame)
                self.draw_frame(frame)
        self.root.after(40, self.update_frame)

    def extract_legacy_feature(self, frame, box: tuple[int, int, int, int]) -> np.ndarray | None:
        x, y, w, h = box
        height, width = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None
        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
        crop = cv2.equalizeHist(crop).astype(np.float32) / 255.0
        crop = (crop - float(crop.mean())) / max(float(crop.std()), 0.05)
        feature = crop.flatten()
        norm = float(np.linalg.norm(feature))
        return feature / norm if norm else None

    def read_image_file(self, path: str) -> np.ndarray | None:
        """Read an image safely even when its Windows path contains non-ASCII text."""
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
            if encoded.size == 0:
                return None
            return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        except (OSError, cv2.error):
            return None

    def extract_sface_feature(self, frame, landmarks: np.ndarray | None) -> np.ndarray | None:
        if not self.sface_model_ready or self.face_recognizer is None or landmarks is None:
            return None
        try:
            aligned = self.face_recognizer.alignCrop(frame, np.asarray(landmarks, dtype=np.float32))
            feature = np.asarray(self.face_recognizer.feature(aligned), dtype=np.float32).reshape(-1)
        except cv2.error:
            return None
        norm = float(np.linalg.norm(feature))
        return feature / norm if norm else None

    def extract_insightface_feature(self, detection: Detection) -> np.ndarray | None:
        if not self.insightface_ready or detection.embedding is None:
            return None
        feature = np.asarray(detection.embedding, dtype=np.float32).reshape(-1)
        if not feature.size or not np.all(np.isfinite(feature)):
            return None
        norm = float(np.linalg.norm(feature))
        return feature / norm if norm else None

    def extract_recognition_feature(
        self,
        frame,
        detection: Detection,
    ) -> tuple[np.ndarray | None, str]:
        if self.active_feature_type == "insightface":
            return self.extract_insightface_feature(detection), "insightface"
        if self.active_feature_type == "sface":
            return self.extract_sface_feature(frame, detection.landmarks), "sface"
        return self.extract_legacy_feature(frame, detection.box), "legacy"

    def evaluate_sample_quality(self, frame, detection: Detection) -> tuple[bool, str]:
        """Reject samples that are unlikely to produce a stable embedding."""
        x, y, w, h = detection.box
        if detection.backend in {"yunet", "insightface"} and detection.confidence < self.sample_min_detection_confidence:
            return False, f"检测置信度只有 {detection.confidence:.0%}"
        if min(w, h) < self.sample_min_face_size:
            return False, f"人脸太小（当前 {w}×{h}，请靠近摄像头）"
        if detection.landmarks is None and self.active_feature_type in {"insightface", "sface"}:
            return False, "没有得到完整的人脸关键点，请正对摄像头"

        height, width = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False, "人脸区域为空"
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        low_brightness, high_brightness = self.sample_brightness_range
        if brightness < low_brightness or brightness > high_brightness:
            return False, f"光线不合适（亮度 {brightness:.0f}，请避免过暗或过曝）"
        if contrast < self.sample_min_contrast:
            return False, "人脸对比度太低，请调整光线"
        if sharpness < self.sample_min_sharpness:
            return False, f"画面偏模糊（清晰度 {sharpness:.0f}，请保持不动）"
        return True, f"质量通过 · 清晰度 {sharpness:.0f} · 亮度 {brightness:.0f}"

    def is_duplicate_sample(self, person_id: int, feature: np.ndarray, feature_type: str) -> bool:
        """Avoid filling a person's gallery with identical adjacent frames."""
        for person, sample, sample_type in self.sample_index:
            if person.id != person_id or sample_type != feature_type or sample.size != feature.size:
                continue
            sample_norm = float(np.linalg.norm(sample))
            if sample_norm and float(np.dot(feature, sample) / sample_norm) >= self.duplicate_sample_threshold:
                return True
        return False

    def match_face(
        self,
        frame,
        box: tuple[int, int, int, int],
        landmarks: np.ndarray | None = None,
        embedding: np.ndarray | None = None,
    ) -> tuple[Person | None, float]:
        if not self.sample_index:
            return None, 0.0

        if self.active_feature_type == "insightface":
            features = [("insightface", embedding)]
        elif self.active_feature_type == "sface":
            features = [("sface", self.extract_sface_feature(frame, landmarks))]
        else:
            features = [("legacy", self.extract_legacy_feature(frame, box))]

        best_person = None
        best_score = -1.0
        best_kind = "legacy"
        for feature_type, feature in features:
            if feature is None:
                continue
            for person, sample, sample_type in self.sample_index:
                if sample_type != feature_type or sample.size != feature.size:
                    continue
                sample_norm = float(np.linalg.norm(sample))
                if sample_norm == 0:
                    continue
                score = float(np.dot(feature, sample) / sample_norm)
                if score > best_score:
                    best_person, best_score, best_kind = person, score, feature_type

        thresholds = {
            "insightface": self.insightface_match_threshold,
            "sface": self.sface_match_threshold,
            "legacy": self.legacy_match_threshold,
        }
        threshold = thresholds[best_kind]
        if best_score < threshold:
            return None, max(0.0, best_score)
        return best_person, best_score

    def is_face_candidate(self, gray: np.ndarray, box: tuple[int, int, int, int]) -> bool:
        """Reject face-shaped background objects before identity matching."""
        x, y, w, h = box
        if h <= 0 or w <= 0:
            return False
        aspect = w / h
        if not self.face_aspect_min <= aspect <= self.face_aspect_max:
            return False

        height, width = gray.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        # Use the eye cascades only as a soft signal. A hard eye requirement
        # rejects people wearing glasses, sunglasses, or with eye glare.
        upper = roi[: max(1, int(roi.shape[0] * 0.68)), :]
        eyes = self.eye_detector.detectMultiScale(
            upper,
            scaleFactor=1.10,
            minNeighbors=4,
            minSize=(10, 10),
        )
        glasses_eyes = self.glasses_eye_detector.detectMultiScale(
            upper,
            scaleFactor=1.10,
            minNeighbors=3,
            minSize=(10, 10),
        )
        if len(eyes) or len(glasses_eyes):
            return True

        # The frontal-face cascade is already the primary classifier. Accept
        # its candidate when eye detection is inconclusive, especially for
        # glasses, glare, masks, or mild head turns.
        return True

    def detect_faces(self, frame) -> list[Detection]:
        if self.insightface_ready and self.insightface_app is not None:
            try:
                faces = self.insightface_app.get(frame)
            except Exception:
                faces = None
            if faces is not None:
                height, width = frame.shape[:2]
                detections: list[Detection] = []
                for face in faces:
                    bbox = np.asarray(face.bbox, dtype=np.float32).reshape(-1)
                    if bbox.size < 4:
                        continue
                    x, y, x2, y2 = [int(round(value)) for value in bbox[:4]]
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    x2 = max(x + 1, min(x2, width))
                    y2 = max(y + 1, min(y2, height))
                    w, h = x2 - x, y2 - y
                    if w < 45 or h < 45:
                        continue
                    landmarks = None
                    if face.kps is not None:
                        landmarks = np.asarray(face.kps, dtype=np.float32)
                    embedding = None
                    if face.normed_embedding is not None:
                        embedding = np.asarray(face.normed_embedding, dtype=np.float32).reshape(-1)
                    elif face.embedding is not None:
                        raw_embedding = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
                        norm = float(np.linalg.norm(raw_embedding))
                        embedding = raw_embedding / norm if norm else None
                    person, score = self.match_face(frame, (x, y, w, h), landmarks, embedding)
                    detections.append(
                        Detection(
                            (x, y, w, h),
                            float(face.det_score or 0.0),
                            person,
                            score,
                            landmarks,
                            "insightface",
                            embedding,
                        )
                    )
                return detections

        if self.sface_model_ready and self.face_detector_yunet is not None:
            height, width = frame.shape[:2]
            self.face_detector_yunet.setInputSize((width, height))
            _, faces = self.face_detector_yunet.detect(frame)
            detections: list[Detection] = []
            if faces is None:
                return detections
            for face in faces:
                x, y, w, h = [int(round(value)) for value in face[:4]]
                if w < 45 or h < 45:
                    continue
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = min(w, width - x)
                h = min(h, height - y)
                landmarks = np.asarray(face, dtype=np.float32)
                person, score = self.match_face(frame, (x, y, w, h), landmarks)
                detections.append(
                    Detection(
                        (x, y, w, h),
                        float(face[14]),
                        person,
                        score,
                        landmarks,
                        "yunet",
                        None,
                    )
                )
            return detections

        small = cv2.resize(frame, None, fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=self.detection_scale_factor,
            minNeighbors=self.detection_min_neighbors,
            minSize=self.detection_min_size,
        )
        detections = []
        for x, y, w, h in faces:
            if not self.is_face_candidate(gray, (x, y, w, h)):
                continue
            box = (x * 2, y * 2, w * 2, h * 2)
            person, score = self.match_face(frame, box)
            detections.append(Detection(box, 0.964, person, score, None, "legacy"))
        return detections

    def process_frame(self, frame) -> None:
        started = time.perf_counter()
        detections = self.detect_faces(frame)
        self.last_detections = detections
        self.latency_value.configure(text=f"{int((time.perf_counter() - started) * 1000)} ms")
        now = time.perf_counter()
        delta = now - self.last_frame_at
        self.fps = 1 / delta if delta > 0 else 0
        self.last_frame_at = now

        recognized = [d for d in detections if d.person is not None]
        best = max(recognized, key=lambda item: item.match_score, default=None)
        candidate_id = best.person.id if best and best.person else None
        self.identity_history.append(candidate_id)
        confirmed = None
        if candidate_id is not None:
            recent = list(self.identity_history)[-self.identity_confirmation_frames :]
            if len(recent) == self.identity_confirmation_frames and all(item == candidate_id for item in recent):
                confirmed = best
        self.current_person = confirmed.person if confirmed else None
        self.update_identity_panel(confirmed, bool(detections))
        if confirmed is not None and confirmed.person is not None:
            person = confirmed.person
            if person.id != self.last_spoken_person_id:
                self.last_spoken_person_id = person.id
                self.last_speech_at = now
                self.welcome_flash_until = now + 3.5
                self.activity.appendleft((person.name, "VERIFIED", time.strftime("%H:%M:%S"), ACCENT))
                self.refresh_activity()
                self.voice_text.configure(text=self.welcome_text(person), fg=ACCENT)
                self.speak_person(person)
        elif not detections:
            self.last_spoken_person_id = None
            self.welcome_flash_until = 0.0
            self.voice_text.configure(text="等待播报", fg=MUTED)

    def update_identity_panel(self, best: Detection | None, has_faces: bool) -> None:
        if best and best.person:
            person = best.person
            self.verify_label.configure(text="VERIFIED", fg=ACCENT)
            self.identity_name.configure(text=person.name)
            self.identity_detail.configure(text=f"员工编号 {person.employee_id or '--'} · {self.language_label(person.language)}")
            self.avatar.configure(text=person.name[:1])
            self.match_value.configure(text=f"{best.match_score:.1%}")
            self.paint_match_bar(best.match_score, ACCENT)
        elif has_faces:
            self.verify_label.configure(text="UNKNOWN", fg=WARN)
            self.identity_name.configure(text="未登记人员")
            self.identity_detail.configure(text="检测到人脸 · 请在人员管理中录入样本")
            self.avatar.configure(text="?")
            score = max((d.match_score for d in self.last_detections), default=0.0)
            self.match_value.configure(text=f"{score:.1%}" if score else "--")
            self.paint_match_bar(score, WARN)
            self.voice_text.configure(text="未登记 · 不播报", fg=WARN)
        else:
            self.verify_label.configure(text="SCANNING", fg=MUTED)
            self.identity_name.configure(text="等待人脸")
            self.identity_detail.configure(text="人员库已加载 · 请面向摄像头")
            self.avatar.configure(text="--")
            self.match_value.configure(text="--")
            self.paint_match_bar(0, ACCENT)

    def paint_match_bar(self, score: float, color: str) -> None:
        self.match_bar.delete("all")
        self.match_bar.update_idletasks()
        self.match_bar.create_rectangle(0, 0, max(1, self.match_bar.winfo_width() * min(score, 1)), 4, fill=color, outline="")

    def draw_frame(self, frame) -> None:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        if width < 20 or height < 20:
            return
        if frame is None:
            self.draw_placeholder(width, height)
            return
        self.canvas.delete("all")
        source_h, source_w = frame.shape[:2]
        scale = min(width / source_w, height / source_h)
        target_w, target_h = int(source_w * scale), int(source_h * scale)
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".png", resized)
        if ok:
            self.frame_photo = tk.PhotoImage(data=base64.b64encode(encoded.tobytes()).decode("ascii"))
            offset_x, offset_y = (width - target_w) // 2, (height - target_h) // 2
            self.canvas.create_image(offset_x, offset_y, image=self.frame_photo, anchor="nw")
            self.draw_grid(width, height)
            for detection in self.last_detections:
                x, y, w, h = detection.box
                x1, y1 = offset_x + int(x * scale), offset_y + int(y * scale)
                x2, y2 = offset_x + int((x + w) * scale), offset_y + int((y + h) * scale)
                self.draw_face_overlay(x1, y1, x2, y2, detection)
            self.draw_camera_chrome(width, height)

    def draw_placeholder(self, width: int, height: int) -> None:
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, width, height, fill="#15252a", outline="")
        self.draw_grid(width, height)
        self.draw_camera_chrome(width, height)
        self.canvas.create_text(width / 2, height / 2 - 12, text="CAMERA FEED STANDBY", fill=ACCENT, font=self._font(17, "bold"))
        self.canvas.create_text(width / 2, height / 2 + 20, text="连接摄像头后点击右侧“重新连接摄像头”", fill=MUTED, font=self._font(11))

    def draw_grid(self, width: int, height: int) -> None:
        for x in range(0, width, 80):
            self.canvas.create_line(x, 0, x, height, fill="#77ded1", stipple="gray50", width=1)
        for y in range(0, height, 80):
            self.canvas.create_line(0, y, width, y, fill="#77ded1", stipple="gray50", width=1)

    def draw_camera_chrome(self, width: int, height: int) -> None:
        size, gap = 36, 20
        for points in [
            (gap, gap, gap + size, gap, gap, gap + size),
            (width - gap, gap, width - gap - size, gap, width - gap, gap + size),
            (gap, height - gap, gap + size, height - gap, gap, height - gap - size),
            (width - gap, height - gap, width - gap - size, height - gap, width - gap, height - gap - size),
        ]:
            self.canvas.create_line(*points, fill=ACCENT, width=1)
        self.canvas.create_text(22, 22, text="CAMERA 01  /  LIVE FEED", anchor="nw", fill="#c4e7e2", font=self._font(9))
        self.canvas.create_line(0, int(height * 0.2), width, int(height * 0.2), fill=ACCENT, stipple="gray50")
        if self.current_person and time.perf_counter() < self.welcome_flash_until:
            welcome, subtitle = self.welcome_text(self.current_person), "身份已确认 · 欢迎语已触发"
        elif self.current_person:
            welcome, subtitle = f"已识别 · {self.current_person.name}", "身份已确认 · 语音冷却中"
        elif self.last_detections:
            welcome, subtitle = "检测到未登记人员", "请先录入人员样本 · 本地处理"
        else:
            welcome, subtitle = "欢迎，等待识别", "人员库匹配待命 · 本地处理"
        self.canvas.create_text(22, height - 82, text="WELCOME MESSAGE", anchor="sw", fill=ACCENT, font=self._font(9, "bold"))
        self.canvas.create_text(22, height - 48, text=welcome, anchor="sw", fill=TEXT, font=self._font(25, "bold"))
        self.canvas.create_text(22, height - 22, text=subtitle, anchor="sw", fill="#b7c9c9", font=self._font(10))

    def draw_face_overlay(self, x1: int, y1: int, x2: int, y2: int, detection: Detection) -> None:
        person = detection.person
        color = ACCENT if person else WARN
        label_name = person.name if person else "未登记人员"
        label = f"{label_name}  /  {detection.match_score:.1%}" if person else f"{label_name}  /  UNKNOWN"
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=1)
        corner = 18
        for points in [
            (x1, y1 + corner, x1, y1, x1 + corner, y1),
            (x2 - corner, y1, x2, y1, x2, y1 + corner),
            (x1, y2 - corner, x1, y2, x1 + corner, y2),
            (x2 - corner, y2, x2, y2, x2, y2 - corner),
        ]:
            self.canvas.create_line(*points, fill="#e4fffa", width=2)
        label_top = max(4, y1 - 32)
        label_width = max(190, 12 * len(label) + 18)
        self.canvas.create_rectangle(x1, label_top, x1 + label_width, label_top + 28, fill=color, outline="")
        self.canvas.create_text(x1 + 10, label_top + 14, text=label, anchor="w", fill=BG, font=self._font(9, "bold"))
        side_text = f"FACE\nMATCH\n{detection.match_score:.1%}" if person else "FACE\nUNKNOWN\n--"
        self.canvas.create_text(x2 + 14, y1 + 44, text=side_text, anchor="nw", fill=color, font=self._font(9))

    def animate_wave(self) -> None:
        if not self.running:
            return
        self.wave_canvas.delete("all")
        active = bool(self.current_person and time.perf_counter() - self.last_speech_at < 3)
        color = ACCENT if active else "#38545b"
        widths = [7, 15, 23, 12, 20, 10, 17, 7, 22, 13, 8, 17]
        x = 2
        for bar_height in widths:
            self.wave_canvas.create_rectangle(x, (24 - bar_height) / 2, x + 3, (24 + bar_height) / 2, fill=color, outline="")
            x += 7
        self.root.after(180, self.animate_wave)

    def language_label(self, language: str) -> str:
        for label, value in LANGUAGE_OPTIONS.items():
            if value == language:
                return label.split(" · ")[0]
        return language

    def welcome_text(self, person: Person) -> str:
        if person.welcome_text.strip():
            return person.welcome_text.replace("{name}", person.name)
        if person.language == "en-US":
            return f"Welcome, {person.name}"
        if person.language == "zh-en":
            return f"欢迎{person.name}，Welcome, {person.name}"
        return f"欢迎{person.name}"

    def speak_current_person(self) -> None:
        if self.current_person:
            self.speak_person(self.current_person)
        else:
            messagebox.showinfo("暂无身份", "请先识别到已录入人员，或在人员管理中采集人脸样本。")

    def speak_person(self, person: Person) -> None:
        text = self.welcome_text(person)

        def worker() -> None:
            def encoded(value: str) -> str:
                return base64.b64encode(value.encode("utf-16le")).decode("ascii")

            if person.language == "zh-en":
                zh_text = f"欢迎{person.name}"
                en_text = f"Welcome, {person.name}"
                script = (
                    f"$zh=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('{encoded(zh_text)}')); "
                    f"$en=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('{encoded(en_text)}')); "
                    "$zhVoice=$s.GetInstalledVoices() | Where-Object {$_.VoiceInfo.Culture.Name -eq 'zh-CN'} | Select-Object -First 1; "
                    "$enVoice=$s.GetInstalledVoices() | Where-Object {$_.VoiceInfo.Culture.Name -eq 'en-US'} | Select-Object -First 1; "
                    "if ($zhVoice) {$s.SelectVoice($zhVoice.VoiceInfo.Name)}; $s.Speak($zh); "
                    "if ($enVoice) {$s.SelectVoice($enVoice.VoiceInfo.Name)}; $s.Speak($en); $s.Dispose()"
                )
            else:
                culture = "en-US" if person.language == "en-US" else "zh-CN"
                script = (
                    f"$t=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('{encoded(text)}')); "
                    f"$v=$s.GetInstalledVoices() | Where-Object {{$_.VoiceInfo.Culture.Name -eq '{culture}'}} | Select-Object -First 1; "
                    "if ($v) {$s.SelectVoice($v.VoiceInfo.Name)}; $s.Speak($t); $s.Dispose()"
                )
            command = (
                "$ErrorActionPreference='Stop'; "
                "Add-Type -AssemblyName System.Speech; "
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Volume=100; $s.Rate=0; $s.SetOutputToDefaultAudioDevice(); "
                + script
            )
            try:
                result = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-WindowStyle", "Hidden", "-Command", command],
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    self.root.after(0, lambda: self.voice_text.configure(text="播报完成", fg=ACCENT))
                else:
                    self.root.after(0, lambda: self.voice_text.configure(text="播报失败 · 检查系统语音", fg=DANGER))
            except (OSError, subprocess.SubprocessError):
                self.root.after(0, lambda: self.voice_text.configure(text="播报失败 · 检查系统语音", fg=DANGER))

        threading.Thread(target=worker, daemon=True).start()

    def open_people_manager(self) -> None:
        if self.people_manager_window and self.people_manager_window.winfo_exists():
            self.people_manager_window.deiconify()
            self.people_manager_window.lift()
            return
        window = tk.Toplevel(self.root)
        self.people_manager_window = window
        window.title("人员管理 · Face Welcome AI")
        window.geometry("900x560")
        window.configure(bg=BG)
        window.protocol("WM_DELETE_WINDOW", window.destroy)

        left = tk.Frame(window, bg=PANEL, padx=14, pady=14)
        left.pack(side="left", fill="both", expand=True, padx=(20, 10), pady=20)
        self._label(left, "PEOPLE DATABASE", 10, MUTED, bg=PANEL).pack(anchor="w", pady=(0, 10))
        columns = ("name", "employee", "language", "samples")
        tree = ttk.Treeview(left, columns=columns, show="headings", height=16)
        samples_title = "可用样本" if self.active_feature_type != "legacy" else "样本数"
        for column, title in [("name", "姓名"), ("employee", "员工编号"), ("language", "播报语言"), ("samples", samples_title)]:
            tree.heading(column, text=title)
        tree.column("name", width=120)
        tree.column("employee", width=110)
        tree.column("language", width=140)
        tree.column("samples", width=70, anchor="center")
        tree.pack(fill="both", expand=True)

        right = tk.Frame(window, bg=BG, width=300)
        right.pack(side="right", fill="y", padx=(10, 20), pady=20)
        right.pack_propagate(False)
        self._label(right, "PERSON PROFILE", 10, MUTED, bg=BG).pack(anchor="w", pady=(0, 18))
        form = tk.Frame(right, bg=BG)
        form.pack(fill="x")
        name_var, employee_var = tk.StringVar(), tk.StringVar()
        language_var = tk.StringVar(value="中文 · zh-CN")
        welcome_var = tk.StringVar()
        status_var = tk.StringVar(value="新建人员")
        editing_id: dict[str, int | None] = {"value": None}

        def field(label: str, variable: tk.StringVar) -> None:
            self._label(form, label, 9, MUTED, bg=BG).pack(anchor="w", pady=(0, 4))
            tk.Entry(form, textvariable=variable, bg="#15252a", fg=TEXT, insertbackground=ACCENT, relief="flat", font=self._font(10)).pack(fill="x", ipady=6, pady=(0, 13))

        field("姓名", name_var)
        field("员工编号", employee_var)
        self._label(form, "播报语言", 9, MUTED, bg=BG).pack(anchor="w", pady=(0, 4))
        ttk.Combobox(form, textvariable=language_var, values=list(LANGUAGE_OPTIONS), state="readonly").pack(fill="x", ipady=4, pady=(0, 13))
        self._label(form, "自定义欢迎语（可选，使用 {name}）", 9, MUTED, bg=BG).pack(anchor="w", pady=(0, 4))
        tk.Entry(form, textvariable=welcome_var, bg="#15252a", fg=TEXT, insertbackground=ACCENT, relief="flat", font=self._font(10)).pack(fill="x", ipady=6, pady=(0, 14))
        model_names = {
            "insightface": "InsightFace / ArcFace",
            "sface": "SFace",
            "legacy": "兼容模式",
        }
        active_model_name = model_names[self.active_feature_type]
        sample_hint = (
            f"当前计数只包含 {active_model_name} 可用样本；旧版灰度样本需要重新采集。建议录入 3-5 份不同角度样本。"
            if self.active_feature_type != "legacy"
            else "建议录入 3-5 份不同角度样本。"
        )
        self._label(form, f"支持中文、English、中英双语；{sample_hint}", 9, MUTED, bg=BG, wraplength=280, justify="left").pack(anchor="w", pady=(0, 16))

        def reload_tree(select_id: int | None = None) -> None:
            self.refresh_people()
            for item in tree.get_children():
                tree.delete(item)
            for person in self.people.values():
                tree.insert("", "end", iid=str(person.id), values=(person.name, person.employee_id or "--", self.language_label(person.language), person.sample_count))
            if select_id is not None and str(select_id) in tree.get_children():
                tree.selection_set(str(select_id))
                tree.focus(str(select_id))

        def clear_form() -> None:
            editing_id["value"] = None
            status_var.set("新建人员")
            name_var.set("")
            employee_var.set("")
            language_var.set("中文 · zh-CN")
            welcome_var.set("")

        def load_selected(_event=None) -> None:
            selection = tree.selection()
            if not selection:
                return
            person = self.people.get(int(selection[0]))
            if not person:
                return
            editing_id["value"] = person.id
            status_var.set(f"编辑人员  /  {person.name}")
            name_var.set(person.name)
            employee_var.set(person.employee_id)
            language_var.set(next((label for label, value in LANGUAGE_OPTIONS.items() if value == person.language), list(LANGUAGE_OPTIONS)[0]))
            welcome_var.set(person.welcome_text)

        def save_person() -> None:
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning("缺少姓名", "请填写人员姓名。", parent=window)
                return
            employee_id = employee_var.get().strip()
            language = LANGUAGE_OPTIONS.get(language_var.get(), "zh-CN")
            try:
                with sqlite3.connect(self.db_path) as connection:
                    if editing_id["value"] is None:
                        cursor = connection.execute("INSERT INTO people(name, employee_id, language, welcome_text) VALUES (?, ?, ?, ?)", (name, employee_id, language, welcome_var.get().strip()))
                        person_id = int(cursor.lastrowid)
                    else:
                        person_id = int(editing_id["value"])
                        connection.execute("UPDATE people SET name=?, employee_id=?, language=?, welcome_text=? WHERE id=?", (name, employee_id, language, welcome_var.get().strip(), person_id))
            except sqlite3.IntegrityError:
                messagebox.showwarning("姓名重复", "人员库中已经存在同名人员，请换一个姓名。", parent=window)
                return
            reload_tree(person_id)
            load_selected()

        def delete_person() -> None:
            selection = tree.selection()
            if not selection:
                return
            person = self.people.get(int(selection[0]))
            if not person or not messagebox.askyesno("删除人员", f"确定删除“{person.name}”及其人脸样本吗？", parent=window):
                return
            with sqlite3.connect(self.db_path) as connection:
                connection.execute("DELETE FROM face_samples WHERE person_id=?", (person.id,))
                connection.execute("DELETE FROM people WHERE id=?", (person.id,))
            clear_form()
            reload_tree()

        def capture_sample() -> None:
            person_id = editing_id["value"]
            if person_id is None:
                messagebox.showinfo("先保存人员", "先填写姓名并点击“保存人员”，再采集人脸样本。", parent=window)
                return
            if self.last_frame is None or not self.last_detections:
                messagebox.showinfo("没有检测到人脸", "请回到摄像头前，确保画面中出现清晰人脸后再采集。", parent=window)
                return
            detection = max(self.last_detections, key=lambda item: item.box[2] * item.box[3])
            quality_ok, quality_message = self.evaluate_sample_quality(self.last_frame, detection)
            if not quality_ok:
                messagebox.showwarning("样本质量不合格", f"{quality_message}。\n请调整姿势、距离或光线后再采集。", parent=window)
                return
            feature, feature_type = self.extract_recognition_feature(self.last_frame, detection)
            if feature is None:
                messagebox.showwarning("样本无效", "当前人脸无法完成对齐，请正对摄像头后重试。", parent=window)
                return
            if self.is_duplicate_sample(person_id, feature, feature_type):
                messagebox.showinfo("样本重复", "这张样本与该人员已有样本过于相似。请轻微转头或改变距离后再采集。", parent=window)
                return
            with sqlite3.connect(self.db_path) as connection:
                connection.execute(
                    "INSERT INTO face_samples(person_id, feature, feature_type) VALUES (?, ?, ?)",
                    (person_id, feature.astype(np.float32).tobytes(), feature_type),
                )
            self.refresh_people()
            reload_tree(person_id)
            load_selected()
            model_label = {
                "insightface": "InsightFace / ArcFace",
                "sface": "SFace",
                "legacy": "兼容模式",
            }.get(feature_type, feature_type)
            messagebox.showinfo("采集完成", f"已保存 1 份人脸样本（{model_label}）。\n{quality_message}\n建议从正面、轻微转头、不同距离各采集几次。", parent=window)

        def upload_samples() -> None:
            person_id = editing_id["value"]
            if person_id is None:
                messagebox.showinfo("先保存人员", "先填写姓名并点击“保存人员”，再上传人脸图片。", parent=window)
                return
            paths = filedialog.askopenfilenames(
                parent=window,
                title="选择人脸样本图片",
                filetypes=[
                    ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                    ("所有文件", "*.*"),
                ],
            )
            if not paths:
                return

            saved = 0
            rejected: list[str] = []
            for path in paths:
                filename = os.path.basename(path)
                frame = self.read_image_file(path)
                if frame is None:
                    rejected.append(f"{filename}：无法读取图片")
                    continue
                detections = self.detect_faces(frame)
                if not detections:
                    rejected.append(f"{filename}：没有检测到合格人脸")
                    continue
                if len(detections) > 1:
                    rejected.append(f"{filename}：检测到多张人脸，请使用单人照片")
                    continue
                detection = detections[0]
                quality_ok, quality_message = self.evaluate_sample_quality(frame, detection)
                if not quality_ok:
                    rejected.append(f"{filename}：{quality_message}")
                    continue
                feature, feature_type = self.extract_recognition_feature(frame, detection)
                if feature is None:
                    rejected.append(f"{filename}：无法生成特征，请换一张正面清晰照片")
                    continue
                if self.is_duplicate_sample(person_id, feature, feature_type):
                    rejected.append(f"{filename}：与已有样本过于相似")
                    continue
                with sqlite3.connect(self.db_path) as connection:
                    connection.execute(
                        "INSERT INTO face_samples(person_id, feature, feature_type) VALUES (?, ?, ?)",
                        (person_id, feature.astype(np.float32).tobytes(), feature_type),
                    )
                self.refresh_people()
                saved += 1

            reload_tree(person_id)
            load_selected()
            result_lines = [f"成功保存 {saved} 份 {self.active_feature_type} 样本。"]
            if rejected:
                result_lines.append(f"有 {len(rejected)} 张图片未保存：")
                result_lines.extend(rejected[:8])
                if len(rejected) > 8:
                    result_lines.append(f"……另外 {len(rejected) - 8} 张未显示")
            result_lines.append("原始图片不会复制到数据库，只保存人脸特征。")
            if saved:
                result_lines.append("建议每个人准备 3-5 张不同角度、不同距离的单人照片。")
            messagebox.showinfo("图片样本处理完成", "\n".join(result_lines), parent=window)

        tree.bind("<<TreeviewSelect>>", load_selected)
        actions = tk.Frame(right, bg=BG)
        actions.pack(side="bottom", fill="x")
        tk.Button(actions, text="新建", command=clear_form, fg=TEXT, bg="#1c2c33", relief="flat", bd=0, padx=10, pady=7, font=self._font(9), cursor="hand2").pack(side="left")
        tk.Button(actions, text="保存人员", command=save_person, fg=BG, bg=ACCENT, relief="flat", bd=0, padx=10, pady=7, font=self._font(9, "bold"), cursor="hand2").pack(side="left", padx=6)
        tk.Button(actions, text="采集当前人脸", command=capture_sample, fg=BG, bg=WARN, relief="flat", bd=0, padx=10, pady=7, font=self._font(9, "bold"), cursor="hand2").pack(fill="x", pady=(8, 0))
        tk.Button(actions, text="从图片上传样本", command=upload_samples, fg=TEXT, bg="#1c2c33", activebackground="#2b414a", relief="flat", bd=0, padx=10, pady=7, font=self._font(9, "bold"), cursor="hand2").pack(fill="x", pady=(6, 0))
        tk.Button(actions, text="删除人员", command=delete_person, fg=DANGER, bg=BG, relief="flat", bd=0, padx=0, pady=7, font=self._font(9), cursor="hand2").pack(anchor="w")
        self._label(right, "", 9, MUTED, bg=BG, textvariable=status_var).pack(side="bottom", anchor="w", pady=(0, 12))
        reload_tree()

    def close(self) -> None:
        self.running = False
        self.close_capture()
        if self.people_manager_window and self.people_manager_window.winfo_exists():
            self.people_manager_window.destroy()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    FaceWelcomeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

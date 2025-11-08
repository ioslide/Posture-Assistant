
import sys
import configparser
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QSpinBox, QListWidget,QTextEdit,
                             QDialog, QDialogButtonBox, QGroupBox, QCheckBox,QFileDialog,
                             QSystemTrayIcon, QMenu, QMessageBox, QApplication, QStyle)
import os
from PyQt6.QtGui import QTextCursor, QAction

from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QIcon, QFont
import cv2
import numpy as np

import time
import threading
from datetime import datetime
from pynput import mouse, keyboard
from collections import deque

if sys.platform == 'win32':
    import win32gui
    import win32process
    import psutil
    import winreg

import mediapipe as mp
from PyQt6.QtCore import QObject, pyqtSignal

import face_recognition
x
APP_NAME = "SmartHealthAssistant"

class ForcedBreakDialog(QDialog):
    def __init__(self, duration=15, title="🚨", message="请休息一下！", parent=None):
        super().__init__(parent)
        self.duration = duration
        self.countdown = duration
        self.title_text = title
        self.message_text = message
        self.init_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_countdown)
        self.timer.start(1000)

    def init_ui(self):
        self.setWindowTitle("强制休息提醒")
        
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title_label = QLabel(self.title_text)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.message_label = QLabel(self.message_text)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setWordWrap(True)

        self.countdown_label = QLabel(f"强制休息倒计时: {self.countdown} 秒")
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.setWordWrap(True)

        layout.addWidget(self.title_label)
        layout.addSpacing(30)
        layout.addWidget(self.message_label)
        layout.addSpacing(50)
        layout.addWidget(self.countdown_label)
        self.setLayout(layout)
        
        # --- 样式表 ---
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(0, 0, 0, 0.85);
            }
            QLabel {
                color: white;
                background-color: transparent;
            }
        """)
        
        # 最后再显示窗口，这样能确保所有控件都已创建
        self.showFullScreen()

    def _update_font_sizes(self):
        """
        核心函数：根据窗口高度动态调整字体大小。
        """
        # --- 守护条件 (关键修复) ---
        # 检查 self.title_label 是否已存在。如果不存在，说明UI未完全初始化，
        # 直接返回以避免 AttributeError。
        if not hasattr(self, 'title_label'):
            return

        h = self.height()
        if h == 0:
            return

        # --- 标题字体 ---
        title_font = self.title_label.font()
        title_font.setPointSize(int(h / 10))
        title_font.setBold(True)
        self.title_label.setFont(title_font)

        # --- 消息字体 ---
        message_font = self.message_label.font()
        message_font.setPointSize(int(h / 8))
        self.message_label.setFont(message_font)

        # --- 倒计时字体 ---
        countdown_font = self.countdown_label.font()
        countdown_font.setPointSize(int(h / 12))
        self.countdown_label.setFont(countdown_font)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_font_sizes()

    def update_countdown(self):
        self.countdown -= 1
        self.countdown_label.setText(f"强制休息倒计时: {self.countdown} 秒")
        if self.countdown <= 0:
            self.timer.stop()
            self.accept()
        else:
            self.raise_()
            self.activateWindow()

    def keyPressEvent(self, event):x
        pass

    def mousePressEvent(self, event):
        pass


class AlertDialog(QDialog):
    def __init__(self, alert_type="sedentary", auto_close_duration=10, parent=None):
        super().__init__(parent)

        self.alert_type = alert_type
        self.init_ui()

        self.auto_close_timer = QTimer()
        self.auto_close_timer.timeout.connect(self.accept)
        self.auto_close_timer.start(auto_close_duration * 1000)

    def init_ui(self):
        self.setWindowTitle("健康提醒")
        self.setFixedSize(400, 200)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Dialog)
        layout = QVBoxLayout()

        title_label = QLabel()
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        message_label = QLabel()
        message_label.setWordWrap(True)
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if self.alert_type == "sedentary":
            title_label.setText("⏰ 久坐提醒")
            message_label.setText("您已经坐了很长时间了！\n建议站起来活动一下，休息一会儿。")
        else:
            title_label.setText("🔔 姿势提醒")
            message_label.setText("检测到您的坐姿不正确！\n请调整您的坐姿")

        ok_button = QPushButton("知道了")
        ok_button.clicked.connect(self.accept)

        layout.addWidget(title_label)
        layout.addSpacing(20)
        layout.addWidget(message_label)
        layout.addSpacing(20)
        layout.addWidget(ok_button)
        self.setLayout(layout)


class ReminderManager(QObject):
    sedentary_alert = pyqtSignal()
    posture_alert = pyqtSignal()

    def __init__(self, sedentary_threshold=30, alert_cooldown=300):
        super().__init__()
        self.sedentary_threshold = sedentary_threshold
        self.alert_cooldown = alert_cooldown
        self.last_sedentary_alert_time = 0
        self.last_posture_alert_time = 0

    def check_and_alert(self, activity_status, posture_status):
        current_time = time.time()
        sedentary_minutes = activity_status.get('sedentary_minutes', 0)
        if sedentary_minutes >= self.sedentary_threshold:
            if current_time - self.last_sedentary_alert_time > self.alert_cooldown:
                self.sedentary_alert.emit()
                self.last_sedentary_alert_time = current_time

        if posture_status.get('bad_posture_detected', False) and not activity_status.get('is_exempt_app', False):
            if current_time - self.last_posture_alert_time > self.alert_cooldown:
                self.posture_alert.emit()
                self.last_posture_alert_time = current_time

    def update_settings(self, sedentary_threshold, alert_cooldown):
        self.sedentary_threshold = sedentary_threshold
        self.alert_cooldown = alert_cooldown

class ActivityMonitor(threading.Thread):
    def __init__(self, inactivity_timeout=60, exempt_apps=None):
        super().__init__()
        self.daemon = True
        self.running = False
        self.inactivity_timeout = inactivity_timeout
        self.exempt_apps = exempt_apps or []
        self.last_activity_time = time.time()
        self.is_active = True
        self.current_app = ""
        self.is_exempt_app = False
        self.sedentary_time = 0
        self.lock = threading.Lock()
        self.user_present = False

    def set_user_presence(self, present):
        with self.lock:
            self.user_present = present

    def update_settings(self, inactivity_timeout, exempt_apps):
        self.inactivity_timeout = inactivity_timeout
        self.exempt_apps = exempt_apps
    

    def on_activity(self, *args):
        with self.lock:
            self.last_activity_time = time.time()

    def start_monitoring(self):
        self.running = True
        self.last_activity_time = time.time()
        self.mouse_listener = mouse.Listener(on_move=self.on_activity, on_click=self.on_activity, on_scroll=self.on_activity)
        self.keyboard_listener = keyboard.Listener(on_press=self.on_activity)
        self.mouse_listener.start()
        self.keyboard_listener.start()
        self.start()

    def stop_monitoring(self):
        self.running = False
        if hasattr(self, 'mouse_listener') and self.mouse_listener.is_alive(): self.mouse_listener.stop()
        if hasattr(self, 'keyboard_listener') and self.keyboard_listener.is_alive(): self.keyboard_listener.stop()


    def get_active_window_process(self):
        if sys.platform == 'win32':
            try:
                hwnd = win32gui.GetForegroundWindow()
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                return psutil.Process(pid).name()
            except Exception: return ""
        return ""

    def run(self):
        while self.running:
            try:
                current_time = time.time()
                with self.lock:
                    self.is_active = (current_time - self.last_activity_time) < self.inactivity_timeout
                    self.current_app = self.get_active_window_process()
                    app_lower = self.current_app.lower()
                    self.is_exempt_app = any(exempt.lower() in app_lower for exempt in self.exempt_apps if exempt)

                    if self.is_active and not self.is_exempt_app and self.user_present:
                        self.sedentary_time += 1
                time.sleep(1)
            
            except Exception as e:
                print(f"!!! ActivityMonitor 线程发生错误: {e}")
                time.sleep(5)

    def get_sedentary_time(self):
        with self.lock: return self.sedentary_time / 60.0

    def reset_timer(self):
        with self.lock: self.sedentary_time = 0

    def get_status(self):
        with self.lock:
            return {'is_active': self.is_active, 'current_app': self.current_app,
                    'is_exempt_app': self.is_exempt_app, 'sedentary_minutes': self.sedentary_time / 60.0}

class PostureDetector(threading.Thread):
    def __init__(self, head_down_threshold=20, head_down_duration=3, smoothing_factor=10, 
                 face_recognition_enabled=True, processing_interval_ms=30,
                 torso_check_enabled=True, shoulder_angle_threshold=10, torso_duration=3):
        super().__init__()
        self.daemon = True
        self.running = False
        
        self.cap = None
        self.frame = None
        self.frame_lock = threading.Lock()
        self.latest_results = {}
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.face_3d_model_points = np.array([
            [0.0, 0.0, 0.0],
            [0.0, -330.0, -65.0],  
            [-225.0, 170.0, -135.0], 
            [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0, -150.0, -125.0]
        ])
        
        self.head_down_threshold = head_down_threshold
        self.head_down_duration = head_down_duration
        self.raw_pitch = 0
        self.pitch = 0
        self.is_head_down = False
        self.pitch_history = deque(maxlen=smoothing_factor)
        self.head_down_start_time = None
        
        self.torso_check_enabled = torso_check_enabled
        self.shoulder_angle_threshold = shoulder_angle_threshold
        self.torso_duration = torso_duration
        self.shoulder_angle = 0
        self.is_leaning = False
        self.bad_torso_start_time = None
        self.bad_posture_detected = False
        self.user_detected = False
        self.user_encoding = None
        self.ENCODING_FILE = "user_face_encoding.npy"
        self.user_locked = False
        self.last_recognition_time = 0
        self.RECOGNITION_INTERVAL = 5
        self.face_recognition_enabled = face_recognition_enabled
        self.processing_interval_ms = processing_interval_ms
        self.processing_time_ms = 0
        self.fps_history = deque(maxlen=20)

    def calculate_head_pose(self, landmarks, image_shape):
        h, w = image_shape[:2]
        face_2d_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),
            (landmarks[152].x * w, landmarks[152].y * h),
            (landmarks[33].x * w, landmarks[33].y * h),
            (landmarks[263].x * w, landmarks[263].y * h),
            (landmarks[61].x * w, landmarks[61].y * h),
            (landmarks[291].x * w, landmarks[291].y * h)
        ], dtype="double")

        focal_length = w
        camera_center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, camera_center[0]], [0, focal_length, camera_center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            self.face_3d_model_points, face_2d_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success: return 0, 0, 0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, translation_vector)))
        
        pitch, yaw, roll = euler_angles[0, 0], euler_angles[1, 0], euler_angles[2, 0]
        pitch = -pitch

        if pitch < -90: pitch = 180 + pitch
        elif pitch > 90: pitch = pitch - 180
            
        return pitch, yaw, roll

    def process_posture(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame) if self.torso_check_enabled else None
        self.latest_results = {'face': face_results, 'pose': pose_results}

        if face_results and face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            pitch, _, _ = self.calculate_head_pose(face_landmarks, frame.shape)
            self.raw_pitch = pitch
            self.pitch_history.append(self.raw_pitch)
            self.pitch = np.mean(self.pitch_history)
            self.is_head_down = self.pitch < -self.head_down_threshold
        else:
            self.is_head_down = False

        if self.torso_check_enabled and pose_results and pose_results.pose_landmarks:
            self.shoulder_angle = self.calculate_torso_pose(pose_results.pose_landmarks.landmark, frame.shape)
            self.is_leaning = abs(self.shoulder_angle) > self.shoulder_angle_threshold
        else:
            self.is_leaning = False

        if self.is_head_down:
            if self.head_down_start_time is None: self.head_down_start_time = time.time()
        else:
            self.head_down_start_time = None
        
        if self.is_leaning:
            if self.bad_torso_start_time is None: self.bad_torso_start_time = time.time()
        else:
            self.bad_torso_start_time = None

        persistent_head_down = (self.head_down_start_time is not None and time.time() - self.head_down_start_time > self.head_down_duration)
        persistent_leaning = (self.bad_torso_start_time is not None and time.time() - self.bad_torso_start_time > self.torso_duration)

        if persistent_head_down or persistent_leaning:
            self.bad_posture_detected = True
            
    def load_user_encoding(self):
        if os.path.exists(self.ENCODING_FILE):
            self.user_encoding = np.load(self.ENCODING_FILE)
            return True
        self.user_encoding = None
        return False

    def update_settings(self, head_down_threshold, head_down_duration, smoothing_factor, face_recognition_enabled, processing_interval_ms, torso_check_enabled, shoulder_angle_threshold, torso_duration):
        self.head_down_threshold, self.head_down_duration, self.face_recognition_enabled, self.processing_interval_ms = head_down_threshold, head_down_duration, face_recognition_enabled, processing_interval_ms
        if self.pitch_history.maxlen != smoothing_factor: self.pitch_history = deque(maxlen=smoothing_factor)
        self.torso_check_enabled, self.shoulder_angle_threshold, self.torso_duration = torso_check_enabled, shoulder_angle_threshold, torso_duration

    def start_detection(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): return False
        self.start()
        return True

    def stop_detection(self):
        self.running = False
        if self.is_alive(): self.join()
        if self.cap: self.cap.release()
    
    def calculate_torso_pose(self, landmarks, image_shape):
        h, w = image_shape[:2]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        if left_shoulder.visibility < 0.6 or right_shoulder.visibility < 0.6: return 0
        ls_coords = np.array([left_shoulder.x * w, left_shoulder.y * h])
        rs_coords = np.array([right_shoulder.x * w, right_shoulder.y * h])
        dy, dx = rs_coords[1] - ls_coords[1], rs_coords[0] - ls_coords[0]
        if dx == 0: return 90
        angle = np.degrees(np.arctan2(dy, dx))
        if angle > 90: angle -= 180
        elif angle < -90: angle += 180
        return angle

    def reset_posture_state(self):
        self.raw_pitch, self.pitch, self.is_head_down, self.head_down_start_time = 0, 0, False, None
        self.shoulder_angle, self.is_leaning, self.bad_torso_start_time = 0, False, None
        self.latest_results, self.user_detected, self.user_locked = {}, False, False

    def run(self):
        while self.running:
            start_time = time.perf_counter()
            if not self.cap or not self.cap.isOpened():
                time.sleep(1)
                continue
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            if self.face_recognition_enabled: self.run_with_face_recognition(frame)
            else: self.run_without_face_recognition(frame)

            with self.frame_lock: self.frame = frame.copy()
            
            end_time = time.perf_counter()
            proc_time_s = end_time - start_time
            self.processing_time_ms = proc_time_s * 1000
            sleep_duration_s = max(0.001, self.processing_interval_ms / 1000.0)
            total_cycle_time_s = proc_time_s + sleep_duration_s
            if total_cycle_time_s > 0: self.fps_history.append(1.0 / total_cycle_time_s)
            time.sleep(sleep_duration_s)

    def run_with_face_recognition(self, frame):
        if self.user_encoding is None:
            self.reset_posture_state()
            return
        current_time = time.time()
        perform_recognition = not self.user_locked or (current_time - self.last_recognition_time > self.RECOGNITION_INTERVAL)
        if perform_recognition:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            user_found_in_frame = any(True in face_recognition.compare_faces([self.user_encoding], enc) for enc in face_encodings)
            if user_found_in_frame:
                self.user_locked, self.user_detected, self.last_recognition_time = True, True, current_time
            else:
                self.reset_posture_state()
                return
        if self.user_locked:
            self.process_posture(frame)
            if not self.latest_results.get('face') or not self.latest_results['face'].multi_face_landmarks:
                self.user_locked, self.user_detected = False, False

    def run_without_face_recognition(self, frame):
        self.process_posture(frame)
        face_detected = self.latest_results.get('face') and self.latest_results['face'].multi_face_landmarks
        pose_detected = self.latest_results.get('pose') and self.latest_results['pose'].pose_landmarks
        self.user_detected = face_detected or (self.torso_check_enabled and pose_detected)

    def get_frame(self):
        with self.frame_lock: return self.frame.copy() if self.frame is not None else None
    def get_detection_results(self): return self.latest_results
    def get_status(self):
        return {'pitch': self.pitch, 'is_head_down': self.is_head_down, 'shoulder_angle': self.shoulder_angle, 'is_leaning': self.is_leaning, 'bad_posture_detected': self.bad_posture_detected, 'user_detected': self.user_detected, 'user_enrolled': self.user_encoding is not None, 'face_recognition_enabled': self.face_recognition_enabled, 'fps': np.mean(self.fps_history) if self.fps_history else 0, 'processing_time_ms': self.processing_time_ms}
    def reset_bad_posture_flag(self):
        self.bad_posture_detected, self.head_down_start_time, self.bad_torso_start_time = False, None, None

class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.init_ui()
        self.face_rec_check.stateChanged.connect(self.update_ui_state)
        self.torso_check.stateChanged.connect(self.update_ui_state)
        self.update_ui_state()

    def init_ui(self):
        self.setWindowTitle("设置")
        self.setFixedSize(550, 900)
        layout = QVBoxLayout()
        
        general_group = QGroupBox("通用设置")
        general_layout = QVBoxLayout()
        self.startup_check = QCheckBox("开机时自动启动")
        self.startup_check.setChecked(self.config.getboolean('General', 'autostart', fallback=False))
        self.startup_check.setToolTip("将此程序添加到系统启动项中，方便使用。")
        general_layout.addWidget(self.startup_check)
        general_group.setLayout(general_layout)


        timer_group = QGroupBox("久坐与活动设置")
        timer_layout = QVBoxLayout()
        sedentary_layout = QHBoxLayout()
        sedentary_layout.addWidget(QLabel("久坐提醒间隔 (分钟):"))
        self.sedentary_spin = QSpinBox()
        self.sedentary_spin.setRange(1, 900); self.sedentary_spin.setValue(self.config.getint('Timer', 'sedentary_threshold', fallback=30))
        sedentary_layout.addWidget(self.sedentary_spin)
        timer_layout.addLayout(sedentary_layout)
        inactivity_layout = QHBoxLayout()
        inactivity_layout.addWidget(QLabel("无操作判定 (秒):"))
        self.inactivity_spin = QSpinBox()
        self.inactivity_spin.setRange(1, 300); self.inactivity_spin.setValue(self.config.getint('Timer', 'inactivity_timeout', fallback=60))
        inactivity_layout.addWidget(self.inactivity_spin)
        timer_layout.addLayout(inactivity_layout)
        cooldown_layout = QHBoxLayout()
        cooldown_layout.addWidget(QLabel("提醒冷却时间 (秒):"))
        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(0, 900); self.cooldown_spin.setValue(self.config.getint('Timer', 'alert_cooldown', fallback=300))
        cooldown_layout.addWidget(self.cooldown_spin)
        timer_layout.addLayout(cooldown_layout)
        timer_group.setLayout(timer_layout)

        posture_group = QGroupBox("姿势检测设置")
        posture_layout = QVBoxLayout()
        self.posture_check = QCheckBox("启用头部姿态检测"); self.posture_check.setChecked(self.config.getboolean('Posture', 'posture_check_enabled', fallback=True))
        posture_layout.addWidget(self.posture_check)
        self.face_rec_check = QCheckBox("启用人脸识别 (仅对已录入用户提醒)"); self.face_rec_check.setChecked(self.config.getboolean('Posture', 'face_recognition_enabled', fallback=True)); self.face_rec_check.setToolTip("开启后，只对录入的用户进行姿势提醒，性能开销较大。\n关闭后，会对摄像头前的任何人进行提醒，性能开销较小。")
        posture_layout.addWidget(self.face_rec_check)
        self.show_landmarks_check = QCheckBox("在画面上显示追踪点 (面部和身体)"); self.show_landmarks_check.setChecked(self.config.getboolean('Posture', 'show_landmarks', fallback=False))
        posture_layout.addWidget(self.show_landmarks_check)
        
        threshold_layout = QHBoxLayout(); threshold_layout.addWidget(QLabel("低头角度阈值 (°):"))
        self.angle_spin = QSpinBox(); self.angle_spin.setRange(1, 90); self.angle_spin.setValue(self.config.getint('Posture', 'head_down_threshold', fallback=20))
        threshold_layout.addWidget(self.angle_spin); posture_layout.addLayout(threshold_layout)
        duration_layout = QHBoxLayout(); duration_layout.addWidget(QLabel("低头持续时间 (秒):"))
        self.duration_spin = QSpinBox(); self.duration_spin.setRange(1, 999); self.duration_spin.setValue(self.config.getint('Posture', 'head_down_duration', fallback=5))
        duration_layout.addWidget(self.duration_spin); posture_layout.addLayout(duration_layout)
        smoothing_layout = QHBoxLayout(); smoothing_layout.addWidget(QLabel("角度平滑系数 (帧):"))
        self.smoothing_spin = QSpinBox(); self.smoothing_spin.setRange(1, 30); self.smoothing_spin.setValue(self.config.getint('Posture', 'smoothing_factor', fallback=10))
        smoothing_layout.addWidget(self.smoothing_spin); posture_layout.addLayout(smoothing_layout)

        posture_layout.addSpacing(15)
        self.torso_check = QCheckBox("启用身体姿态检测"); self.torso_check.setChecked(self.config.getboolean('Posture', 'torso_check_enabled', fallback=True)); self.torso_check.setToolTip("检测身体是否倾斜。可能会增加CPU占用。")
        posture_layout.addWidget(self.torso_check)
        shoulder_layout = QHBoxLayout(); shoulder_layout.addWidget(QLabel("身体倾斜角度阈值 (°):"))
        self.shoulder_angle_spin = QSpinBox(); self.shoulder_angle_spin.setRange(1, 999); self.shoulder_angle_spin.setValue(self.config.getint('Posture', 'shoulder_angle_threshold', fallback=10))
        shoulder_layout.addWidget(self.shoulder_angle_spin); posture_layout.addLayout(shoulder_layout)
        torso_duration_layout = QHBoxLayout(); torso_duration_layout.addWidget(QLabel("身体倾斜持续时间 (秒):"))
        self.torso_duration_spin = QSpinBox(); self.torso_duration_spin.setRange(1, 999); self.torso_duration_spin.setValue(self.config.getint('Posture', 'torso_duration', fallback=5))
        torso_duration_layout.addWidget(self.torso_duration_spin); posture_layout.addLayout(torso_duration_layout)
        posture_layout.addSpacing(15)

        break_layout = QHBoxLayout(); break_layout.addWidget(QLabel("强制休息时长 (秒):"))
        self.break_duration_spin = QSpinBox(); self.break_duration_spin.setRange(1, 999); self.break_duration_spin.setValue(self.config.getint('Posture', 'break_duration', fallback=15))
        break_layout.addWidget(self.break_duration_spin); posture_layout.addLayout(break_layout)
        posture_group.setLayout(posture_layout)
        
        perf_group = QGroupBox("性能设置"); perf_layout = QVBoxLayout()
        interval_layout = QHBoxLayout(); interval_layout.addWidget(QLabel("处理间隔 (毫秒):"))
        self.interval_spin = QSpinBox(); self.interval_spin.setRange(10, 200); self.interval_spin.setSingleStep(10); self.interval_spin.setValue(self.config.getint('Posture', 'processing_interval_ms', fallback=30)); self.interval_spin.setToolTip("每次姿势检测循环后的休眠时间。\n值越大，CPU占用越低，但检测频率也越低。")
        interval_layout.addWidget(self.interval_spin); perf_layout.addLayout(interval_layout); perf_group.setLayout(perf_layout)
        
        alert_group = QGroupBox("提醒方式"); alert_layout = QVBoxLayout()
        self.popup_check = QCheckBox("弹窗提醒"); self.popup_check.setChecked(self.config.getboolean('Alert', 'popup_enabled', fallback=True))
        alert_layout.addWidget(self.popup_check)
        autoclose_layout = QHBoxLayout(); autoclose_layout.addWidget(QLabel("普通弹窗自动关闭 (秒):"))
        self.popup_autoclose_spin = QSpinBox(); self.popup_autoclose_spin.setRange(5, 60); self.popup_autoclose_spin.setValue(self.config.getint('Alert', 'popup_autoclose_duration', fallback=10))
        autoclose_layout.addWidget(self.popup_autoclose_spin); alert_layout.addLayout(autoclose_layout); alert_group.setLayout(alert_layout)
        
        app_group = QGroupBox("豁免应用列表 (每行一个，例如 chrome.exe)"); app_layout = QVBoxLayout()
        self.app_list_edit = QTextEdit()
        exempt_apps_list = [app.strip() for app in self.config.get('Apps', 'exempt_apps', fallback='').split(',') if app.strip()]
        self.app_list_edit.setText('\n'.join(exempt_apps_list)); app_layout.addWidget(self.app_list_edit)
        button_layout = QHBoxLayout()
        self.add_app_button = QPushButton("从本地添加..."); self.add_app_button.clicked.connect(self.add_app_from_file)
        self.remove_app_button = QPushButton("删除选中行"); self.remove_app_button.clicked.connect(self.remove_selected_line)
        button_layout.addWidget(self.add_app_button); button_layout.addWidget(self.remove_app_button); button_layout.addStretch()
        app_layout.addLayout(button_layout); app_group.setLayout(app_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)

        layout.addWidget(general_group)
        layout.addWidget(timer_group)
        layout.addWidget(posture_group)
        layout.addWidget(perf_group)
        layout.addWidget(alert_group)
        layout.addWidget(app_group)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def update_ui_state(self):
        is_torso_checked = self.torso_check.isChecked()
        self.shoulder_angle_spin.setEnabled(is_torso_checked)
        self.torso_duration_spin.setEnabled(is_torso_checked)

    def add_app_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择要豁免的应用程序", "", "可执行文件 (*.exe);;所有文件 (*.*)")
        if file_path:
            exe_name = os.path.basename(file_path)
            current_apps_clean = {app.strip() for app in self.app_list_edit.toPlainText().split('\n') if app.strip()}
            if exe_name not in current_apps_clean: self.app_list_edit.append(exe_name)

    def remove_selected_line(self):
        cursor = self.app_list_edit.textCursor()
        if cursor.hasSelection(): cursor.removeSelectedText()
        else: cursor.select(QTextCursor.SelectionType.LineUnderCursor); cursor.removeSelectedText()
        self.app_list_edit.setText('\n'.join([line for line in self.app_list_edit.toPlainText().split('\n') if line.strip()]))

    def get_settings(self):
        exempt_apps_list = [line.strip() for line in self.app_list_edit.toPlainText().split('\n') if line.strip()]
        return {
            'autostart': self.startup_check.isChecked(),
            'sedentary_threshold': self.sedentary_spin.value(), 'inactivity_timeout': self.inactivity_spin.value(),
            'alert_cooldown': self.cooldown_spin.value(),
            'posture_check_enabled': self.posture_check.isChecked(),
            'face_recognition_enabled': self.face_rec_check.isChecked(),
            'head_down_threshold': self.angle_spin.value(), 'head_down_duration': self.duration_spin.value(),
            'smoothing_factor': self.smoothing_spin.value(), 'torso_check_enabled': self.torso_check.isChecked(),
            'shoulder_angle_threshold': self.shoulder_angle_spin.value(),
            'torso_duration': self.torso_duration_spin.value(), 'break_duration': self.break_duration_spin.value(),
            'processing_interval_ms': self.interval_spin.value(), 'popup_enabled': self.popup_check.isChecked(),
            'popup_autoclose_duration': self.popup_autoclose_spin.value(),
            'exempt_apps': ','.join(exempt_apps_list), 'show_landmarks': self.show_landmarks_check.isChecked()}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = configparser.ConfigParser()
        self.load_config()
        self.init_core_modules()
        self.break_dialogs = []
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose

        self.init_ui()
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)
        self.start_monitoring()

    def load_config(self):
        self.config.read('config.ini', encoding='utf-8')
        
        ### NEW: 添加通用设置区 ###
        if not self.config.has_section('General'): self.config.add_section('General')
        self.config.set('General', 'autostart', self.config.get('General', 'autostart', fallback='False'))

        if not self.config.has_section('Timer'): self.config.add_section('Timer')
        self.config.set('Timer', 'sedentary_threshold', self.config.get('Timer', 'sedentary_threshold', fallback='30'))
        self.config.set('Timer', 'inactivity_timeout', self.config.get('Timer', 'inactivity_timeout', fallback='60'))
        self.config.set('Timer', 'alert_cooldown', self.config.get('Timer', 'alert_cooldown', fallback='300'))
        
        if not self.config.has_section('Posture'): self.config.add_section('Posture')
        self.config.set('Posture', 'posture_check_enabled', self.config.get('Posture', 'posture_check_enabled', fallback='True'))
        self.config.set('Posture', 'face_recognition_enabled', self.config.get('Posture', 'face_recognition_enabled', fallback='True'))
        self.config.set('Posture', 'show_landmarks', self.config.get('Posture', 'show_landmarks', fallback='False'))
        self.config.set('Posture', 'head_down_threshold', self.config.get('Posture', 'head_down_threshold', fallback='20'))
        self.config.set('Posture', 'head_down_duration', self.config.get('Posture', 'head_down_duration', fallback='5'))
        self.config.set('Posture', 'break_duration', self.config.get('Posture', 'break_duration', fallback='15'))
        self.config.set('Posture', 'smoothing_factor', self.config.get('Posture', 'smoothing_factor', fallback='10'))
        self.config.set('Posture', 'processing_interval_ms', self.config.get('Posture', 'processing_interval_ms', fallback='30'))
        self.config.set('Posture', 'torso_check_enabled', self.config.get('Posture', 'torso_check_enabled', fallback='True'))
        self.config.set('Posture', 'shoulder_angle_threshold', self.config.get('Posture', 'shoulder_angle_threshold', fallback='10'))
        self.config.set('Posture', 'torso_duration', self.config.get('Posture', 'torso_duration', fallback='5'))

        if not self.config.has_section('Alert'): self.config.add_section('Alert')
        self.config.set('Alert', 'popup_enabled', self.config.get('Alert', 'popup_enabled', fallback='True'))
        self.config.set('Alert', 'popup_autoclose_duration', self.config.get('Alert', 'popup_autoclose_duration', fallback='10'))

        if not self.config.has_section('Apps'): self.config.add_section('Apps')
        self.config.set('Apps', 'exempt_apps', self.config.get('Apps', 'exempt_apps', fallback=''))

        with open('config.ini', 'w', encoding='utf-8') as f: self.config.write(f)
        
        self.set_startup(self.config.getboolean('General', 'autostart'))


    def init_core_modules(self):
        exempt_apps = [app.strip() for app in self.config.get('Apps', 'exempt_apps', fallback='').split(',') if app.strip()]
        self.activity_monitor = ActivityMonitor(inactivity_timeout=self.config.getint('Timer', 'inactivity_timeout'), exempt_apps=exempt_apps)
        
        self.posture_detector = PostureDetector(
            head_down_threshold=self.config.getint('Posture', 'head_down_threshold'),
            head_down_duration=self.config.getint('Posture', 'head_down_duration'),
            smoothing_factor=self.config.getint('Posture', 'smoothing_factor'),
            face_recognition_enabled=self.config.getboolean('Posture', 'face_recognition_enabled'),
            processing_interval_ms=self.config.getint('Posture', 'processing_interval_ms'),
            torso_check_enabled=self.config.getboolean('Posture', 'torso_check_enabled'),
            shoulder_angle_threshold=self.config.getint('Posture', 'shoulder_angle_threshold'),
            torso_duration=self.config.getint('Posture', 'torso_duration'))
        self.posture_detector.load_user_encoding()

        self.reminder_manager = ReminderManager(
            sedentary_threshold=self.config.getint('Timer', 'sedentary_threshold'),
            alert_cooldown=self.config.getint('Timer', 'alert_cooldown'))
        self.reminder_manager.sedentary_alert.connect(self.show_sedentary_alert)
        self.reminder_manager.posture_alert.connect(self.show_posture_alert)

    def init_ui(self):
        self.setWindowTitle("智能久坐与姿势提醒助手")
        self.setGeometry(100, 100, 800, 600)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        self.camera_label = QLabel("摄像头加载中...")
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        main_layout.addWidget(self.camera_label)
        
        info_layout = QHBoxLayout()
        timer_group = QGroupBox("久坐计时")
        timer_layout = QVBoxLayout()
        self.timer_label = QLabel("00:00")
        timer_font = self.timer_label.font(); timer_font.setPointSize(24); timer_font.setBold(True)
        self.timer_label.setFont(timer_font); self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        timer_layout.addWidget(self.timer_label)
        timer_group.setLayout(timer_layout)
        info_layout.addWidget(timer_group)
        
        posture_group = QGroupBox("姿势状态")
        posture_layout = QVBoxLayout()
        self.posture_label = QLabel("等待用户...")
        posture_font = self.posture_label.font(); posture_font.setPointSize(14)
        self.posture_label.setFont(posture_font); self.posture_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        posture_layout.addWidget(self.posture_label)
        posture_group.setLayout(posture_layout)
        info_layout.addWidget(posture_group)
        
        perf_group = QGroupBox("性能开销")
        perf_layout = QVBoxLayout()
        self.performance_label = QLabel("- FPS / - ms")
        perf_font = self.performance_label.font(); perf_font.setPointSize(14)
        self.performance_label.setFont(perf_font); self.performance_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        perf_layout.addWidget(self.performance_label)
        perf_group.setLayout(perf_layout)
        info_layout.addWidget(perf_group)

        activity_group = QGroupBox("活动状态")
        activity_layout = QVBoxLayout()
        self.activity_label = QLabel("活动中")
        activity_font = self.activity_label.font(); activity_font.setPointSize(14)
        self.activity_label.setFont(activity_font); self.activity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        activity_layout.addWidget(self.activity_label)
        self.app_label = QLabel("当前应用: -")
        self.app_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        activity_layout.addWidget(self.app_label)
        activity_group.setLayout(activity_layout)
        info_layout.addWidget(activity_group)

        main_layout.addLayout(info_layout)
        button_layout = QHBoxLayout()
        self.enroll_button = QPushButton("录入我的面部信息"); self.enroll_button.clicked.connect(self.enroll_face)
        button_layout.addWidget(self.enroll_button)
        self.reset_button = QPushButton("重置计时器"); self.reset_button.clicked.connect(self.reset_timer)
        button_layout.addWidget(self.reset_button)
        self.settings_button = QPushButton("设置"); self.settings_button.clicked.connect(self.show_settings)
        button_layout.addWidget(self.settings_button)
        main_layout.addLayout(button_layout)
        central_widget.setLayout(main_layout)
        self.init_tray()

    @pyqtSlot()
    def enroll_face(self):
        frame = self.posture_detector.get_frame()
        if frame is None:
            QMessageBox.warning(self, "错误", "无法获取摄像头画面，请重试。")
            return
        if not self.posture_detector.face_recognition_enabled:
            QMessageBox.information(self, "提示", "人脸识别功能当前已关闭。\n如需录入，请先在“设置”中启用人脸识别。")
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_recognition.face_locations(rgb_frame))
        if len(face_encodings) == 1:
            np.save(self.posture_detector.ENCODING_FILE, face_encodings[0])
            self.posture_detector.load_user_encoding()
            QMessageBox.information(self, "成功", "您的面部信息已成功录入！\n程序现在将只识别您。")
        elif len(face_encodings) > 1: QMessageBox.warning(self, "录入失败", "检测到多张人脸，请确保画面中只有您自己。")
        else: QMessageBox.warning(self, "录入失败", "未检测到人脸，请正对摄像头并确保光线充足。")

    def init_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        tray_menu = QMenu()
        show_action = QAction("显示", self); show_action.triggered.connect(self.show)
        tray_menu.addAction(show_action)
        quit_action = QAction("退出", self); quit_action.triggered.connect(self.quit_application)
        tray_menu.addAction(quit_action)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
        self.tray_icon.activated.connect(lambda r: self.show() if r == QSystemTrayIcon.ActivationReason.DoubleClick else None)
    
    def start_monitoring(self):
        self.activity_monitor.start_monitoring()
        posture_enabled = self.config.getboolean('Posture', 'posture_check_enabled', fallback=True)
        torso_enabled = self.config.getboolean('Posture', 'torso_check_enabled', fallback=True)
        if (posture_enabled or torso_enabled) and not self.posture_detector.start_detection():
            QMessageBox.warning(self, "警告", "无法打开摄像头，姿态检测功能将不可用")

    def update_display(self):
        frame = self.posture_detector.get_frame()
        if frame is not None:
            if self.config.getboolean('Posture', 'show_landmarks', fallback=False):
                results = self.posture_detector.get_detection_results()
                if results:
                    if results.get('face') and results['face'].multi_face_landmarks:
                        for face_landmarks in results['face'].multi_face_landmarks:
                            self.mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks, connections=self.mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    if results.get('pose') and results['pose'].pose_landmarks:
                        self.mp_drawing.draw_landmarks(image=frame, landmark_list=results['pose'].pose_landmarks, connections=self.mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
            h, w, ch = frame.shape
            qt_image = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888).rgbSwapped()
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        
        activity_status = self.activity_monitor.get_status()
        posture_status = self.posture_detector.get_status()

        self.activity_monitor.set_user_presence(posture_status.get('user_detected', False))

        self.activity_label.setText(f"{'✅ 活动中' if activity_status['is_active'] else '💤 空闲'}")
        self.app_label.setText(f"当前应用: {activity_status['current_app']}")
        minutes, seconds = divmod(int(activity_status['sedentary_minutes'] * 60), 60)
        self.timer_label.setText(f"{minutes:02d}:{seconds:02d}")
        
        if posture_status.get('user_detected', False):
            self.reminder_manager.check_and_alert(activity_status, posture_status)

        if not posture_status.get('user_detected', False):
            self.timer_label.setStyleSheet("color: orange;")
            self.timer_label.setToolTip("用户未在摄像头前，久坐计时暂停")
        elif activity_status['is_exempt_app']:
            self.timer_label.setStyleSheet("color: orange;")
            self.timer_label.setToolTip(f"正在运行豁免应用 ({activity_status['current_app']})，久坐计时暂停")
        else:
            self.timer_label.setStyleSheet("") # 恢复默认颜色
            self.timer_label.setToolTip("")


        fps, proc_time = posture_status.get('fps', 0), posture_status.get('processing_time_ms', 0)
        self.performance_label.setText(f"{fps:.1f} FPS / {proc_time:.1f} ms")
        self.performance_label.setStyleSheet("color: green;" if fps > 20 else ("color: orange;" if fps > 10 else "color: red;"))

        if posture_status['face_recognition_enabled']:
            self.enroll_button.setEnabled(True)
            if not posture_status['user_enrolled']: self.posture_label.setText("👤 请先录入面部信息"); self.posture_label.setStyleSheet("color: blue;")
            elif not posture_status['user_detected']: self.posture_label.setText("👀 正在寻找已录入用户..."); self.posture_label.setStyleSheet("color: orange;")
            else: self.update_posture_label_text(posture_status)
        else:
            self.enroll_button.setEnabled(False)
            if not posture_status['user_detected']: self.posture_label.setText("👀 正在寻找任何人..."); self.posture_label.setStyleSheet("color: orange;")
            else: self.update_posture_label_text(posture_status)

        if posture_status.get('user_detected', False):
            self.reminder_manager.check_and_alert(activity_status, posture_status)

    def update_posture_label_text(self, posture_status):
        messages, is_bad = [], False
        if posture_status.get('is_head_down', False): messages.append(f"低头({posture_status['pitch']:.1f}°)"); is_bad = True
        if posture_status.get('is_leaning', False): messages.append(f"身体倾斜({posture_status['shoulder_angle']:.1f}°)"); is_bad = True
        if is_bad: self.posture_label.setText(f"⚠️ {' & '.join(messages)}"); self.posture_label.setStyleSheet("color: red;")
        else: self.posture_label.setText(f"✅ 姿势良好 (俯仰角: {posture_status['pitch']:.1f}°)"); self.posture_label.setStyleSheet("color: green;")


    @pyqtSlot()
    def show_sedentary_alert(self):
        if not self.config.getboolean('Alert', 'popup_enabled', fallback=True):
            return

        self._show_forced_break_dialog(
            title="⏰ 久坐提醒 ⏰",
            message="您已经坐了很长时间了！\n请站起来活动一下，保护您的健康。"
        )
        self.activity_monitor.reset_timer()

    @pyqtSlot()
    def show_posture_alert(self):
        if not self.config.getboolean('Alert', 'popup_enabled', fallback=True):
            return
            
        self._show_forced_break_dialog(
            title="姿势警告 🚨",
            message="检测到持续不良姿势！"
        )
        self.posture_detector.reset_bad_posture_flag()
        
    def _show_forced_break_dialog(self, title, message):
        break_duration = self.config.getint('Posture', 'break_duration', fallback=15)
        screens = QApplication.screens()
        
        self.break_dialogs.clear()

        for i, screen in enumerate(screens):
            current_title = title if i == 0 else ""
            current_message = message if i == 0 else ""

            dialog = ForcedBreakDialog(
                duration=break_duration,
                title=current_title,
                message=current_message
            )
            
            dialog.setGeometry(screen.geometry())
            dialog.showFullScreen()
            self.break_dialogs.append(dialog)

        if self.break_dialogs:
            self.break_dialogs[0].exec()
        
        self.break_dialogs.clear()

        
    def reset_timer(self):
        self.activity_monitor.reset_timer()
        QMessageBox.information(self, "提示", "计时器已重置")

    def show_settings(self):
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            for section, key_prefix in [('General',''), ('Timer', ''), ('Posture', ''), ('Alert', ''), ('Apps', '')]:
                for key, value in settings.items():
                    if key.startswith(key_prefix) or \
                       (section == 'Posture' and key in ['break_duration', 'processing_interval_ms', 'show_landmarks']) or \
                       (section == 'General' and key == 'autostart'):
                        self.config.set(section, key, str(value))
            
            with open('config.ini', 'w', encoding='utf-8') as f: self.config.write(f)
            self.apply_settings()
            QMessageBox.information(self, "提示", "设置已保存")
            
    def apply_settings(self):
        self.set_startup(self.config.getboolean('General', 'autostart'))
        
        self.reminder_manager.update_settings(sedentary_threshold=self.config.getint('Timer', 'sedentary_threshold'), alert_cooldown=self.config.getint('Timer', 'alert_cooldown'))
        self.posture_detector.update_settings(
            head_down_threshold=self.config.getint('Posture', 'head_down_threshold'),
            head_down_duration=self.config.getint('Posture', 'head_down_duration'),
            smoothing_factor=self.config.getint('Posture', 'smoothing_factor'),
            face_recognition_enabled=self.config.getboolean('Posture', 'face_recognition_enabled'),
            processing_interval_ms=self.config.getint('Posture', 'processing_interval_ms'),
            torso_check_enabled=self.config.getboolean('Posture', 'torso_check_enabled'),
            shoulder_angle_threshold=self.config.getint('Posture', 'shoulder_angle_threshold'),
            torso_duration=self.config.getint('Posture', 'torso_duration'))
        exempt_apps = [app.strip() for app in self.config.get('Apps', 'exempt_apps').split(',') if app.strip()]
        self.activity_monitor.update_settings(inactivity_timeout=self.config.getint('Timer', 'inactivity_timeout'), exempt_apps=exempt_apps)

    def set_startup(self, enable: bool):
        if sys.platform == 'win32':
            key = winreg.HKEY_CURRENT_USER
            path = r"Software\Microsoft\Windows\CurrentVersion\Run"

            script_path = os.path.abspath(sys.argv[0])
            command = f'"{sys.executable}" "{script_path}"'
            
            try:
                with winreg.OpenKey(key, path, 0, winreg.KEY_ALL_ACCESS) as reg_key:
                    if enable:
                        winreg.SetValueEx(reg_key, APP_NAME, 0, winreg.REG_SZ, command)
                        print(f"已设置开机启动: {APP_NAME} -> {command}")
                    else:
                        try:
                            winreg.DeleteValue(reg_key, APP_NAME)
                            print(f"已取消开机启动: {APP_NAME}")
                        except FileNotFoundError:
                            print(f"开机启动项 '{APP_NAME}' 不存在，无需删除。")
            except Exception as e:
                print(f"操作注册表失败: {e}")
                QMessageBox.warning(self, "权限错误", f"无法修改开机启动项。\n请尝试以管理员身份运行此程序。\n错误: {e}")
        elif sys.platform == 'darwin':
            # macOS 的实现比较复杂，通常需要创建 .plist 文件
            print("macOS 平台的开机启动功能尚未实现。")
            pass
        elif sys.platform.startswith('linux'):
            print("Linux 平台的开机启动功能尚未实现。")
            pass

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        self.tray_icon.showMessage("程序已最小化到托盘", "点击托盘图标可恢复窗口", QSystemTrayIcon.MessageIcon.Information, 2000)

    def quit_application(self):
        self.activity_monitor.stop_monitoring()
        self.posture_detector.stop_detection()
        QApplication.quit()


def main():
    app = QApplication(sys.argv)

    app.setQuitOnLastWindowClosed(False)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

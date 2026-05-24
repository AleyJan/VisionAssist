import time
import threading
import subprocess
import queue
import math
import struct
import cv2
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO
from speech import Speaker
from reading_mode import ReadingMode
from voice_recognition import VoiceRecognizer
from moondream_mode import SceneDescriber

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
YOLO_N_PATH  = '/home/raspberrypi/VisionAssist/models/yolo11n.onnx'
YOLO_S_PATH  = '/home/raspberrypi/VisionAssist/models/yolo11s.onnx'
PIPER_BIN    = '/home/raspberrypi/piper/piper'
PIPER_MODEL  = '/home/raspberrypi/piper-voices/en_US-lessac-medium.onnx'
CONF_N       = 0.75
CONF_S       = 0.60
IMGSZ        = 320
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
CAMERA_INDEX = 0
STREAM_PORT  = 5000

# Cooldowns
COOLDOWN_APPROACHING = 1
COOLDOWN_STATIONARY  = 3
COOLDOWN_RETREATING  = 6
COOLDOWN_UNKNOWN     = 6

# Motion detection
MIN_MOTION_AREA    = 3000
MIN_UNKNOWN_FRAMES = 4

# Trajectory
TRAJ_HISTORY    = 8
APPROACH_THRESH = 1.12
RETREAT_THRESH  = 0.88
MOVE_THRESH     = 20

# ─────────────────────────────────────────
# TRAPEZOID SAFETY ZONE
# ─────────────────────────────────────────
ZONE_TL = (0.35, 0.35)
ZONE_TR = (0.65, 0.35)
ZONE_BR = (0.95, 1.00)
ZONE_BL = (0.05, 1.00)

TL = (int(ZONE_TL[0] * FRAME_WIDTH), int(ZONE_TL[1] * FRAME_HEIGHT))
TR = (int(ZONE_TR[0] * FRAME_WIDTH), int(ZONE_TR[1] * FRAME_HEIGHT))
BR = (int(ZONE_BR[0] * FRAME_WIDTH), int(ZONE_BR[1] * FRAME_HEIGHT))
BL = (int(ZONE_BL[0] * FRAME_WIDTH), int(ZONE_BL[1] * FRAME_HEIGHT))

TRAP_ZONE = np.array([BL, BR, TR, TL], dtype=np.int32)

# ─────────────────────────────────────────
# PRIORITY CLASSES
# ─────────────────────────────────────────
PRIORITY_CLASSES = {
    "person",
    "car", "motorcycle", "bus", "truck", "bicycle",
    "traffic light", "stop sign", "fire hydrant",
    "chair", "couch", "bed", "dining table", "toilet",
    "bottle", "cup", "laptop", "cell phone",
    "book", "backpack", "handbag", "umbrella",
    "dog", "cat", "tv", "sink", "refrigerator",
    "oven", "clock", "potted plant", "bench",
}


# ─────────────────────────────────────────
# BEEP SYSTEM
# ─────────────────────────────────────────
_beep_running = False
_beep_thread  = None
_beep_lock    = threading.Lock()


def _generate_beep(frequency=880, duration=0.08, volume=0.5):
    sample_rate = 22050
    num_samples = int(sample_rate * duration)
    samples = []
    fade = int(sample_rate * 0.01)
    for i in range(num_samples):
        s = math.sin(2 * math.pi * frequency * i / sample_rate)
        s *= volume
        if i < fade:
            s *= i / fade
        elif i > num_samples - fade:
            s *= (num_samples - i) / fade
        samples.append(int(s * 32767))
    return struct.pack(f'{len(samples)}h', *samples)


def _play_raw(raw_audio):
    try:
        proc = subprocess.Popen(
            ['paplay', '--raw', '--rate=22050',
             '--format=s16le', '--channels=1'],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        proc.stdin.write(raw_audio)
        proc.stdin.close()
        proc.wait()
    except Exception:
        pass


def _beep_loop(interval, frequency, duration, volume):
    global _beep_running
    beep_audio = _generate_beep(frequency, duration, volume)
    while _beep_running:
        _play_raw(beep_audio)
        time.sleep(interval)


def start_beep(level):
    global _beep_running, _beep_thread
    params = {
        1: {'interval': 1.2,  'frequency': 660,
            'duration': 0.08, 'volume': 0.4},
        2: {'interval': 0.7,  'frequency': 770,
            'duration': 0.08, 'volume': 0.5},
        3: {'interval': 0.35, 'frequency': 880,
            'duration': 0.08, 'volume': 0.7},
        4: {'interval': 0.15, 'frequency': 1100,
            'duration': 0.08, 'volume': 0.9},
    }
    p = params.get(level, params[2])
    with _beep_lock:
        _beep_running = False
        if _beep_thread and _beep_thread.is_alive():
            _beep_thread.join(timeout=0.5)
        _beep_running = True
        _beep_thread = threading.Thread(
            target=_beep_loop,
            args=(p['interval'], p['frequency'],
                  p['duration'], p['volume']),
            daemon=True
        )
        _beep_thread.start()


def stop_beep():
    global _beep_running
    with _beep_lock:
        _beep_running = False


def get_beep_level(box_area, frame_area):
    ratio = box_area / frame_area
    if ratio > 0.40:
        return 4
    elif ratio > 0.25:
        return 3
    elif ratio > 0.10:
        return 2
    elif ratio > 0.03:
        return 1
    return 0


# ─────────────────────────────────────────
# NON-BLOCKING PIPER TTS
# ─────────────────────────────────────────
_tts_queue   = queue.Queue(maxsize=1)
_tts_playing = False
_tts_lock    = threading.Lock()


def _tts_worker():
    global _tts_playing
    while True:
        text = _tts_queue.get()
        if text is None:
            break
        with _tts_lock:
            _tts_playing = True
        try:
            piper = subprocess.Popen(
                [PIPER_BIN, '--model', PIPER_MODEL,
                 '--output-raw'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            paplay = subprocess.Popen(
                ['paplay', '--raw', '--rate=22050',
                 '--format=s16le', '--channels=1'],
                stdin=piper.stdout,
                stderr=subprocess.DEVNULL
            )
            piper.stdin.write(text.encode())
            piper.stdin.close()
            paplay.wait()
        except Exception as e:
            print(f"[TTS] Error: {e}")
        finally:
            with _tts_lock:
                _tts_playing = False
        _tts_queue.task_done()


threading.Thread(target=_tts_worker, daemon=True).start()


def speak_alert(text):
    global _tts_playing
    with _tts_lock:
        if _tts_playing:
            return
    while not _tts_queue.empty():
        try:
            _tts_queue.get_nowait()
        except Exception:
            pass
    try:
        _tts_queue.put_nowait(text)
    except Exception:
        pass


# ─────────────────────────────────────────
# SAFETY ZONE HELPERS
# ─────────────────────────────────────────
def is_inside_zone(cx, cy):
    result = cv2.pointPolygonTest(
        TRAP_ZONE, (float(cx), float(cy)), False)
    return result >= 0


def get_zone_direction(cx):
    if cx < FRAME_WIDTH * 0.33:
        return "on your left"
    elif cx < FRAME_WIDTH * 0.66:
        return "directly ahead"
    else:
        return "on your right"


def draw_safety_zone(frame):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [TRAP_ZONE], (0, 255, 255))
    cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)
    cv2.polylines(frame, [TRAP_ZONE], isClosed=True,
                  color=(0, 255, 255), thickness=2)
    label_x = (TL[0] + TR[0]) // 2 - 55
    label_y = TL[1] - 8
    cv2.putText(frame, 'SAFETY ZONE',
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 2)
    left_third  = FRAME_WIDTH // 3
    right_third = 2 * FRAME_WIDTH // 3
    cv2.line(frame, (left_third, TL[1]),
             (left_third, BL[1]), (0, 255, 255), 1)
    cv2.line(frame, (right_third, TR[1]),
             (right_third, BR[1]), (0, 255, 255), 1)
    cv2.putText(frame, 'L', (TL[0]+10, TL[1]+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1)
    cv2.putText(frame, 'C',
                (FRAME_WIDTH//2-8, TL[1]+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1)
    cv2.putText(frame, 'R', (TR[0]-20, TR[1]+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1)
    return frame


# ─────────────────────────────────────────
# TRAJECTORY TRACKER
# ─────────────────────────────────────────
class TrajectoryTracker:
    def __init__(self):
        self.size_history = {}
        self.pos_history  = {}

    def update(self, label, direction, box_area, cx):
        key = (label, direction)
        if key not in self.size_history:
            self.size_history[key] = []
        self.size_history[key].append(box_area)
        if len(self.size_history[key]) > TRAJ_HISTORY:
            self.size_history[key].pop(0)
        if key not in self.pos_history:
            self.pos_history[key] = []
        self.pos_history[key].append(cx)
        if len(self.pos_history[key]) > TRAJ_HISTORY:
            self.pos_history[key].pop(0)
        return self._compute(key)

    def _compute(self, key):
        sizes     = self.size_history.get(key, [])
        positions = self.pos_history.get(key, [])
        if len(sizes) < 4:
            return "STATIONARY", "NONE"
        mid         = len(sizes) // 2
        older_size  = sum(sizes[:mid]) / mid
        recent_size = sum(sizes[mid:]) / (len(sizes) - mid)
        size_traj = "STATIONARY"
        if older_size > 0:
            ratio = recent_size / older_size
            if ratio > APPROACH_THRESH:
                size_traj = "APPROACHING"
            elif ratio < RETREAT_THRESH:
                size_traj = "RETREATING"
        pos_traj = "NONE"
        if len(positions) >= 4:
            older_pos  = sum(positions[:mid]) / mid
            recent_pos = sum(positions[mid:]) / \
                         (len(positions) - mid)
            delta = recent_pos - older_pos
            if delta > MOVE_THRESH:
                pos_traj = "MOVING RIGHT"
            elif delta < -MOVE_THRESH:
                pos_traj = "MOVING LEFT"
        return size_traj, pos_traj

    def clean_old(self, active_keys):
        for key in list(self.size_history.keys()):
            if key not in active_keys:
                del self.size_history[key]
        for key in list(self.pos_history.keys()):
            if key not in active_keys:
                del self.pos_history[key]


# ─────────────────────────────────────────
# FLASK WEB STREAM
# ─────────────────────────────────────────
app_flask    = Flask(__name__)
latest_frame = None
frame_lock   = threading.Lock()


def set_latest_frame(frame):
    global latest_frame
    with frame_lock:
        latest_frame = frame.copy()


def generate_stream():
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            frame = latest_frame.copy()
        ret, jpeg = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + jpeg.tobytes()
               + b'\r\n')
        time.sleep(0.05)


@app_flask.route('/')
def index():
    return '''
    <html>
    <head>
        <title>VisionAssist</title>
        <style>
            body{background:#111;color:#fff;
                 font-family:sans-serif;
                 text-align:center;margin:0;padding:20px;}
            h1{color:#00ffff;margin-bottom:10px;}
            img{border:2px solid #00ffff;
                border-radius:8px;max-width:100%;}
            .legend{margin-top:15px;display:flex;
                    justify-content:center;gap:20px;
                    font-size:13px;flex-wrap:wrap;}
            .dot{display:inline-block;width:14px;
                 height:14px;border-radius:3px;
                 margin-right:5px;vertical-align:middle;}
            .red{background:#ff3333;}
            .green{background:#33ff33;}
            .yellow{background:#ffff00;}
            .orange{background:#ff8800;}
        </style>
    </head>
    <body>
        <h1>VisionAssist — Safety Zone</h1>
        <img src="/video_feed"/>
        <div class="legend">
            <span><span class="dot yellow"></span>
                Safety Zone</span>
            <span><span class="dot red"></span>
                Inside (ALERT)</span>
            <span><span class="dot green"></span>
                Outside (safe)</span>
            <span><span class="dot orange"></span>
                Unknown</span>
        </div>
    </body>
    </html>
    '''


@app_flask.route('/video_feed')
def video_feed():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def start_flask():
    app_flask.run(host='0.0.0.0', port=STREAM_PORT,
                  debug=False, threaded=True)


# ─────────────────────────────────────────
# THREADED CAMERA
# ─────────────────────────────────────────
class ThreadedCamera:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam!")
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True
        threading.Thread(target=self._capture_loop,
                         daemon=True).start()
        time.sleep(0.5)
        print(f"[Camera] Threaded capture at /dev/video{index}")

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()
        print("[Camera] Stopped")


# ─────────────────────────────────────────
# MOTION DETECTOR
# ─────────────────────────────────────────
class MotionDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=60,
            detectShadows=False)
        self.warmup_frames  = 0
        self.unknown_streak = {}
        print("[Motion] Background subtractor ready!")

    def detect_motion(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        self.warmup_frames += 1
        if self.warmup_frames < 30:
            return []
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(
            fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(
            fg_mask, kernel, iterations=2)
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        regions     = []
        active_dirs = set()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_MOTION_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2
            if not is_inside_zone(cx, cy):
                continue
            direction = get_zone_direction(cx)
            active_dirs.add(direction)
            self.unknown_streak[direction] = \
                self.unknown_streak.get(direction, 0) + 1
            if self.unknown_streak[direction] < \
               MIN_UNKNOWN_FRAMES:
                continue
            regions.append({
                'cx': cx, 'cy': cy,
                'x1': x, 'y1': y,
                'x2': x+w, 'y2': y+h,
                'area': area,
                'direction': direction
            })
        for d in list(self.unknown_streak.keys()):
            if d not in active_dirs:
                self.unknown_streak[d] = 0
        return regions


# ─────────────────────────────────────────
# DETECTION ENGINE
# ─────────────────────────────────────────
class DetectionEngine:
    def __init__(self):
        print("[Camera] Starting USB webcam...")
        self.camera  = ThreadedCamera(CAMERA_INDEX)
        self.motion  = MotionDetector()
        self.tracker = TrajectoryTracker()
        print("[YOLO] Loading models...")
        self.model_n     = YOLO(YOLO_N_PATH, task='detect')
        self.model_s     = YOLO(YOLO_S_PATH, task='detect')
        self.frame_count = 0
        print("[YOLO] Both models ready!")

    def enhance_frame(self, frame):
        return cv2.convertScaleAbs(frame, alpha=1.1, beta=5)

    def detect(self):
        self.frame_count += 1
        frame = self.camera.read()
        if frame is None:
            return None, None, [], [], []

        enhanced = self.enhance_frame(frame)
        results_n = self.model_n(enhanced, verbose=False,
                                  conf=CONF_N, imgsz=IMGSZ)
        results_s = None
        if self.frame_count % 10 == 0:
            results_s = self.model_s(enhanced, verbose=False,
                                      conf=CONF_S, imgsz=IMGSZ)

        display         = frame.copy()
        inside_zone     = []
        outside_zone    = []
        seen            = {}
        yolo_zone_boxes = []
        active_keys     = []

        for results in [results_n, results_s]:
            if results is None:
                continue
            for box in results[0].boxes:
                label = results[0].names[int(box.cls)]
                if label not in PRIORITY_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx        = int((x1 + x2) / 2)
                cy        = int((y1 + y2) / 2)
                conf      = float(box.conf)
                box_area  = (x2 - x1) * (y2 - y1)
                inside    = is_inside_zone(cx, cy)
                direction = get_zone_direction(cx)

                key = (label, inside)
                if key in seen and seen[key] >= conf:
                    continue
                seen[key] = conf

                size_traj = "STATIONARY"
                pos_traj  = "NONE"

                if inside:
                    traj_key = (label, direction)
                    active_keys.append(traj_key)
                    size_traj, pos_traj = self.tracker.update(
                        label, direction, box_area, cx)

                if inside:
                    color = (0, 0, 255) if \
                        size_traj == "APPROACHING" \
                        else (0, 50, 200)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(display,
                               (int(x1), int(y1)),
                               (int(x2), int(y2)),
                               color, 2)
                cv2.circle(display, (cx, cy), 5, color, -1)

                symbol = ""
                if inside:
                    if size_traj == "APPROACHING":
                        symbol = ">>"
                    elif size_traj == "RETREATING":
                        symbol = "<<"
                    if pos_traj == "MOVING LEFT":
                        symbol += "<"
                    elif pos_traj == "MOVING RIGHT":
                        symbol += ">"

                cv2.putText(display,
                             f"{label} {symbol} {conf:.2f}",
                             (int(x1), int(y1) - 8),
                             cv2.FONT_HERSHEY_SIMPLEX,
                             0.5, color, 2)

                det = {
                    "label":     label,
                    "cx":        cx,
                    "cy":        cy,
                    "conf":      conf,
                    "box_area":  box_area,
                    "direction": direction,
                    "size_traj": size_traj,
                    "pos_traj":  pos_traj,
                }
                if inside:
                    inside_zone.append(det)
                    yolo_zone_boxes.append(
                        (int(x1), int(y1),
                         int(x2), int(y2)))
                else:
                    outside_zone.append(det)

        self.tracker.clean_old(active_keys)

        motion_regions = self.motion.detect_motion(frame)
        unknown_motion = []
        for region in motion_regions:
            cx = region['cx']
            cy = region['cy']
            overlaps = any(
                bx1 < cx < bx2 and by1 < cy < by2
                for (bx1, by1, bx2, by2) in yolo_zone_boxes
            )
            if not overlaps:
                cv2.rectangle(display,
                               (region['x1'], region['y1']),
                               (region['x2'], region['y2']),
                               (0, 165, 255), 2)
                cv2.putText(display, 'UNKNOWN',
                             (region['x1'],
                              region['y1'] - 8),
                             cv2.FONT_HERSHEY_SIMPLEX,
                             0.5, (0, 165, 255), 2)
                unknown_motion.append(region)

        draw_safety_zone(display)
        return frame, display, inside_zone, \
               outside_zone, unknown_motion

    def capture(self):
        return self.camera.read()

    def stop(self):
        self.camera.stop()


# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
class VisionAssistApp:
    def __init__(self):
        print("[System] Initializing VisionAssist...")
        self.speaker  = Speaker()
        self.detector = DetectionEngine()
        self.reader   = ReadingMode(self.speaker)
        self.scene    = SceneDescriber(self.speaker)
        self.voice    = VoiceRecognizer(
            self.on_voice_command,
            on_listening=self.on_listening,
            on_stop_listening=self.on_stop_listening
        )

        self.last_announced = {}
        self.last_unknown   = 0
        self.nav_count      = 0
        self.mode           = "NAVIGATION"
        self.running        = True
        self.waiting_since  = None
        self.is_listening   = False
        self.fps_times      = []
        self.fps            = 0

        print("[System] All components ready!")

    def _go_navigation(self):
        if self.mode == "READING":
            self.reader.stop()
        self.waiting_since = None
        self.mode = "NAVIGATION"
        print("[System] Navigation mode active.")

    def on_listening(self):
        """Pause everything when user says hey vision."""
        self.is_listening = True
        stop_beep()
        print("[Voice] Pausing — listening for command...")

    def on_stop_listening(self):
        """Resume after command received or timed out."""
        self.is_listening = False
        print("[Voice] Resuming navigation...")

    def update_fps(self):
        now = time.time()
        self.fps_times.append(now)
        self.fps_times = [t for t in self.fps_times
                          if now - t < 1.0]
        self.fps = len(self.fps_times)

    def get_cooldown(self, size_traj):
        if size_traj == "APPROACHING":
            return COOLDOWN_APPROACHING
        elif size_traj == "RETREATING":
            return COOLDOWN_RETREATING
        return COOLDOWN_STATIONARY

    def should_announce(self, key, cooldown):
        return (time.time() -
                self.last_announced.get(key, 0)) > cooldown

    def build_message(self, det):
        label     = det['label']
        direction = det['direction']
        size_traj = det['size_traj']
        pos_traj  = det['pos_traj']
        if pos_traj == "MOVING LEFT":
            return f"{label} moving to your left"
        elif pos_traj == "MOVING RIGHT":
            return f"{label} moving to your right"
        if size_traj == "APPROACHING":
            return f"{label} approaching {direction}"
        elif size_traj == "RETREATING":
            return f"{label} moving away {direction}"
        return f"{label} {direction}"

    def run_navigation(self):
        # Pause navigation while user is speaking
        if self.is_listening:
            time.sleep(0.1)
            return

        frame, display, inside_zone, outside_zone, \
            unknown_motion = self.detector.detect()

        if frame is None:
            return

        self.update_fps()
        self.nav_count += 1

        if display is not None:
            cv2.putText(display, f'Mode: {self.mode}',
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.putText(display, f'FPS: {self.fps}',
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)
            cv2.putText(display,
                        f'IN:{len(inside_zone)} '
                        f'OUT:{len(outside_zone)} '
                        f'UNK:{len(unknown_motion)}',
                        (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1)
            # Show listening status on screen
            if self.is_listening:
                cv2.putText(display, 'LISTENING...',
                            (FRAME_WIDTH//2 - 70, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)
            set_latest_frame(display)

        now            = time.time()
        announced      = False
        max_beep_level = 0

        def priority(d):
            if d['size_traj'] == 'APPROACHING':
                return 0
            if d['pos_traj'] in ('MOVING LEFT',
                                  'MOVING RIGHT'):
                return 1
            if d['size_traj'] == 'STATIONARY':
                return 2
            return 3

        for det in sorted(inside_zone, key=priority):
            label     = det['label']
            direction = det['direction']
            cooldown  = self.get_cooldown(det['size_traj'])
            ann_key   = (label, direction, det['size_traj'])

            beep_lvl = get_beep_level(
                det['box_area'],
                FRAME_WIDTH * FRAME_HEIGHT
            )
            if beep_lvl > max_beep_level:
                max_beep_level = beep_lvl

            if self.should_announce(ann_key, cooldown):
                self.last_announced[ann_key] = now
                msg = self.build_message(det)
                speak_alert(msg)
                print(f"[ZONE ALERT] {msg}")
                announced = True
                break

        # Update beep
        if inside_zone and max_beep_level > 0:
            start_beep(max_beep_level)
        else:
            stop_beep()

        # Unknown motion
        if unknown_motion and not announced:
            if now - self.last_unknown > COOLDOWN_UNKNOWN:
                biggest = max(unknown_motion,
                              key=lambda r: r['area'])
                direction = biggest['direction']
                msg = f"unknown object {direction}"
                speak_alert(msg)
                self.last_unknown = now
                print(f"[UNKNOWN] {msg}")
                if max_beep_level == 0:
                    start_beep(1)

        if self.nav_count % 30 == 0:
            print(f"[FPS] {self.fps} | "
                  f"IN:{len(inside_zone)} | "
                  f"OUT:{len(outside_zone)} | "
                  f"UNK:{len(unknown_motion)}",
                  end='\r')

    def on_voice_command(self, cmd):
        print(f"[Voice] Command: {cmd}")
        if cmd == "navigate":
            self._go_navigation()
            self.speaker.speak("Navigation mode active.")
        elif cmd == "read":
            self._go_navigation()
            self.mode = "READING"
            self.reader.start()
        elif cmd == "describe":
            self._go_navigation()
            self.mode = "WAITING"
            self.waiting_since = time.time()
            frame, display, inside, outside, unknown = \
                self.detector.detect()
            self.scene.describe(inside + outside)
            print("[System] Auto-returning in 8 seconds...")
        elif cmd == "where am i":
            self._go_navigation()
            self.mode = "WAITING"
            self.waiting_since = time.time()
            frame, display, inside, outside, unknown = \
                self.detector.detect()
            self.scene.where_am_i(inside + outside)
            print("[System] Auto-returning in 8 seconds...")
        elif cmd == "help":
            self._go_navigation()
            self.mode = "WAITING"
            self.waiting_since = time.time()
            self.speaker.speak(
                "Commands: hey vision read this. "
                "hey vision stop reading. "
                "hey vision describe. "
                "hey vision where am i. "
                "hey vision start."
            )

    def run(self):
        threading.Thread(target=start_flask,
                         daemon=True).start()
        print(f"[Stream] Live view at "
              f"http://192.168.1.10:{STREAM_PORT}")
        self.voice.start()
        self.speaker.speak(
            "VisionAssist ready. Navigation mode active.")
        print("[System] Running. Ctrl+C to stop.")

        try:
            while self.running:
                if self.mode == "NAVIGATION":
                    self.run_navigation()

                elif self.mode == "READING":
                    if self.is_listening:
                        time.sleep(0.1)
                        continue
                    frame = self.detector.capture()
                    if frame is not None:
                        display = frame.copy()
                        draw_safety_zone(display)
                        cv2.putText(display, 'Mode: READING',
                                    (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2)
                        set_latest_frame(display)
                        self.reader.read_frame(frame)
                        time.sleep(0.1)

                elif self.mode == "WAITING":
                    frame = self.detector.capture()
                    if frame is not None:
                        display = frame.copy()
                        draw_safety_zone(display)
                        cv2.putText(display, 'Mode: WAITING',
                                    (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 165, 0), 2)
                        set_latest_frame(display)
                    time.sleep(0.2)
                    if self.waiting_since and \
                       time.time() - self.waiting_since > 8:
                        print("[System] Auto-returning...")
                        self._go_navigation()

                else:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        print("[System] Shutting down...")
        stop_beep()
        self.running = False
        self.voice.stop()
        self.detector.stop()
        print("[System] All systems offline.")


if __name__ == "__main__":
    app = VisionAssistApp()
    app.run()

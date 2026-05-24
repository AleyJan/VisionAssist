import time
import threading
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
CONF_N       = 0.75
CONF_S       = 0.60
IMGSZ        = 320
COOLDOWN     = 4
MIN_FRAMES   = 2
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
CAMERA_INDEX = 0
STREAM_PORT  = 5000   # open http://192.168.1.7:5000 in browser

# ─────────────────────────────────────────
# SAFETY ZONE CONFIG
# ─────────────────────────────────────────
ZONE_LEFT   = 0.20
ZONE_RIGHT  = 0.80
ZONE_TOP    = 0.10
ZONE_BOTTOM = 1.00

ZX1 = int(FRAME_WIDTH  * ZONE_LEFT)
ZY1 = int(FRAME_HEIGHT * ZONE_TOP)
ZX2 = int(FRAME_WIDTH  * ZONE_RIGHT)
ZY2 = int(FRAME_HEIGHT * ZONE_BOTTOM)

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
# SAFETY ZONE HELPERS
# ─────────────────────────────────────────
def is_inside_zone(cx, cy):
    return ZX1 < cx < ZX2 and ZY1 < cy < ZY2


def get_zone_direction(cx):
    zone_width = ZX2 - ZX1
    third = zone_width / 3
    if cx < ZX1 + third:
        return "on your left"
    elif cx < ZX1 + 2 * third:
        return "directly ahead"
    else:
        return "on your right"


def draw_safety_zone(frame):
    # Yellow zone border
    cv2.rectangle(frame, (ZX1, ZY1), (ZX2, ZY2),
                  (0, 255, 255), 2)
    # Zone label
    cv2.putText(frame, 'SAFETY ZONE',
                (ZX1 + 5, ZY1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2)
    # Dividers
    zone_width = ZX2 - ZX1
    third = zone_width // 3
    cv2.line(frame, (ZX1 + third, ZY1),
             (ZX1 + third, ZY2), (0, 255, 255), 1)
    cv2.line(frame, (ZX1 + 2*third, ZY1),
             (ZX1 + 2*third, ZY2), (0, 255, 255), 1)
    # Section labels
    cv2.putText(frame, 'LEFT',
                (ZX1 + 5, ZY1 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 255), 1)
    cv2.putText(frame, 'CENTER',
                (ZX1 + third + 5, ZY1 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 255), 1)
    cv2.putText(frame, 'RIGHT',
                (ZX1 + 2*third + 5, ZY1 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 255), 1)
    return frame


# ─────────────────────────────────────────
# FLASK WEB STREAM
# ─────────────────────────────────────────
app_flask = Flask(__name__)
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
        ret, jpeg = cv2.imencode('.jpg', frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, 70])
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
        <title>VisionAssist — Live View</title>
        <style>
            body {
                background: #111;
                color: #fff;
                font-family: sans-serif;
                text-align: center;
                margin: 0;
                padding: 20px;
            }
            h1 { color: #00ffff; margin-bottom: 10px; }
            img {
                border: 2px solid #00ffff;
                border-radius: 8px;
                max-width: 100%;
            }
            .legend {
                margin-top: 15px;
                display: flex;
                justify-content: center;
                gap: 30px;
                font-size: 14px;
            }
            .dot {
                display: inline-block;
                width: 14px; height: 14px;
                border-radius: 3px;
                margin-right: 6px;
                vertical-align: middle;
            }
            .red   { background: #ff3333; }
            .green { background: #33ff33; }
            .yellow{ background: #ffff00; }
        </style>
    </head>
    <body>
        <h1>VisionAssist — Safety Zone Live View</h1>
        <img src="/video_feed" />
        <div class="legend">
            <span><span class="dot yellow"></span>Safety Zone</span>
            <span><span class="dot red"></span>Inside Zone (ALERT)</span>
            <span><span class="dot green"></span>Outside Zone (safe)</span>
        </div>
        <p style="color:#888; margin-top:10px; font-size:12px;">
            Auto-refreshing · http://192.168.1.7:5000
        </p>
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
# DETECTION ENGINE
# ─────────────────────────────────────────
class DetectionEngine:
    def __init__(self):
        print("[Camera] Starting USB webcam...")
        self.camera = ThreadedCamera(CAMERA_INDEX)
        print("[YOLO] Loading models...")
        self.model_n     = YOLO(YOLO_N_PATH, task='detect')
        self.model_s     = YOLO(YOLO_S_PATH, task='detect')
        self.frame_count = 0
        print("[YOLO] Both models ready!")

    def enhance_frame(self, frame):
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=5)
        return frame

    def detect(self):
        self.frame_count += 1
        frame = self.camera.read()
        if frame is None:
            return None, None, [], []

        enhanced = self.enhance_frame(frame)

        results_n = self.model_n(enhanced, verbose=False,
                                  conf=CONF_N, imgsz=IMGSZ)
        results_s = None
        if self.frame_count % 10 == 0:
            results_s = self.model_s(enhanced, verbose=False,
                                      conf=CONF_S, imgsz=IMGSZ)

        display      = frame.copy()
        inside_zone  = []
        outside_zone = []
        seen         = {}

        for results in [results_n, results_s]:
            if results is None:
                continue
            for box in results[0].boxes:
                label = results[0].names[int(box.cls)]
                if label not in PRIORITY_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx   = int((x1 + x2) / 2)
                cy   = int((y1 + y2) / 2)
                conf = float(box.conf)
                inside = is_inside_zone(cx, cy)

                key = (label, inside)
                if key in seen and seen[key] >= conf:
                    continue
                seen[key] = conf

                # RED = inside zone, GREEN = outside
                color = (0, 0, 255) if inside else (0, 255, 0)

                cv2.rectangle(display,
                               (int(x1), int(y1)),
                               (int(x2), int(y2)),
                               color, 2)
                cv2.circle(display, (cx, cy), 5, color, -1)
                cv2.putText(display,
                             f"{label} {conf:.2f}",
                             (int(x1), int(y1) - 8),
                             cv2.FONT_HERSHEY_SIMPLEX,
                             0.55, color, 2)

                det = {
                    "label":     label,
                    "cx":        cx,
                    "cy":        cy,
                    "conf":      conf,
                    "direction": get_zone_direction(cx) if inside
                                 else self._outer_dir(cx)
                }
                if inside:
                    inside_zone.append(det)
                else:
                    outside_zone.append(det)

        draw_safety_zone(display)
        return frame, display, inside_zone, outside_zone

    def _outer_dir(self, cx):
        if cx < FRAME_WIDTH * 0.33:
            return "on your left"
        elif cx < FRAME_WIDTH * 0.66:
            return "in front of you"
        return "on your right"

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
        self.voice    = VoiceRecognizer(self.on_voice_command)

        self.last_announced = {}
        self.nav_count      = 0
        self.mode           = "NAVIGATION"
        self.running        = True
        self.waiting_since  = None
        self.fps_times      = []
        self.fps            = 0

        print("[System] All components ready!")

    def _go_navigation(self):
        if self.mode == "READING":
            self.reader.stop()
        self.waiting_since = None
        self.mode = "NAVIGATION"
        print("[System] Navigation mode active.")

    def update_fps(self):
        now = time.time()
        self.fps_times.append(now)
        self.fps_times = [t for t in self.fps_times if now - t < 1.0]
        self.fps = len(self.fps_times)

    def should_announce(self, key):
        return time.time() - self.last_announced.get(key, 0) > COOLDOWN

    def run_navigation(self):
        frame, display, inside_zone, outside_zone = \
            self.detector.detect()

        if frame is None:
            return

        self.update_fps()
        self.nav_count += 1

        # Add HUD info to display
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
                        f'IN ZONE: {len(inside_zone)}  '
                        f'OUTSIDE: {len(outside_zone)}',
                        (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (200, 200, 200), 1)
            # Send to web stream
            set_latest_frame(display)

        # Audio — inside zone only
        now = time.time()
        for det in inside_zone:
            label     = det['label']
            direction = det['direction']
            ann_key   = (label, direction)
            if self.should_announce(ann_key):
                self.last_announced[ann_key] = now
                message = f"{label} {direction}"
                self.speaker.speak(message)
                print(f"[ZONE ALERT] {message}")

        if self.nav_count % 30 == 0:
            print(f"[FPS] {self.fps} | "
                  f"IN: {len(inside_zone)} | "
                  f"OUT: {len(outside_zone)}", end='\r')

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
            frame, display, inside, outside = self.detector.detect()
            self.scene.describe(inside + outside)
            print("[System] Auto-returning in 8 seconds...")
        elif cmd == "where am i":
            self._go_navigation()
            self.mode = "WAITING"
            self.waiting_since = time.time()
            frame, display, inside, outside = self.detector.detect()
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
        # Start Flask stream in background thread
        threading.Thread(target=start_flask,
                         daemon=True).start()
        print(f"[Stream] Live view at http://192.168.1.7:{STREAM_PORT}")

        self.voice.start()
        self.speaker.speak("VisionAssist ready. Navigation mode active.")
        print("[System] Running. Ctrl+C to stop.")

        try:
            while self.running:
                if self.mode == "NAVIGATION":
                    self.run_navigation()

                elif self.mode == "READING":
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
                        print("[System] Auto-returning to navigation...")
                        self._go_navigation()

                else:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        print("[System] Shutting down...")
        self.running = False
        self.voice.stop()
        self.detector.stop()
        print("[System] All systems offline.")


if __name__ == "__main__":
    app = VisionAssistApp()
    app.run()

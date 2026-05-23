import time
import threading
import cv2
import numpy as np
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
CONF_N       = 0.80
CONF_S       = 0.65
IMGSZ        = 320
COOLDOWN     = 6
MIN_FRAMES   = 3
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
CAMERA_INDEX = 0
SHOW_WINDOW  = False

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
    "dog", "cat",
}


# ─────────────────────────────────────────
# THREADED CAMERA
# ─────────────────────────────────────────
class ThreadedCamera:
    """
    Captures frames continuously in background thread.
    YOLO always gets latest frame without waiting for camera.
    """
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

        threading.Thread(target=self._capture_loop, daemon=True).start()
        time.sleep(0.5)
        print(f"[Camera] Threaded capture started at /dev/video{index}")

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
    """
    Handles threaded camera, preprocessing, and dual YOLO inference.
    YOLO11n runs every frame for real-time safety.
    YOLO11s runs every 10 frames for small object detection.
    """
    def __init__(self):
        print("[Camera] Starting USB webcam...")
        self.camera = ThreadedCamera(CAMERA_INDEX)

        print("[YOLO] Loading models...")
        self.model_n    = YOLO(YOLO_N_PATH, task='detect')
        self.model_s    = YOLO(YOLO_S_PATH, task='detect')
        self.frame_count = 0
        print("[YOLO] Both models ready!")

    def enhance_frame(self, frame):
        """Mild preprocessing — A4Tech webcam has good quality."""
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=5)
        return frame

    def get_direction(self, center_x):
        if center_x < FRAME_WIDTH * 0.33:
            return "on your left"
        elif center_x > FRAME_WIDTH * 0.66:
            return "on your right"
        return "in front of you"

    def get_urgency(self, box_area):
        ratio = box_area / (FRAME_WIDTH * FRAME_HEIGHT)
        if ratio > 0.35:
            return "very close", True
        elif ratio > 0.12:
            return "ahead", False
        return "nearby", False

    def detect(self):
        """
        Gets latest frame from threaded camera.
        Runs both YOLO models and returns detections.
        """
        self.frame_count += 1
        frame = self.camera.read()

        if frame is None:
            return None, None, []

        enhanced = self.enhance_frame(frame)

        # YOLO11n every frame
        results_n = self.model_n(enhanced, verbose=False,
                                  conf=CONF_N, imgsz=IMGSZ)

        # YOLO11s every 10 frames
        results_s = None
        if self.frame_count % 10 == 0:
            results_s = self.model_s(enhanced, verbose=False,
                                      conf=CONF_S, imgsz=IMGSZ)

        # Collect raw detections
        raw = []
        for results in [results_n, results_s]:
            if results is None:
                continue
            for box in results[0].boxes:
                label = results[0].names[int(box.cls)]
                if label not in PRIORITY_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center_x = (x1 + x2) / 2
                box_area = (x2 - x1) * (y2 - y1)
                direction = self.get_direction(center_x)
                urgency, is_urgent = self.get_urgency(box_area)
                conf = float(box.conf)
                raw.append({
                    "label":     label,
                    "direction": direction,
                    "urgency":   urgency,
                    "is_urgent": is_urgent,
                    "conf":      conf
                })

        # Deduplicate — keep highest confidence per label+direction
        seen = {}
        for det in raw:
            key = (det['label'], det['direction'])
            if key not in seen or det['conf'] > seen[key]['conf']:
                seen[key] = det

        annotated = results_n[0].plot()
        return frame, annotated, list(seen.values())

    def capture(self):
        """Get latest frame without running detection."""
        return self.camera.read()

    def stop(self):
        self.camera.stop()


# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
class VisionAssistApp:
    """
    Main application controller.

    Modes:
      NAVIGATION — YOLO runs continuously, announces objects
      READING    — EasyOCR reads text in background, camera stays live
      WAITING    — idle after describe/where, auto-returns after 8s

    Wake word: hey vision
    """
    def __init__(self):
        print("[System] Initializing VisionAssist...")
        self.speaker  = Speaker()
        self.detector = DetectionEngine()
        self.reader   = ReadingMode(self.speaker)
        self.scene    = SceneDescriber(self.speaker)
        self.voice    = VoiceRecognizer(self.on_voice_command)

        self.frame_memory   = {}
        self.last_announced = {}
        self.mode           = "NAVIGATION"
        self.running        = True
        self.waiting_since  = None
        self.nav_count      = 0

        # FPS tracking
        self.fps_times = []
        self.fps       = 0

        print("[System] All components ready!")

    def _go_navigation(self):
        """Switch cleanly to navigation mode from any mode."""
        if self.mode == "READING":
            self.reader.stop()
        self.frame_memory.clear()
        self.waiting_since = None
        self.mode = "NAVIGATION"
        print("[System] Navigation mode active.")

    def update_stability(self, detected_labels):
        for label in detected_labels:
            self.frame_memory[label] = self.frame_memory.get(label, 0) + 1
        for label in list(self.frame_memory.keys()):
            if label not in detected_labels:
                self.frame_memory[label] -= 1
                if self.frame_memory[label] <= 0:
                    del self.frame_memory[label]

    def should_announce(self, label):
        stable = self.frame_memory.get(label, 0) >= MIN_FRAMES
        cooled = time.time() - self.last_announced.get(label, 0) > COOLDOWN
        return stable and cooled

    def update_fps(self):
        now = time.time()
        self.fps_times.append(now)
        self.fps_times = [t for t in self.fps_times if now - t < 1.0]
        self.fps = len(self.fps_times)

    def run_navigation(self):
        frame, annotated, detections = self.detector.detect()

        if frame is None:
            return

        self.update_fps()
        self.nav_count += 1

        # Print FPS every 30 frames
        if self.nav_count % 30 == 0:
            print(f"[FPS] {self.fps}", end='\r')

        # Show video window
        if SHOW_WINDOW and annotated is not None:
            cv2.putText(annotated, f'Mode: {self.mode}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f'FPS: {self.fps}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(annotated, 'Press Q to quit', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('VisionAssist', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                return

        detected_labels = [d['label'] for d in detections]
        self.update_stability(detected_labels)

        now = time.time()
        urgent = []
        normal = []

        for det in detections:
            label = det['label']
            if not self.should_announce(label):
                continue
            self.last_announced[label] = now
            if det['is_urgent']:
                urgent.insert(0, f"warning! {label} very close {det['direction']}")
            else:
                normal.append(f"{label} {det['urgency']} {det['direction']}")

        for msg in urgent[:2]:
            self.speaker.speak_now(msg)
            print(f"[Nav] {msg}")

        if normal and not urgent:
            message = ", ".join(normal[:3])
            self.speaker.speak(message)
            print(f"[Nav] {message}")

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
            frame, annotated, detections = self.detector.detect()
            self.scene.describe(detections)
            print("[System] Auto-returning to navigation in 8 seconds...")

        elif cmd == "where am i":
            self._go_navigation()
            self.mode = "WAITING"
            self.waiting_since = time.time()
            frame, annotated, detections = self.detector.detect()
            self.scene.where_am_i(detections)
            print("[System] Auto-returning to navigation in 8 seconds...")

        elif cmd == "help":
            self._go_navigation()
            self.mode = "WAITING"
            self.waiting_since = time.time()
            self.speaker.speak(
                "Available commands: "
                "hey vision read this. "
                "hey vision stop reading. "
                "hey vision describe. "
                "hey vision where am i. "
                "hey vision start."
            )
            print("[System] Auto-returning to navigation in 8 seconds...")

    def run(self):
        self.voice.start()
        self.speaker.speak("VisionAssist ready. Navigation mode active.")
        print("[System] Running. Press Ctrl+C to stop.")

        try:
            while self.running:
                if self.mode == "NAVIGATION":
                    self.run_navigation()

                elif self.mode == "READING":
                    frame = self.detector.capture()
                    if SHOW_WINDOW and frame is not None:
                        display = frame.copy()
                        cv2.putText(display, 'Mode: READING', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                        cv2.putText(display, 'Scanning for text...', (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1)
                        cv2.imshow('VisionAssist', display)
                        cv2.waitKey(1)
                    if self.mode == "READING" and frame is not None:
                        self.reader.read_frame(frame)
                        time.sleep(0.1)

                elif self.mode == "WAITING":
                    frame = self.detector.capture()
                    if SHOW_WINDOW and frame is not None:
                        display = frame.copy()
                        cv2.putText(display, 'Mode: WAITING', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 165, 0), 2)
                        cv2.imshow('VisionAssist', display)
                        cv2.waitKey(1)
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
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        print("[System] All systems offline.")


if __name__ == "__main__":
    app = VisionAssistApp()
    app.run()

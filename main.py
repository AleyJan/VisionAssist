import time
import threading
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
from speech import Speaker
from reading_mode import ReadingMode
from voice_recognition import VoiceRecognizer
from moondream_mode import SceneDescriber

YOLO_N_PATH  = '/home/raspberrypi/VisionAssist/models/yolo11n.onnx'
YOLO_S_PATH  = '/home/raspberrypi/VisionAssist/models/yolo11s.onnx'
CONF_N       = 0.80
CONF_S       = 0.65
IMGSZ        = 320
COOLDOWN     = 6
MIN_FRAMES   = 3
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

PRIORITY_CLASSES = {
    "person",
    "car", "motorcycle", "bus", "truck", "bicycle",
    "traffic light", "stop sign", "fire hydrant",
    "chair", "couch", "bed", "dining table", "toilet",
    "bottle", "cup", "laptop", "cell phone",
    "book", "backpack", "handbag", "umbrella",
    "dog", "cat",
}


class DetectionEngine:
    def __init__(self):
        print("[Camera] Starting...")
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)},
            controls={
                "Brightness": 0.15,
                "Contrast": 1.25,
                "Saturation": 1.1,
                "Sharpness": 1.0,
                "ExposureTime": 25000,
                "AnalogueGain": 1.8
            }
        ))
        self.picam2.start()
        print("[Camera] Ready!")
        print("[YOLO] Loading models...")
        self.model_n = YOLO(YOLO_N_PATH, task='detect')
        self.model_s = YOLO(YOLO_S_PATH, task='detect')
        self.frame_count = 0
        print("[YOLO] Both models ready!")

    def enhance_frame(self, frame):
        frame = cv2.convertScaleAbs(frame, alpha=1.25, beta=15)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

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
        self.frame_count += 1
        frame = self.picam2.capture_array()
        enhanced = self.enhance_frame(frame)

        results_n = self.model_n(enhanced, verbose=False, conf=CONF_N, imgsz=IMGSZ)
        results_s = None
        if self.frame_count % 5 == 0:
            results_s = self.model_s(enhanced, verbose=False, conf=CONF_S, imgsz=IMGSZ)

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
                    "label": label,
                    "direction": direction,
                    "urgency": urgency,
                    "is_urgent": is_urgent,
                    "conf": conf
                })

        seen = {}
        for det in raw:
            key = (det['label'], det['direction'])
            if key not in seen or det['conf'] > seen[key]['conf']:
                seen[key] = det

        return frame, list(seen.values())

    def capture(self):
        return self.picam2.capture_array()

    def stop(self):
        self.picam2.stop()
        print("[Camera] Stopped")


class VisionAssistApp:
    """
    Modes:
      NAVIGATION  — YOLO runs continuously
      READING     — EasyOCR runs in loop
      WAITING     — idle after describe/where, auto-returns to nav after 8s
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
        self.mode    = "NAVIGATION"
        self.running = True
        self.waiting_since = None  # tracks when WAITING mode started
        print("[System] All components ready!")

    def _go_navigation(self):
        """Switch cleanly to navigation mode."""
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

    def run_navigation(self):
        frame, detections = self.detector.detect()
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
            # Universal go-back-to-navigation command
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
            frame, detections = self.detector.detect()
            self.scene.describe(detections)
            print("[System] Auto-returning to navigation in 8 seconds...")

        elif cmd == "where am i":
            self._go_navigation()
            self.mode = "WAITING"
            self.waiting_since = time.time()
            frame, detections = self.detector.detect()
            self.scene.where_am_i(detections)
            print("[System] Auto-returning to navigation in 8 seconds...")

        elif cmd == "help":
            self._go_navigation()
            self.mode = "WAITING"
            self.waiting_since = time.time()
            self.speaker.speak(
                "Commands: "
                "hey pi read this for reading. "
                "hey pi stop reading to navigate. "
                "hey pi describe for scene. "
                "hey pi where am i for location. "
                "hey pi start to begin navigation."
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
                    frame, _ = self.detector.detect()
                    if self.mode == "READING":
                        self.reader.read_frame(frame)
                        time.sleep(3)

                elif self.mode == "WAITING":
                    # Keep camera alive while waiting
                    self.detector.capture()
                    time.sleep(0.2)

                    # Auto-return to navigation after 8 seconds
                    if self.waiting_since and time.time() - self.waiting_since > 8:
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

import sounddevice as sd
import vosk
import queue
import json
import threading
import time

VOSK_MODEL_PATH = "/home/raspberrypi/VisionAssist/models/vosk/vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
DEVICE      = 1
GRAMMAR     = '["hey", "vision", "read", "this", "stop", "reading", "describe", "where", "am", "i", "help", "start", "navigate", "[unk]"]'

# How long to wait for a command after hearing "hey vision"
WAKE_TIMEOUT = 4.0


class VoiceRecognizer:
    """
    Strict two-step voice recognition:

    Step 1 — Wake phrase detection
      Must hear "hey" AND "vision" in same utterance
      If only "hey" or only "vision" → ignore

    Step 2 — Command detection
      After wake phrase confirmed → listen for command
      Has WAKE_TIMEOUT seconds to receive command
      If no command received → return to listening

    This prevents false triggers from:
      - Background noise
      - Partial words
      - TV/music in background
    """
    def __init__(self, command_callback,
                 on_listening=None, on_stop_listening=None):
        print("[Voice] Loading Vosk model...")
        self.model      = vosk.Model(VOSK_MODEL_PATH)
        self.recognizer = vosk.KaldiRecognizer(
            self.model, SAMPLE_RATE, GRAMMAR)
        self.audio_queue        = queue.Queue()
        self.command_callback   = command_callback
        self.on_listening       = on_listening       # called when user starts speaking
        self.on_stop_listening  = on_stop_listening  # called when done listening
        self.running            = False
        self.wake_detected      = False
        self.wake_time          = 0
        print("[Voice] Ready!")

    def audio_callback(self, indata, frames, time, status):
        self.audio_queue.put(bytes(indata))

    def process_audio(self):
        while self.running:
            data = self.audio_queue.get()
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip().lower()

                if not text or "[unk]" in text:
                    continue

                print(f"[Voice] Heard: {text}")
                self._process_text(text)

    def _process_text(self, text):
        words = text.split()

        # ── Step 1: Check for wake phrase ──
        has_hey    = "hey" in words
        has_vision = "vision" in words

        if has_hey and has_vision:
            # Full wake phrase detected
            self.wake_detected = True
            self.wake_time     = time.time()
            print("[Voice] Wake phrase detected — listening for command...")

            # Notify app to pause audio
            if self.on_listening:
                self.on_listening()

            # Check if command is in same utterance
            self._match_command(text)
            return

        # ── Step 2: If wake was recent, accept command ──
        if self.wake_detected:
            elapsed = time.time() - self.wake_time
            if elapsed < WAKE_TIMEOUT:
                self._match_command(text)
            else:
                # Wake timed out
                self.wake_detected = False
                print("[Voice] Wake timed out — back to listening")
                if self.on_stop_listening:
                    self.on_stop_listening()

    def _match_command(self, text):
        matched = False

        if "stop" in text and "reading" in text:
            self.command_callback("navigate")
            matched = True
        elif "start" in text or "navigate" in text:
            self.command_callback("navigate")
            matched = True
        elif "read" in text and "reading" not in text \
                and "stop" not in text:
            self.command_callback("read")
            matched = True
        elif "describe" in text:
            self.command_callback("describe")
            matched = True
        elif "where" in text and "am" in text:
            self.command_callback("where am i")
            matched = True
        elif "help" in text and "describe" not in text \
                and "read" not in text:
            self.command_callback("help")
            matched = True

        if matched:
            self.wake_detected = False
            if self.on_stop_listening:
                self.on_stop_listening()

    def start(self):
        self.running = True
        threading.Thread(
            target=self.process_audio,
            daemon=True
        ).start()
        self.stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=8000,
            device=DEVICE,
            dtype="int16",
            channels=1,
            callback=self.audio_callback
        )
        self.stream.start()
        print("[Voice] Listening for commands...")

    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()
        print("[Voice] Stopped")

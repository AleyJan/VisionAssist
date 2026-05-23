import sounddevice as sd
import vosk
import queue
import json
import threading

VOSK_MODEL_PATH = "/home/raspberrypi/VisionAssist/models/vosk/vosk-model-small-en-us-0.15"
SAMPLE_RATE = 16000
DEVICE = 1
GRAMMAR = '["hey", "vision", "read", "this", "stop", "reading", "describe", "where", "am", "i", "help", "start", "navigate", "[unk]"]'


class VoiceRecognizer:
    """
    Offline voice command recognition using Vosk.
    Wake word: hey vision
    Commands:
      hey vision read this     → reading mode
      hey vision stop reading  → back to navigation
      hey vision start         → back to navigation
      hey vision describe      → describe scene once
      hey vision where am i    → location guess
      hey vision help          → list commands
    """
    def __init__(self, command_callback):
        print("[Voice] Loading Vosk model...")
        self.model = vosk.Model(VOSK_MODEL_PATH)
        self.recognizer = vosk.KaldiRecognizer(self.model, SAMPLE_RATE, GRAMMAR)
        self.audio_queue = queue.Queue()
        self.command_callback = command_callback
        self.running = False
        print("[Voice] Ready!")

    def audio_callback(self, indata, frames, time, status):
        self.audio_queue.put(bytes(indata))

    def process_audio(self):
        while self.running:
            data = self.audio_queue.get()
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip().lower()
                if text and "[unk]" not in text:
                    print(f"[Voice] Heard: {text}")
                    self._match_command(text)

    def _match_command(self, text):
        # Stop reading — check FIRST before anything else
        # "reading" must not trigger "read"
        if "stop" in text and "reading" in text:
            self.command_callback("navigate")

        # Navigate/start explicitly
        elif "start" in text or "navigate" in text:
            self.command_callback("navigate")

        # Read — "reading" alone must NOT trigger this
        # requires "read" but NOT "reading" and NOT "stop"
        elif "read" in text and "reading" not in text and "stop" not in text:
            self.command_callback("read")

        # Describe scene
        elif "describe" in text:
            self.command_callback("describe")

        # Where am I
        elif "where" in text and "am" in text:
            self.command_callback("where am i")

        # Help
        elif "help" in text:
            self.command_callback("help")

    def start(self):
        self.running = True
        threading.Thread(target=self.process_audio, daemon=True).start()
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

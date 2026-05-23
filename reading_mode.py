import easyocr
import threading
from speech import Speaker


class ReadingMode:
    """
    EasyOCR based text reading mode.
    Runs OCR in background thread so camera never freezes.
    Activated by: hey vision read this
    Deactivated by: hey vision stop reading
    """
    def __init__(self, speaker: Speaker):
        print("[Reading] Loading EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        self.speaker = speaker
        self.running = False
        self.is_scanning = False
        print("[Reading] EasyOCR ready!")

    def read_frame(self, frame):
        if not self.running:
            return
        # Don't start new scan if already scanning
        if self.is_scanning:
            return
        # Run OCR in background thread so camera doesn't freeze
        threading.Thread(
            target=self._scan, args=(frame,), daemon=True
        ).start()

    def _scan(self, frame):
        self.is_scanning = True
        print("[Reading] Scanning for text...")
        try:
            results = self.reader.readtext(frame)
            if not results:
                print("[Reading] No text found")
                self.is_scanning = False
                return
            lines = [text for (_, text, conf) in results if conf > 0.4]
            if lines:
                full_text = " ".join(lines)
                print(f"[Reading] Found: {full_text}")
                self.speaker.speak(full_text)
            else:
                print("[Reading] No clear text found")
        except Exception as e:
            print(f"[Reading] Error: {e}")
        finally:
            self.is_scanning = False

    def start(self):
        self.running = True
        self.is_scanning = False
        print("[Reading] Reading mode activated")
        self.speaker.speak("Reading mode. Point camera at text.")

    def stop(self):
        self.running = False
        self.is_scanning = False
        print("[Reading] Reading mode deactivated")

import easyocr
from speech import Speaker

class ReadingMode:
    """
    EasyOCR-based text reading mode.
    Activated by voice command: 'hey pi read this'
    Deactivated by: 'hey pi stop reading'
    """
    def __init__(self, speaker: Speaker):
        print("[Reading] Loading EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        self.speaker = speaker
        self.running = False
        print("[Reading] EasyOCR ready!")

    def read_frame(self, frame):
        if not self.running:
            return
        print("[Reading] Scanning for text...")
        results = self.reader.readtext(frame)
        if not results:
            print("[Reading] No text found")
            return

        # Only keep high confidence text (0.4 minimum)
        lines = [text for (_, text, conf) in results if conf > 0.4]
        if lines:
            full_text = " ".join(lines)
            print(f"[Reading] Found: {full_text}")
            self.speaker.speak(full_text)
        else:
            print("[Reading] No clear text found")

    def start(self):
        self.running = True
        print("[Reading] Reading mode activated")
        self.speaker.speak("Reading mode. Point camera at text.")

    def stop(self):
        self.running = False
        print("[Reading] Reading mode deactivated")

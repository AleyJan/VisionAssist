# VisionAssist 🦯

An open-source, offline, real-time assistive glasses system for visually 
impaired people. Built on Raspberry Pi 5 using computer vision and AI to 
provide audio feedback about the surrounding environment.

---

## Demo

Point camera at environment → hear audio descriptions automatically.

**Voice Commands:**
- `hey pi read this` → reads text aloud (signs, books, labels)
- `hey pi stop reading` → returns to navigation
- `hey pi describe` → describes current scene
- `hey pi where am i` → guesses current location
- `hey pi start` → resume navigation from any mode
- `hey pi help` → lists all commands

---

## Hardware Required

| Component | Purpose | Cost |
|---|---|---|
| Raspberry Pi 5 (4GB) | Main processor | ~$60 |
| Pi Camera Module 2 | Visual input | ~$25 |
| Bluetooth Earphones | Audio output | varies |
| MicroSD Card 32GB+ | Storage | ~$10 |
| Power Bank | Portable power | ~$15 |
| **Total** | | **~$110** |

---

## Features

### Navigation Mode (Default)
- Dual YOLO model pipeline running continuously
- YOLO11n at ~15 FPS for real-time obstacle detection
- YOLO11s every 5 frames for small object detection
- Direction detection — left, right, in front
- Urgency levels — nearby, ahead, warning very close
- 20 priority classes — people, vehicles, furniture, animals
- 3-frame stability check — no flickering announcements
- 6 second cooldown — no repeated announcements
- CLAHE image preprocessing — works in low light

### Reading Mode
- Activated by voice command
- EasyOCR reads signs, books, labels, printed text
- Navigation pauses automatically
- Returns to navigation on voice command

### Scene Description Mode
- Describes what is visible in natural language
- Uses YOLO detections to build scene summary
- Auto-returns to navigation after 8 seconds

### Voice Recognition
- Fully offline using Vosk
- No internet required
- Recognizes specific commands only

### Text to Speech
- Piper TTS — human quality voice
- Streams to Bluetooth earphones
- Auto-reconnects if disconnected

---

## System Architecture

### Modes

| Mode | Trigger | What runs | Returns to nav |
|---|---|---|---|
| NAVIGATION | startup / hey pi start | YOLO detection loop | never (default) |
| READING | hey pi read this | EasyOCR loop | hey pi stop reading |
| WAITING | hey pi describe | idle | auto after 8 seconds |

### File Structure

VisionAssist/
main.py              — main app, detection engine, mode manager
speech.py            — shared Piper TTS module
reading_mode.py      — EasyOCR text reading
voice_recognition.py — Vosk offline voice commands
moondream_mode.py    — scene description using YOLO labels
evaluate_models.py   — model performance evaluation script
models/
yolo11n.onnx       — YOLO11n nano (primary, 15 FPS)
yolo11s.onnx       — YOLO11s small (secondary, every 5 frames)
vosk/              — offline speech recognition model

---

## Installation

### Step 1 — Flash Raspberry Pi OS
Use Raspberry Pi Imager. Select Raspberry Pi OS Bookworm 64-bit.

### Step 2 — Update system
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git espeak-ng espeak-ng-data
```

### Step 3 — Create virtual environment
```bash
python3 -m venv ~/yolo-env --system-site-packages
source ~/yolo-env/bin/activate
```

### Step 4 — Install PyTorch CPU only
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 5 — Install dependencies
```bash
pip install ultralytics --no-deps
pip install opencv-python numpy pillow pyyaml tqdm scipy
pip install matplotlib psutil requests ultralytics-thop
pip install onnxruntime onnx onnxslim
pip install picamera2 --no-deps
pip install easyocr vosk sounddevice
pip install ncnn
```

### Step 6 — Install Piper TTS
```bash
cd ~
wget https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
tar -xzf piper_linux_aarch64.tar.gz
mkdir -p ~/piper-voices
cd ~/piper-voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

### Step 7 — Download YOLO models
```bash
source ~/yolo-env/bin/activate
cd ~/VisionAssist
python3 -c "
from ultralytics import YOLO
YOLO('yolo11n.pt').export(format='onnx', imgsz=320)
YOLO('yolo11s.pt').export(format='onnx', imgsz=320)
"
mv yolo11n.onnx models/
mv yolo11s.onnx models/
```

### Step 8 — Download Vosk model
```bash
mkdir -p ~/VisionAssist/models/vosk
cd ~/VisionAssist/models/vosk
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

### Step 9 — Connect Bluetooth earphones
```bash
bluetoothctl
connect YOUR_DEVICE_MAC
exit
pactl set-card-profile bluez_card.YOUR_DEVICE_MAC a2dp-sink
```

### Step 10 — Run
```bash
source ~/yolo-env/bin/activate
cd ~/VisionAssist
python3 main.py
```

---

## Performance

| Model | Input Size | FPS | Use |
|---|---|---|---|
| YOLO11n ONNX | 320x320 | ~15-16 | Every frame |
| YOLO11s ONNX | 320x320 | ~9-10 | Every 5 frames |
| Dual pipeline | mixed | ~13-15 | Combined |

---

## Detection Classes

VisionAssist only announces objects relevant to navigation:

**People & Vehicles:** person, car, motorcycle, bus, truck, bicycle

**Traffic:** traffic light, stop sign, fire hydrant

**Indoor obstacles:** chair, couch, bed, dining table, toilet

**Common objects:** bottle, cup, laptop, cell phone, book, backpack, handbag, umbrella

**Animals:** dog, cat

---

## Roadmap

- [x] Real-time YOLO11n object detection
- [x] Dual model pipeline (nano + small)
- [x] ONNX optimization for Pi 5
- [x] Direction detection (left/center/right)
- [x] Urgency system (nearby/ahead/warning)
- [x] Class filtering (priority classes only)
- [x] CLAHE image preprocessing
- [x] Frame stability tracking
- [x] Piper TTS human voice
- [x] Vosk offline voice commands
- [x] EasyOCR text reading
- [x] Scene description
- [x] Mode manager (navigation/reading/waiting)
- [ ] ByteTrack object tracking
- [ ] Ultrasonic sensor for accurate distance
- [ ] Auto start on boot
- [ ] Moondream2 scene description (when ARM support improves)
- [ ] Fine-tune YOLO on custom environment
- [ ] Glasses hardware assembly

---

## Why VisionAssist?

| Feature | VisionAssist | Commercial alternatives |
|---|---|---|
| Cost | ~$110 | $2,000 - $5,000 |
| Internet required | No | Often yes |
| Open source | Yes | No |
| Custom voice commands | Yes | Limited |
| Works in darkness | Partial (camera) | Limited |
| Offline speech | Yes | Rarely |

---

## Developer

**Ali Jan** — Built for visually impaired communities,
especially in developing countries where expensive 
assistive technology is out of reach.

> *VisionAssist — See the world through sound.*

---

## License

MIT License — free to use, modify, and distribute.

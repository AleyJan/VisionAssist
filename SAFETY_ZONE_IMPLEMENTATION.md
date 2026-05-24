# VisionAssist — Safety Zone Implementation Plan
**Date: 24 May 2026**
**Author: Ali Jan**
**Project: VisionAssist — AI Powered Glasses for Visually Impaired**

---

## Overview

Replace the current bounding-box-based detection system with a
user-defined Safety Zone approach. The safety zone is a fixed
pixel rectangle in the camera frame. Anything entering this zone
triggers audio feedback — whether YOLO identifies it or not.

---

## The Core Idea

```
Camera Frame (640x480)
┌─────────────────────────────────────┐
│                                     │
│   GREEN boxes = outside zone        │
│   (detected but not threatening)    │
│  ┌───────────────────────────────┐  │
│  │                               │  │
│  │      SAFETY ZONE              │  │
│  │                               │  │
│  │   RED boxes = inside zone     │  │
│  │   (triggers audio alert)      │  │
│  │                               │  │
│  └───────────────────────────────┘  │
│                                     │
└─────────────────────────────────────┘
              👤 User
```

**Two announcement cases:**
- YOLO identifies object inside zone → say its name + direction
- Motion inside zone but YOLO doesn't identify → say "unknown object detected"

---

## Implementation Phases

---

### Phase A — Safety Zone Visual (TODAY)
**Goal:** Draw the safety zone on video window and test it visually

**Steps:**

**A1 — Define safety zone coordinates**
```
ZONE_LEFT   = 0.20  → 128px from left
ZONE_RIGHT  = 0.80  → 512px from right
ZONE_TOP    = 0.10  → 48px from top
ZONE_BOTTOM = 1.00  → 480px (full bottom)
```
Zone is tunable — just change these 4 numbers.

**A2 — Draw safety zone rectangle on video window**
- Yellow rectangle border showing the safety zone boundary
- Label "SAFETY ZONE" at top of rectangle
- Zone always visible on screen

**A3 — Color code bounding boxes**
- GREEN box = object detected OUTSIDE safety zone (not threatening)
- RED box = object detected INSIDE safety zone (threatening)
- Object label shown above each box

**A4 — Test visually**
- Walk around room
- Verify green/red switching works correctly
- Tune zone size if needed

**Deliverable:** Video window showing zone + colored boxes
**Estimated time:** 1 hour

---

### Phase B — Motion Detection Inside Zone
**Goal:** Detect when ANYTHING enters zone, even if YOLO misses it

**Steps:**

**B1 — Background learning**
- First 30 frames → learn what empty zone looks like
- Store as background reference frame

**B2 — Background subtraction**
- Every frame → compare zone pixels to background
- If pixel difference > threshold → motion detected
- Use OpenCV MOG2 background subtractor (built-in, no install needed)

**B3 — Motion sensitivity tuning**
- MIN_MOTION_AREA = 500 pixels (ignore tiny noise)
- Larger area = less sensitive (fewer false alarms)
- Smaller area = more sensitive (catches subtle movement)

**B4 — Motion direction**
- If motion centroid is in left third of zone → "from your left"
- If motion centroid is in center third → "directly ahead"
- If motion centroid is in right third → "from your right"

**Deliverable:** Motion detection working inside zone
**Estimated time:** 1 hour

---

### Phase C — Smart Audio Announcements
**Goal:** Combine YOLO detection + motion detection for smart audio

**Steps:**

**C1 — Check if YOLO detection is inside zone**
```
For each YOLO detection:
    Get center point (cx, cy) of bounding box
    If ZONE_LEFT < cx < ZONE_RIGHT
    AND ZONE_TOP  < cy < ZONE_BOTTOM:
        → Object is INSIDE zone
        → RED bounding box
        → Trigger audio
    Else:
        → Object is OUTSIDE zone
        → GREEN bounding box
        → No audio
```

**C2 — Two announcement types**
```
Case A — YOLO identifies object in zone:
    "Person directly ahead of you"
    "Chair coming from your left"
    "Car on your right"

Case B — Motion detected but YOLO didn't identify:
    "Unknown object directly ahead"
    "Object detected on your left"
    "Something approaching from your right"
```

**C3 — Announcement cooldown per zone section**
```
Left section cooldown:   4 seconds
Center section cooldown: 4 seconds
Right section cooldown:  4 seconds
Each section independent — object moving left to right
announces in each section separately
```

**C4 — Priority system**
```
PRIORITY 1 (highest): Person inside zone → immediate announce
PRIORITY 2:           Vehicle inside zone → immediate announce
PRIORITY 3:           Other object inside zone → announce
PRIORITY 4 (lowest):  Unknown motion inside zone → announce
```

**Deliverable:** Smart audio working with safety zone
**Estimated time:** 2 hours

---

### Phase D — Trajectory Tracking
**Goal:** Detect if object is approaching or moving away

**Steps:**

**D1 — Track object size over time**
```python
# For each detected object keep last 5 frame sizes
size_history[label] = [s1, s2, s3, s4, s5]

# Compare recent average to older average
if recent > older * 1.15:
    trajectory = "APPROACHING"
elif recent < older * 0.85:
    trajectory = "MOVING AWAY"
else:
    trajectory = "STATIONARY"
```

**D2 — Smarter announcements**
```
"Person approaching from your left"    ← getting bigger
"Car moving away ahead of you"         ← getting smaller
"Chair directly ahead of you"          ← stationary
```

**D3 — Urgency based on trajectory**
```
APPROACHING + inside zone → highest urgency
STATIONARY  + inside zone → medium urgency
MOVING AWAY + inside zone → low urgency (may not announce)
```

**Deliverable:** Trajectory-aware announcements
**Estimated time:** 2 hours

---

### Phase E — Ultrasonic Integration (needs hardware)
**Goal:** Add real distance measurement to safety zone

**Steps:**

**E1 — Connect HC-SR04P to Pi 5 GPIO**
```
HC-SR04P → Pi 5
VCC      → Pin 2  (5V... wait, use 3.3V version)
GND      → Pin 6  (GND)
TRIG     → Pin 11 (GPIO17)
ECHO     → Pin 13 (GPIO27)
```

**E2 — Distance zones**
```
< 50cm  → DANGER  → urgent beep + voice
50-150cm → WARNING → warning voice
150-300cm → CAUTION → calm voice
> 300cm  → CLEAR  → silence
```

**E3 — Combine ultrasonic + camera zone**
```
Object in camera zone AND distance < 100cm → MAXIMUM ALERT
Object in camera zone AND distance > 200cm → CAUTION only
No object in camera zone AND distance < 50cm → UNKNOWN ALERT
```

**E4 — Distance in announcement**
```
"Person ahead, 45 centimeters"
"Warning! Chair very close, 30 centimeters"
"Unknown object, 60 centimeters ahead"
```

**Deliverable:** Real distance in all announcements
**Estimated time:** 3 hours (includes wiring)
**Hardware needed:** HC-SR04P sensor + jumper wires

---

### Phase F — Auto Boot on Startup
**Goal:** VisionAssist starts automatically when Pi powers on

**Steps:**

**F1 — Create systemd service**
```bash
sudo nano /etc/systemd/system/visionassist.service
```

**F2 — Service file content**
```
[Unit]
Description=VisionAssist Assistive Glasses
After=network.target bluetooth.target sound.target
Wants=bluetooth.target

[Service]
Type=simple
User=raspberrypi
WorkingDirectory=/home/raspberrypi/VisionAssist
Environment=DISPLAY=:0
ExecStartPre=/bin/sleep 15
ExecStart=/home/raspberrypi/yolo-env/bin/python main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**F3 — Enable service**
```bash
sudo systemctl enable visionassist
sudo systemctl start visionassist
```

**Deliverable:** VisionAssist starts on every boot automatically
**Estimated time:** 30 minutes

---

## Summary Table

| Phase | Feature | Time | Hardware |
|---|---|---|---|
| A | Safety zone visual + colored boxes | 1 hour | None |
| B | Motion detection inside zone | 1 hour | None |
| C | Smart audio announcements | 2 hours | None |
| D | Trajectory tracking | 2 hours | None |
| E | Ultrasonic real distance | 3 hours | HC-SR04P |
| F | Auto boot on startup | 30 min | None |
| **Total** | | **~10 hours** | **HC-SR04P** |

---

## Config Values (tunable without touching logic)

```python
# Safety Zone (fraction of frame 0.0 to 1.0)
ZONE_LEFT   = 0.20
ZONE_RIGHT  = 0.80
ZONE_TOP    = 0.10
ZONE_BOTTOM = 1.00

# Motion detection
MIN_MOTION_AREA = 500    # pixels — ignore smaller motion

# Cooldown
ZONE_COOLDOWN = 4        # seconds between same-direction announcements

# Trajectory sensitivity
APPROACH_THRESHOLD = 1.15   # 15% size increase = approaching
RETREAT_THRESHOLD  = 0.85   # 15% size decrease = moving away

# Ultrasonic zones (cm)
DANGER_DISTANCE  = 50
WARNING_DISTANCE = 150
CAUTION_DISTANCE = 300
```

---

## Current Status

```
Phase A — Safety zone visual      [ ] Not started
Phase B — Motion detection        [ ] Not started
Phase C — Smart announcements     [ ] Not started
Phase D — Trajectory tracking     [ ] Not started
Phase E — Ultrasonic integration  [ ] Waiting for hardware
Phase F — Auto boot               [ ] Not started
```

---

## Files to Modify

```
main.py              ← primary changes (safety zone logic)
moondream_mode.py    ← update describe to use zone detections
VISION_ASSIST.md     ← update product document
README.md            ← update with safety zone feature
```

---

*VisionAssist — See the world through sound.*

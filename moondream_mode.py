from speech import Speaker

class SceneDescriber:
    """
    Smart scene description using YOLO detection labels.
    Replaces moondream which requires GPU and doesn't work on Pi 5.
    Takes detected objects and builds a natural language description.
    """
    def __init__(self, speaker: Speaker):
        self.speaker = speaker
        print("[Scene] Scene describer ready!")

    def describe(self, detections):
        """
        Takes list of detections from DetectionEngine
        and produces a natural scene description.
        """
        if not detections:
            self.speaker.speak("Nothing detected in the scene.")
            return

        # Group by direction
        left = [d['label'] for d in detections if d['direction'] == 'on your left']
        front = [d['label'] for d in detections if d['direction'] == 'in front of you']
        right = [d['label'] for d in detections if d['direction'] == 'on your right']

        parts = []
        if front:
            items = " and ".join(front)
            parts.append(f"{items} directly ahead")
        if left:
            items = " and ".join(left)
            parts.append(f"{items} on your left")
        if right:
            items = " and ".join(right)
            parts.append(f"{items} on your right")

        if parts:
            message = "I can see " + ", ".join(parts) + "."
            print(f"[Scene] {message}")
            self.speaker.speak(message)

    def where_am_i(self, detections):
        """
        Guesses the environment based on detected objects.
        """
        labels = [d['label'] for d in detections]

        if 'bed' in labels or 'pillow' in labels:
            place = "You appear to be in a bedroom."
        elif 'toilet' in labels or 'sink' in labels:
            place = "You appear to be in a bathroom."
        elif 'dining table' in labels or 'cup' in labels or 'bottle' in labels:
            place = "You appear to be in a kitchen or dining area."
        elif 'laptop' in labels or 'chair' in labels or 'keyboard' in labels:
            place = "You appear to be in an office or study area."
        elif 'car' in labels or 'bus' in labels or 'traffic light' in labels:
            place = "You appear to be outdoors near a road."
        elif 'person' in labels and len(labels) > 3:
            place = "You appear to be in a crowded area."
        elif not labels:
            place = "I cannot determine your location. No objects detected."
        else:
            place = f"I can see {', '.join(set(labels))}. Location unclear."

        print(f"[Scene] {place}")
        self.speaker.speak(place)

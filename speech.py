import subprocess
import threading
import queue
import time

PIPER_MODEL   = '/home/raspberrypi/piper-voices/en_US-lessac-medium.onnx'
AIRPODS_DEVICE = 'bluez_output.41_42_BE_BD_9C_4B.1'
FALLBACK_DEVICE = None  # will use system default if AirPods disconnect

class Speaker:
    """
    Shared TTS module. Uses Piper + PulseAudio.
    Auto-detects if AirPods are connected.
    Falls back to system audio if disconnected.
    Auto-reconnects AirPods when they come back.
    """
    def __init__(self):
        self._queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._keepalive, daemon=True).start()
        print("[Speech] Piper TTS ready!")

    def _get_audio_device(self):
        """Check if AirPods are connected, return device name or None."""
        try:
            result = subprocess.run(
                ['pactl', 'list', 'sinks', 'short'],
                capture_output=True, text=True
            )
            for line in result.stdout.splitlines():
                if 'bluez' in line:
                    return line.split('\t')[1]
        except:
            pass
        return None

    def _reconnect_airpods(self):
        """Try to reconnect AirPods via bluetoothctl."""
        try:
            subprocess.run(
                ['bluetoothctl', 'connect', '41:42:BE:BD:9C:4B'],
                capture_output=True, timeout=8
            )
        except:
            pass

    def _keepalive(self):
        """
        Runs every 30 seconds.
        Checks if AirPods are still connected.
        Tries to reconnect if they dropped.
        """
        while True:
            time.sleep(30)
            device = self._get_audio_device()
            if not device:
                print("[Speech] AirPods disconnected — attempting reconnect...")
                self._reconnect_airpods()
                time.sleep(5)
                device = self._get_audio_device()
                if device:
                    print("[Speech] AirPods reconnected!")
                else:
                    print("[Speech] Reconnect failed — will retry in 30s")

    def speak(self, text):
        """Speak text — clears queue first so only latest message plays."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except:
                pass
        self._queue.put(text)

    def speak_now(self, text):
        """Speak text without clearing queue — for urgent warnings."""
        self._queue.put(text)

    def _worker(self):
        while True:
            text = self._queue.get()
            if text is None:
                break
            self._say(text)
            self._queue.task_done()

    def _say(self, text):
        """Actually speak the text using Piper + paplay."""
        device = self._get_audio_device()

        try:
            piper = subprocess.Popen(
                ['/home/raspberrypi/piper/piper',
                 '--model', PIPER_MODEL,
                 '--output-raw'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )

            paplay_cmd = ['paplay', '--raw', '--rate=22050',
                          '--format=s16le', '--channels=1']

            if device:
                paplay_cmd.append(f'--device={device}')

            paplay = subprocess.Popen(
                paplay_cmd,
                stdin=piper.stdout,
                stderr=subprocess.DEVNULL
            )

            piper.stdin.write(text.encode())
            piper.stdin.close()
            paplay.wait()

        except Exception as e:
            print(f"[Speech] Error: {e}")

    def stop(self):
        self._queue.put(None)

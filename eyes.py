"""
eyes.py - lets the character watch your screen.

A background loop grabs the screen about once a second, skips frames where nothing
changed, and asks a small vision model (through Ollama) for a one-line description.
brain.py reads the latest description whenever it talks, and can comment on its own
when something new happens.

Setup (once):  ollama pull qwen2.5vl:3b        (or "moondream" for faster, dumber eyes)
macOS will ask to let Terminal record the screen: System Settings > Privacy & Security >
Screen & System Audio Recording > turn on Terminal, then quit and reopen Terminal.
"""

import base64
import io
import json
import threading
import time
import urllib.request

# ---------------------------------------------------------------- settings
VISION_MODEL = "qwen2.5vl:3b"   # "moondream" = ~2x faster but much less smart
OLLAMA_URL = "http://localhost:11434/api/chat"
MONITOR = 1                     # 1 = main display
LOOK_EVERY = 1.0                # seconds between screen checks
CHANGE_THRESHOLD = 6.0          # how different a frame must be to bother the vision model (0-255)
FRAME_WIDTH = 512               # smaller = faster vision, less detail
CAPTION_PROMPTS = {
    "game": ("This is a screenshot of someone playing a video game. In ONE sentence (max 25 words), "
             "say what is happening right now. Name the game if obvious. Focus on the player's situation "
             "and anything going badly: dying, low health, losing, missing, getting lost, failing."),
    "video": ("This is a screenshot of someone's screen while they watch a video (e.g. YouTube). "
              "In ONE sentence (max 25 words), say what the video is showing right now. Include the "
              "video title or on-screen captions/subtitles if readable, and who or what is on screen."),
}


class Eyes:
    def __init__(self, on_caption=None, mode="game"):
        import mss                      # imported here so other modes never need it
        from PIL import Image
        self._mss, self._Image = mss, Image
        self.on_caption = on_caption    # called with (caption, seconds_taken) on every new caption
        self.prompt = CAPTION_PROMPTS.get(mode, CAPTION_PROMPTS["game"])
        self.caption = ""
        self.caption_time = 0.0
        self.last_thumb = None
        self.paused = False
        self.lock = threading.Lock()

    # ---- capture
    def grab(self):
        with self._mss.mss() as sct:
            mon = sct.monitors[min(MONITOR, len(sct.monitors) - 1)]
            shot = sct.grab(mon)
        img = self._Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        h = int(img.height * FRAME_WIDTH / img.width)
        return img.resize((FRAME_WIDTH, h))

    def changed(self, img):
        """Cheap check: compare a tiny grayscale thumbnail with the last one."""
        thumb = list(img.convert("L").resize((64, 36)).getdata())
        if self.last_thumb is None:
            self.last_thumb = thumb
            return True
        diff = sum(abs(a - b) for a, b in zip(thumb, self.last_thumb)) / len(thumb)
        if diff >= CHANGE_THRESHOLD:
            self.last_thumb = thumb
            return True
        return False

    # ---- vision model
    def describe(self, img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        data = json.dumps({
            "model": VISION_MODEL, "stream": False, "keep_alive": "30m",
            "messages": [{"role": "user", "content": self.prompt,
                          "images": [base64.b64encode(buf.getvalue()).decode()]}],
            "options": {"num_predict": 60, "temperature": 0.2},
        }).encode()
        req = urllib.request.Request(OLLAMA_URL, data, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["message"]["content"].strip().replace("\n", " ")

    def warm_up(self):
        """Load the vision model and check screen access, so problems show up at startup."""
        img = self.grab()
        if img.convert("L").getextrema() == (0, 0):
            raise RuntimeError("screen capture is black - give Terminal Screen Recording permission, "
                               "then quit and reopen Terminal")
        t0 = time.time()
        self.describe(img)
        return time.time() - t0

    # ---- background loop
    def latest(self, max_age=20):
        """Most recent description, or '' if it's stale."""
        with self.lock:
            return self.caption if time.time() - self.caption_time < max_age else ""

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            t0 = time.time()
            try:
                if not self.paused:
                    img = self.grab()
                    if self.changed(img):
                        caption = self.describe(img)
                        secs = time.time() - t0
                        with self.lock:
                            self.caption, self.caption_time = caption, time.time()
                        if self.on_caption:
                            self.on_caption(caption, secs)
            except Exception as e:
                print(f"\n(eyes: {e})", flush=True)
                time.sleep(5)
            time.sleep(max(0.0, LOOK_EVERY - (time.time() - t0)))

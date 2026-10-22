"""
ears.py - speech-to-text for Nova. Listens to the mic, waits for you to talk,
records until you pause, then transcribes on-device with Whisper (Apple MLX).

Used by brain.py when you run:  python3 brain.py --voice
Setup (once):  python3 -m pip install mlx-whisper sounddevice numpy
First run downloads the Whisper model (~150 MB) and macOS asks for mic access.
"""

import collections
import queue
import time

import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------- settings
STT_MODEL = "mlx-community/whisper-base.en-mlx"   # fast. More accurate: "mlx-community/whisper-turbo"
RATE = 16000                 # Whisper wants 16 kHz mono
FRAME_SEC = 0.03             # analyse the mic in 30 ms slices
START_FRAMES = 3             # ~90 ms of loudness = you started talking
END_SILENCE_SEC = 0.8        # this much quiet = you finished
MAX_UTTERANCE_SEC = 20
MIN_UTTERANCE_SEC = 0.4      # ignore coughs and clicks
PREROLL_SEC = 0.3            # keep a bit of audio from just before you started
SENSITIVITY = 3.0            # how far above room noise counts as speech; raise if it triggers on noise

# Whisper sometimes "hears" these in silence or noise
JUNK = {"", "you", "thank you.", "thanks for watching!", "bye.", ".", "thank you for watching."}


def looks_hallucinated(text):
    """Whisper sometimes fills noise with one word on repeat ("Okay. Okay. Okay..."). Drop those."""
    words = [w.strip(".,!?").lower() for w in text.split()]
    words = [w for w in words if w]
    return len(words) >= 6 and len(set(words)) / len(words) < 0.35


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


class Ears:
    def __init__(self):
        import mlx_whisper                          # imported here so typing mode never needs it
        self._whisper = mlx_whisper
        self.frame = int(RATE * FRAME_SEC)
        self.q = queue.Queue()
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                                     blocksize=self.frame, callback=self._on_audio)
        self.stream.start()
        self.threshold = 0.01

    def _on_audio(self, data, frames, t, status):
        self.q.put(data[:, 0].copy())

    def _frames(self):
        while True:
            yield self.q.get()

    def flush(self):
        """Throw away anything heard so far (e.g. Nova's own voice)."""
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def calibrate(self, seconds=1.0):
        """Measure room noise so we know what 'talking' sounds like."""
        self.flush()
        levels = [rms(f) for _, f in zip(range(int(seconds / FRAME_SEC)), self._frames())]
        noise = float(np.median(levels)) if levels else 0.0
        self.threshold = max(noise * SENSITIVITY, 0.006)
        return noise

    def warm_up(self):
        """Load Whisper now so the first real transcription isn't slow."""
        self.transcribe(np.zeros(RATE, dtype=np.float32))

    def record(self, frames=None):
        """Block until someone speaks, then return the utterance as a float32 array."""
        frames = frames if frames is not None else self._frames()
        preroll = collections.deque(maxlen=int(PREROLL_SEC / FRAME_SEC))
        loud_run, speech, silent = 0, [], 0
        end_frames = int(END_SILENCE_SEC / FRAME_SEC)
        max_frames = int(MAX_UTTERANCE_SEC / FRAME_SEC)
        for f in frames:
            loud = rms(f) > self.threshold
            if not speech:                           # waiting for you to start
                preroll.append(f)
                loud_run = loud_run + 1 if loud else 0
                if loud_run >= START_FRAMES:
                    speech = list(preroll)
                continue
            speech.append(f)                         # recording
            silent = 0 if loud else silent + 1
            if silent >= end_frames or len(speech) >= max_frames:
                break
        return np.concatenate(speech) if speech else np.zeros(0, dtype=np.float32)

    def transcribe(self, audio):
        out = self._whisper.transcribe(audio, path_or_hf_repo=STT_MODEL,
                                       language="en", verbose=None)
        return out.get("text", "").strip()

    def hear(self):
        """Wait for one complete thing you say and return it as text (skips noise)."""
        while True:
            audio = self.record()
            if len(audio) < RATE * MIN_UTTERANCE_SEC:
                continue
            t0 = time.time()
            text = self.transcribe(audio)
            if text.lower() in JUNK or looks_hallucinated(text):
                continue
            return text, time.time() - t0

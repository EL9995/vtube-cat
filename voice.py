"""
voice.py - text-to-speech engines for Nova. Each one turns a sentence into a .wav file;
brain.py plays it and drives the mouth from its loudness.

  "kokoro" - local neural voice, sounds close to human. 54 voices. Needs Python 3.12
             via uv (the launchers handle it). Downloads ~350 MB of model files on first use.
  "mac"    - the built-in macOS `say` voice. Instant, zero setup, sounds like a computer.

Kokoro voices worth trying (American): af_heart (best), af_bella, af_nicole, af_sarah,
am_michael, am_fenrir, am_puck.   British: bf_emma, bm_george, bm_fable.
Full list: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

Optional RVC voice conversion: after Kokoro speaks a sentence, an RVC model re-skins it
into a character's voice (see brain.py RVC_* settings). Runs Applio's engine in a
background worker (rvc_worker.py) with its own Python 3.12 environment, set up on first use.
"""

import json
import os
import shutil
import subprocess
import tempfile
import threading
import urllib.request
import wave
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
KOKORO_DIR = HERE / "kokoro_models"
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"
KOKORO_FILES = ["kokoro-v1.0.onnx", "voices-v1.0.bin"]


def _download(name):
    dest = KOKORO_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    KOKORO_DIR.mkdir(exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"downloading {name} (one time)...", flush=True)
    with urllib.request.urlopen(KOKORO_URL + name, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"  {done * 100 // total}%", end="\r", flush=True)
    tmp.rename(dest)
    print(f"  {name} done.     ", flush=True)
    return dest


# ---------------------------------------------------------------- RVC voice conversion
APPLIO_COMMIT = "67a38f6abab3e90a55efa534ea3185a222c102f0"   # pinned, known-good Applio version
APPLIO_DIR = HERE / "applio"
RVC_REQUIREMENTS = HERE / "rvc_requirements.txt"


def _find_uv():
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if not Path(uv).exists():
        raise RuntimeError("RVC needs uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
    return uv


def _ensure_applio():
    if (APPLIO_DIR / "rvc" / "infer" / "infer.py").exists():
        return
    print("downloading Applio voice engine (one time)...", flush=True)
    url = f"https://github.com/IAHispano/Applio/archive/{APPLIO_COMMIT}.zip"
    zpath = HERE / "applio_download.zip"
    with urllib.request.urlopen(url, timeout=120) as r, open(zpath, "wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(HERE)
    (HERE / f"Applio-{APPLIO_COMMIT}").rename(APPLIO_DIR)
    zpath.unlink()


class RVC:
    """Talks to rvc_worker.py, which keeps the voice model loaded between sentences."""

    def __init__(self, model_dir, pitch=0, index_rate=0.4, protect=0.33):
        folder = (HERE / model_dir) if not Path(model_dir).is_absolute() else Path(model_dir)
        pths = sorted(folder.glob("*.pth"))
        if not pths:
            raise FileNotFoundError(f"no .pth voice file in {folder}")
        idx = sorted(folder.glob("*.index"))
        self.job = {"model": str(pths[0]), "index": str(idx[0]) if idx else "",
                    "pitch": pitch, "index_rate": index_rate if idx else 0, "protect": protect}
        _ensure_applio()
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        cmd = [_find_uv(), "run", "--no-project", "--python", "3.12",
               "--with-requirements", str(RVC_REQUIREMENTS),
               "python", str(HERE / "rvc_worker.py"), str(APPLIO_DIR)]
        print(f"starting RVC voice '{folder.name}' (first time installs ~1-2 GB, be patient)...",
              flush=True)
        self.proc = subprocess.Popen(cmd, cwd=HERE, env=env, text=True, bufsize=1,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        self.ready = threading.Event()
        self.log = open(HERE / "rvc_worker.log", "w")
        threading.Thread(target=self._pipe_stderr, daemon=True).start()
        hello = json.loads(self.proc.stdout.readline() or '{"ready": false, "error": "worker exited"}')
        self.ready.set()
        if not hello.get("ready"):
            raise RuntimeError(f"RVC worker failed: {hello.get('error')} (see rvc_worker.log)")
        print(f"RVC ready on {hello.get('device')}.", flush=True)
        self.lock = threading.Lock()
        self.last_secs = 0.0

    def _pipe_stderr(self):
        """Show setup progress in the terminal; after that, send the engine's chatter to a log."""
        for line in self.proc.stderr:
            self.log.write(line)
            self.log.flush()
            if not self.ready.is_set() and any(k in line for k in
                                               ("ownload", "%", "nstall", "repar", "Resolved")):
                print("  " + line.rstrip()[:110], flush=True)

    def convert(self, src, dst):
        with self.lock:
            self.proc.stdin.write(json.dumps({**self.job, "in": str(src), "out": str(dst)}) + "\n")
            self.proc.stdin.flush()
            res = json.loads(self.proc.stdout.readline() or '{"ok": false, "error": "worker died"}')
        if not res.get("ok"):
            raise RuntimeError(res.get("error"))
        self.last_secs = res.get("secs", 0.0)


class Voice:
    def __init__(self, engine="kokoro", kokoro_voice="af_heart", speed=1.0, mac_voice=None,
                 rvc_model=None, rvc_pitch=0, rvc_index_rate=0.4):
        self.rvc = None
        if rvc_model:
            try:
                self.rvc = RVC(rvc_model, rvc_pitch, rvc_index_rate)
            except Exception as e:                   # don't let a bad voice model kill Nova
                print(f"(RVC off: {e})", flush=True)
        self.engine = engine
        self.kokoro_voice = kokoro_voice
        self.speed = speed
        self.mac_voice = mac_voice
        self.kokoro = None
        if engine == "kokoro":
            from kokoro_onnx import Kokoro            # only needed for this engine
            model, voices = (_download(f) for f in KOKORO_FILES)
            print(f"loading Kokoro voice '{kokoro_voice}'...", flush=True)
            self.kokoro = Kokoro(str(model), str(voices))
            if kokoro_voice not in self.kokoro.get_voices():
                raise ValueError(f"unknown Kokoro voice '{kokoro_voice}'. Options: "
                                 + ", ".join(self.kokoro.get_voices()))
            self.render("Ready.", Path(tempfile.gettempdir()) / "nova_warmup.wav")   # first call is slow; do it now

    def render(self, text, path):
        """Write `text` as speech to `path` (.wav, 16-bit mono), through RVC if enabled."""
        if not self.rvc:
            return self._speak(text, path)
        raw = Path(path).with_name(Path(path).stem + "_raw.wav")
        self._speak(text, raw)
        try:
            self.rvc.convert(raw, path)
        except Exception as e:                       # never go silent: fall back to plain voice
            print(f"\n(RVC failed: {e} - using plain voice)", flush=True)
            shutil.copyfile(raw, path)
        return True

    def _speak(self, text, path):
        if self.engine == "kokoro":
            import numpy as np
            lang = "en-gb" if self.kokoro_voice.startswith("b") else "en-us"
            samples, rate = self.kokoro.create(text, voice=self.kokoro_voice,
                                               speed=self.speed, lang=lang)
            pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(pcm)
            return True
        cmd = ["say"] + (["-v", self.mac_voice] if self.mac_voice else []) + \
              ["-o", str(path), "--data-format=LEI16@22050", text]
        subprocess.run(cmd, check=True, capture_output=True)
        return True

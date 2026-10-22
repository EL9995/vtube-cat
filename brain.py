"""
brain.py - the "brain" for the AI VTuber MVP.

You type -> Hermes (via Ollama) streams a reply -> Nova starts speaking the first
sentence while the rest is still being written, with lip sync from the real audio.
The body daemon (vts_body.py) handles emotion, gaze and mouth.

Run order:
  1. VTube Studio open, model loaded, API on
  2. python3 vts_body.py      (terminal 1)
  3. python3 brain.py         (terminal 2)  - type to chat
     python3 brain.py --voice               - talk to her with your mic (needs ears.py)

Edit persona.md to change who Nova is. No installs needed beyond what the body uses.
"""

import json
import queue
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import wave
from array import array
from pathlib import Path

# ---------------------------------------------------------------- settings
MODEL = "hermes3:3b"            # your Ollama model name (or e.g. "hermes3:8b")
TTS_ENGINE = "kokoro"     # "kokoro" (natural AI voice) or "mac" (built-in `say`, instant setup)
KOKORO_VOICE = "af_bella" # try af_bella, af_nicole, am_michael, bf_emma... (list in voice.py)
VOICE_SPEED = 1.0         # 0.8 = slower, 1.2 = faster
RVC_MODEL = None          # folder with an RVC .pth (+ .index), or None to turn RVC off
RVC_PITCH = -12           # semitones. Female voice -> male RVC: -12. Male -> female: +12. Same: 0
RVC_INDEX_RATE = 0.4      # 0-1: higher = more like the character, lower = clearer words
VOICE = None              # Mac voice for TTS_ENGINE="mac", e.g. "Zoe (Premium)". `say -v ?` lists them
OLLAMA_URL = "http://localhost:11434/api/chat"
BODY = ("127.0.0.1", 47865)
PERSONA_FILE = Path(__file__).with_name("persona.md")
WATCH_SCREEN = False      # True = watches your screen and commentates (or run with --watch). Needs eyes.py
COMMENT_COOLDOWN = 12     # min seconds of quiet before commenting on the screen unprompted
WATCH_MODE = "game"       # "game" = roast their gameplay, "video" = heckle whatever they're watching
WATCH_STYLES = {
    "game": ("You're watching the player play a video game, through [screen: ...] notes. "
             "Trash-talk their gameplay: ruthless but playful, in character, one short line. "
             "Don't describe the screen back to them. React to it like you're watching over their shoulder."),
    "video": ("You're watching a video with the user, through [screen: ...] notes. "
              "React like a friend on the couch: hot takes, heckling, jokes about what's on screen, "
              "in character, one short line. Don't summarize the video back to them."),
}
MAX_SENTENCES = 3         # hard cap per reply; extra sentences are dropped
HISTORY_TURNS = 6         # back-and-forths remembered (small models parrot long histories)
KEEP_WARM = "30m"         # keep Hermes loaded in memory between messages
LIPSYNC_FPS = 30
MOUTH_FLOOR = 0.9         # 0-1.2: raise if the mouth hangs open, lower if it barely opens
MOUTH_CONTRAST = 1.4      # 1 = soft, 2 = snappy open/close on each syllable
MOUTH_GAIN = 1.6          # how wide the mouth opens overall
AUDIO_DELAY = 0.04        # seconds the speakers lag behind; raise if mouth leads the voice

TAG_RE = re.compile(r"\[(emotion|look|anim|energy)\s*:\s*([^\]]+)\]", re.I)
SENTENCE_RE = re.compile(r"^(.*?[.!?]+[\"')\]]*)\s+", re.S)   # a finished sentence + following space
TMP = Path(tempfile.gettempdir())


# ---------------------------------------------------------------- body link
class BodyLink:
    def __init__(self):
        self.lock = threading.Lock()
        try:
            self.sock = socket.create_connection(BODY, timeout=5)
            self.f = self.sock.makefile("rw")
        except OSError:
            self.sock = None

    def send(self, cmd):
        if not self.sock:
            return ""
        with self.lock:   # the speaker thread and main thread both talk to the body
            try:
                self.f.write(cmd + "\n")
                self.f.flush()
                return self.f.readline().strip()
            except OSError:
                self.sock = None
                return ""


def model_options(body):
    """Ask the body which expressions and animations this avatar has."""
    reply = body.send("list")
    emos, anims = [], []
    m = re.search(r"expressions:\s*(.*?)\s*(?:\|\s*hotkeys:\s*(.*))?$", reply)
    if m:
        if m.group(1) and m.group(1) != "(none)":
            emos = [e.strip() for e in m.group(1).split(",") if e.strip()]
        if m.group(2):
            anims = [h.split("[")[0].strip() for h in m.group(2).split(",")
                     if "TriggerAnimation" in h]
    return emos, anims


# ---------------------------------------------------------------- prompt
def split_examples(persona):
    """Pull 'User: ... / Name: ...' example pairs out of the persona text.
    They're fed to the model as real earlier messages instead, so it copies the style
    without continuing a 'User:/Name:' script on its own."""
    keep, examples, pending_user = [], [], None
    for line in persona.splitlines():
        m = re.match(r"^\s*(User|You|Me)\s*:\s*(.+)$", line)
        a = re.match(r"^\s*([A-Z][\w ]{0,20}?)\s*:\s*(.+)$", line)
        if m:
            pending_user = m.group(2).strip()
        elif a and pending_user:
            examples += [{"role": "user", "content": pending_user},
                         {"role": "assistant", "content": a.group(2).strip()}]
            pending_user = None
        elif line.strip().lower().rstrip(":") in ("example exchanges", "examples"):
            continue
        else:
            keep.append(line)
    return "\n".join(keep).strip(), examples


def build_system(emos, anims):
    persona = PERSONA_FILE.read_text().strip() if PERSONA_FILE.exists() else "You are Nova, a VTuber."
    persona, examples = split_examples(persona)
    lines = [
        persona, "",
        "You control an animated avatar. Begin each reply with 0-3 tags, then your spoken line.",
        "Tags (only use these exact options):",
        f"[emotion:X] X = {', '.join(emos)}, or neutral" if emos else "[emotion:neutral]",
        "[look:X] X = center, left, right, up, down, chat",
        "[energy:N] N = 0.0 (calm) to 1.0 (hyped)",
    ]
    if anims:
        lines.append(f"[anim:X] X = {', '.join(anims)} (use rarely)")
    lines += [
        "Example: [emotion:happy][energy:0.8] Oh nice, that's actually sick.",
        "Never mention the tags. Everything after them is spoken aloud, so no emojis, "
        "asterisks, or stage directions.",
        f"Reply ONLY with your own next line. Never write lines for the user. Max {MAX_SENTENCES} sentences.",
    ]
    return "\n".join(lines), examples


STOP_AT = ["User:", "\nUser", "You:", "\nYou:", "Me:"]   # cut it off if it starts writing your lines


def ask_stream(messages, stops=()):
    """Yield the reply from Ollama a few words at a time."""
    data = json.dumps({
        "model": MODEL, "messages": messages, "stream": True, "keep_alive": KEEP_WARM,
        "options": {"num_predict": 120, "num_ctx": 4096,
                    "repeat_penalty": 1.15, "repeat_last_n": 256,     # discourage repeating itself
                    "stop": STOP_AT + list(stops)},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        for line in r:
            if not line.strip():
                continue
            j = json.loads(line)
            if j.get("error"):
                raise RuntimeError(j["error"])
            yield j.get("message", {}).get("content", "")
            if j.get("done"):
                break


def warm_up():
    """Load Hermes into memory now, so the first real reply isn't slow."""
    data = json.dumps({"model": MODEL, "messages": [], "keep_alive": KEEP_WARM}).encode()
    req = urllib.request.Request(OLLAMA_URL, data, {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=180).read()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- voice + lip sync
def clean_speech(text):
    text = TAG_RE.sub("", text)
    text = re.sub(r"\*[^*]*\*", "", text)          # *waves*
    text = re.sub(r"[^\w\s.,!?'\":;()\-]", "", text)  # emojis etc.
    return " ".join(text.split()).lstrip("-").strip()


def mouth_envelope(path):
    """Mouth openness (0-1) per animation frame, following syllables rather than raw loudness,
    so dense voices (RVC) still open and close instead of hanging open."""
    with wave.open(path) as w:
        rate, width = w.getframerate(), w.getsampwidth()
        samples = array("h", w.readframes(w.getnframes())) if width == 2 else array("h")
    step = max(1, rate // LIPSYNC_FPS)
    rms = []
    for i in range(0, len(samples), step):
        chunk = samples[i:i + step]
        rms.append((sum(x * x for x in chunk) / len(chunk)) ** 0.5 if chunk else 0.0)
    n = len(rms)
    if not n:
        return []
    ranked = sorted(rms)
    floor, peak = ranked[int(n * 0.2)], ranked[int(n * 0.95)]   # between-word noise / loud speech
    span = (peak - floor) or 1.0
    # local average over ~a quarter second: syllables are the bumps above it
    win = max(1, int(0.25 * LIPSYNC_FPS))
    env, prev = [], 0.0
    for i, r in enumerate(rms):
        local = sum(rms[max(0, i - win):i + win + 1]) / len(rms[max(0, i - win):i + win + 1])
        level = max(0.0, (r - floor) / span)                       # 0 at the noise floor
        bump = max(0.0, (r - local * MOUTH_FLOOR) / span)          # emphasis on syllable peaks
        v = min(1.0, MOUTH_GAIN * (0.4 * level + 0.6 * bump)) ** MOUTH_CONTRAST
        v = 0.0 if v < 0.1 else v                                  # snap shut between syllables
        prev = prev + (v - prev) * (0.75 if v > prev else 0.6)     # open fast, close fast-ish
        env.append(round(prev, 2))
    return env


class Speaker:
    """Two-stage pipeline: one thread renders sentences to audio, another plays them.
    So sentence 2 is being rendered while sentence 1 is being spoken."""

    def __init__(self, body, voice):
        self.body = body
        self.voice = voice
        self.enabled = bool(shutil.which("afplay"))   # macOS audio player
        self.text_q = queue.Queue()
        self.audio_q = queue.Queue()
        self.turn_done = threading.Event()
        self.n = 0
        threading.Thread(target=self._render_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    def say(self, sentence):
        self.text_q.put(sentence)

    def finish_turn(self):
        """Block until everything queued this turn has been spoken."""
        self.turn_done.clear()
        self.text_q.put(None)
        self.turn_done.wait()

    def _voice(self):
        return ["-v", VOICE] if VOICE else []

    def _render_loop(self):
        while True:
            sentence = self.text_q.get()
            if sentence is None or not self.enabled:
                self.audio_q.put(None if sentence is None else (None, None, sentence))
                continue
            self.n += 1
            path = TMP / f"nova_line_{self.n % 8}.wav"   # recycle a few files
            try:
                self.voice.render(sentence, path)
                env = mouth_envelope(str(path))
            except Exception as e:
                print(f"\n(voice error: {e})", flush=True)
                path, env = None, None
            self.audio_q.put((path, env, sentence))

    def _play_loop(self):
        while True:
            item = self.audio_q.get()
            if item is None:
                self.body.send("mouth 0")
                self.turn_done.set()
                continue
            path, env, sentence = item
            if not self.enabled:
                continue
            if not env:                              # fallback: plain say + flapping mouth
                if not shutil.which("say"):
                    continue
                self.body.send("talk 120")
                subprocess.run(["say"] + self._voice() + [sentence])
                self.body.send("talk 0")
                continue
            player = subprocess.Popen(["afplay", str(path)])
            start = time.time() + AUDIO_DELAY
            while player.poll() is None:
                i = int((time.time() - start) * LIPSYNC_FPS)
                if 0 <= i < len(env):
                    self.body.send(f"mouth {env[i]}")
                time.sleep(1 / LIPSYNC_FPS)


# ---------------------------------------------------------------- one turn
def _norm(sentence):
    return re.sub(r"[^a-z0-9 ]", "", sentence.lower()).strip()


def recent_lines(history, turns=3):
    """Sentences the character said in its last few replies (for the anti-repeat filter)."""
    said = set()
    for msg in [m for m in history if m["role"] == "assistant"][-turns:]:
        for sent in re.split(r"(?<=[.!?])\s+", clean_speech(msg["content"])):
            if len(_norm(sent).split()) >= 3:          # short reactions ("Don't.") may recur
                said.add(_norm(sent))
    return said


def respond(body, speaker, messages, recent=frozenset(), name="nova"):
    stops = [f"{name.title()}:", f"\n{name.title()}"]
    """Stream the reply; fire tags immediately and speak each sentence as it lands.
    Sentences repeated from the last few replies are skipped, which breaks catchphrase loops.
    Returns what was actually said (tags + spoken text), to store in history."""
    raw, buf, started = "", "", False
    spoken, skipped, tags = [], [], []
    t0 = time.time()

    def emit(sentence):
        nonlocal started
        sentence = clean_speech(sentence)
        if not sentence:
            return
        if _norm(sentence) in recent:
            skipped.append(sentence)
            return
        if not started:
            started = True
            print(f"{name}> ({time.time() - t0:.1f}s) ", end="", flush=True)
        print(sentence + " ", end="", flush=True)
        spoken.append(sentence)
        speaker.say(sentence)

    for chunk in ask_stream(messages, stops):
        if len(spoken) >= MAX_SENTENCES:
            break                                      # enough said; stop generating
        raw += chunk
        buf += chunk
        for kind, val in TAG_RE.findall(buf):         # act on tags the moment they appear
            body.send(f"{kind.lower()} {val.strip()}")
            tags.append(f"[{kind.lower()}:{val.strip()}]")
        buf = TAG_RE.sub("", buf)
        if "[" in buf and "]" not in buf.split("[")[-1]:
            continue                                   # half-written tag; wait for the rest
        while (m := SENTENCE_RE.match(buf)) and len(spoken) < MAX_SENTENCES:
            emit(m.group(1))
            buf = buf[m.end():]
    if len(spoken) < MAX_SENTENCES:
        emit(buf)                                      # whatever's left at the end
    if not spoken and skipped:                         # everything was a repeat: say it anyway
        recent = frozenset()
        for sentence in skipped:
            emit(sentence)
    print(flush=True)
    speaker.finish_turn()
    return "".join(tags) + " " + " ".join(spoken)


# ---------------------------------------------------------------- main
def required_packages(voice_mode=False):
    """Extra Python packages the current settings need (the launchers install them)."""
    pkgs = {}
    if TTS_ENGINE == "kokoro":
        pkgs.update({"kokoro_onnx": "kokoro-onnx", "numpy": "numpy"})
    if voice_mode:
        pkgs.update({"mlx_whisper": "mlx-whisper", "sounddevice": "sounddevice", "numpy": "numpy"})
    if WATCH_SCREEN or "--watch" in sys.argv or "--video" in sys.argv:
        pkgs.update({"mss": "mss", "PIL": "pillow"})
    return pkgs


def _similar(a, b):
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) > 0.6


def main():
    body = BodyLink()
    if not body.sock:
        print("(body not running - Nova can still chat and talk, but the avatar won't move)")
    from voice import Voice
    voice = Voice(TTS_ENGINE, KOKORO_VOICE, VOICE_SPEED, VOICE,
                  RVC_MODEL, RVC_PITCH, RVC_INDEX_RATE)
    speaker = Speaker(body, voice)
    emos, anims = model_options(body)
    system, examples = build_system(emos, anims)
    persona_text = PERSONA_FILE.read_text() if PERSONA_FILE.exists() else ""
    m = re.search(r"You are (\w+)", persona_text)
    name = m.group(1).lower() if m else "nova"
    print(f"warming up {MODEL}...", flush=True)
    if not warm_up():
        print(f"(couldn't reach Ollama with MODEL='{MODEL}'. Is Ollama running? Check `ollama list`.)")
    ears = None
    if "--voice" in sys.argv:
        try:
            from ears import Ears
            ears = Ears()
            print("loading Whisper...", flush=True)
            ears.warm_up()
            print("measuring room noise - stay quiet for a sec...", flush=True)
            ears.calibrate()
        except ImportError:
            print("(voice mode needs: python3 -m pip install mlx-whisper sounddevice numpy)")
            return
        except Exception as e:
            print(f"(couldn't open the mic: {e}. Check System Settings > Privacy > Microphone.)")
            return

    events = queue.Queue()          # everything that can make the character talk: ("user"|"screen"|"quit", ...)
    speaking = threading.Event()    # set while a reply is being generated/spoken
    turn = [0]                      # bumps every reply, so the mic can discard what it heard mid-reply

    eyes = None
    if WATCH_SCREEN or "--watch" in sys.argv or "--video" in sys.argv:
        try:
            from eyes import Eyes, VISION_MODEL
            watch_mode = "video" if "--video" in sys.argv else WATCH_MODE
            eyes = Eyes(on_caption=lambda cap, secs: events.put(("screen", cap, secs)), mode=watch_mode)
            print(f"loading vision model {VISION_MODEL}...", flush=True)
            secs = eyes.warm_up()
            print(f"eyes ready ({secs:.1f}s per look).", flush=True)
            system += "\n\n" + WATCH_STYLES.get(watch_mode, WATCH_STYLES["game"])
        except ImportError:
            print("(watch mode needs: mss pillow - use the launchers, they install it)")
            return
        except Exception as e:
            print(f"(eyes off: {e}. Did you `ollama pull` the vision model?)")
            eyes = None

    def listen_loop():
        while True:
            try:
                if ears:
                    while speaking.is_set():
                        time.sleep(0.05)
                    time.sleep(0.3)                    # let the room echo of its voice die down
                    ears.flush()                       # drop anything heard while it was talking
                    started = turn[0]
                    print("(listening...)", end="\r", flush=True)
                    text, stt_time = ears.hear()
                    if turn[0] != started or speaking.is_set():
                        continue                       # it talked while we listened; that audio is suspect
                    events.put(("user", text, stt_time))
                else:
                    events.put(("user", input("you> ").strip(), None))
            except (EOFError, KeyboardInterrupt):
                events.put(("quit",))
                return

    mode = "Just talk" if ears else "Type to chat"
    watching = " Watching your screen." if eyes else ""
    print(f"{name.title()} is live. Expressions: {', '.join(emos) or 'none'}. {mode}.{watching} "
          "Ctrl+C to quit.\n")
    threading.Thread(target=listen_loop, daemon=True).start()
    if eyes:
        eyes.start()

    history = []
    last_spoke = time.time()
    last_commented = ""
    while True:
        try:
            ev = events.get(timeout=0.5)
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            print()
            break
        # if several things piled up while it was talking: you come first, then the newest screen
        pending = [ev]
        while not events.empty():
            pending.append(events.get_nowait())
        if any(e[0] == "quit" for e in pending):
            break
        users = [e for e in pending if e[0] == "user" and e[1]]
        screens = [e for e in pending if e[0] == "screen"]
        if users:
            for extra in screens:                      # keep the newest screen note for later
                events.put(extra)
            text, stt = users[0][1], users[0][2]
            if stt is not None:
                print(f"you> {text}   ({stt:.1f}s to transcribe)")
            screen = eyes.latest() if eyes else ""
            content = f"{text}\n[screen right now: {screen}]" if screen else text
        elif screens:
            cap = screens[-1][1]
            quiet_for = time.time() - last_spoke
            if quiet_for < COMMENT_COOLDOWN or _similar(cap, last_commented):
                continue                               # too soon, or nothing new to roast
            last_commented = cap
            print(f"\n(saw: {cap})", flush=True)
            content = f"[screen: {cap}]\n(The user didn't say anything. React in ONE short line.)"
        else:
            continue

        history.append({"role": "user", "content": content})
        speaking.set()
        turn[0] += 1
        body.send("look up")                           # "thinking" glance hides model latency
        try:
            raw = respond(body, speaker,
                          [{"role": "system", "content": system}] + examples + history[-HISTORY_TURNS * 2:],
                          recent=recent_lines(history[:-1]), name=name)
            history.append({"role": "assistant", "content": raw})
        except Exception as e:
            print(f"\n(couldn't reach Ollama: {e}. Is it running, and is MODEL='{MODEL}' right?)")
            history.pop()
        finally:
            body.send("look center")
            last_spoke = time.time()
            speaking.clear()


if __name__ == "__main__":
    main()

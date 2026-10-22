"""
vts_body.py - the "body" daemon for an AI VTuber.

Keeps a VTube Studio model alive 30x per second (breathing, blinking, head sway,
gaze) and accepts high-level commands from a "brain" (Hermes, or you typing):

    emotion happy        -> turns on the model's "happy" expression (neutral clears it)
    anim wave            -> fires a VTS hotkey by name (animations, toggles, etc.)
    look chat            -> eyes/head drift toward a target (left/right/up/down/center/chat)
    look 0.5 -0.2        -> or an exact x y target, each -1..1
    energy 0.8           -> how lively the idle motion is (0 = still, 1 = hyper)
    talk 3               -> fake mouth flapping for 3 s (for testing without TTS)
    mouth 0.7            -> set mouth openness directly (the brain streams this for lip sync)
    list                 -> show the model's expressions and hotkeys
    status / help / quit

Commands come from two places:
  1. This terminal (type them in).
  2. TCP on localhost:47865, one command per line. Plain text or JSON like
     {"cmd": "emotion", "arg": "happy"}. This is the port Hermes's tool layer will use.

Requires: Python 3.10+, `pip install websockets`
In VTube Studio: Settings -> turn on the Plugin API (port 8001). The first run
shows an "Allow plugin?" popup inside VTS - click Allow. The token is saved next
to this file so you only do that once.
"""

import asyncio
import json
import math
import os
import random
import time
import uuid
from pathlib import Path

import websockets

# ---------------------------------------------------------------- settings
VTS_URL = "ws://127.0.0.1:8001"
PLUGIN_NAME = "AI Body Daemon"
PLUGIN_DEV = "Eric"
TOKEN_FILE = Path(__file__).with_name("vts_token.txt")
COMMAND_HOST, COMMAND_PORT = "127.0.0.1", 47865
TICK_HZ = 30
FLIP_X = 1  # set to -1 if "look left" makes the model look right

GAZE_TARGETS = {
    "center": (0.0, 0.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up": (0.0, 1.0),
    "down": (0.0, -1.0),
    "chat": (0.8, -0.4),  # where a chat box usually sits; tweak to your layout
}

# VTS default input parameters this daemon drives
PARAMS = [
    "FaceAngleX", "FaceAngleY", "FaceAngleZ", "FacePositionY",
    "EyeOpenLeft", "EyeOpenRight",
    "EyeLeftX", "EyeLeftY", "EyeRightX", "EyeRightY",
    "MouthOpen",
]


def log(msg):
    print(f"[body] {msg}", flush=True)


class VTSError(Exception):
    pass


# ---------------------------------------------------------------- VTS connection
class VTSClient:
    """Thin wrapper over the VTube Studio websocket API."""

    def __init__(self, url):
        self.url = url
        self.ws = None
        self.pending = {}
        self._last_error = None
        self._reader_task = None

    async def connect(self):
        self.ws = await websockets.connect(self.url, max_size=16 * 1024 * 1024)
        self._reader_task = asyncio.create_task(self._reader())

    async def close(self):
        if self.ws:
            await self.ws.close()

    async def _reader(self):
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                fut = self.pending.pop(msg.get("requestID"), None)
                if fut and not fut.done():
                    fut.set_result(msg)
                elif msg.get("messageType") == "APIError":
                    # errors from fire-and-forget parameter injection; log once, not 30x/sec
                    err = msg.get("data", {}).get("message")
                    if err != self._last_error:
                        log(f"VTS says: {err}")
                        self._last_error = err
        except websockets.ConnectionClosed:
            pass
        finally:
            for fut in self.pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("VTS disconnected"))
            self.pending.clear()

    @staticmethod
    def _envelope(mtype, data, rid):
        return json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": rid,
            "messageType": mtype,
            "data": data or {},
        })

    async def request(self, mtype, data=None, timeout=10):
        """Send a request and wait for its response. Raises VTSError on API errors."""
        rid = uuid.uuid4().hex[:32]
        fut = asyncio.get_running_loop().create_future()
        self.pending[rid] = fut
        await self.ws.send(self._envelope(mtype, data, rid))
        msg = await asyncio.wait_for(fut, timeout)
        if msg.get("messageType") == "APIError":
            raise VTSError(msg.get("data", {}).get("message", "unknown error"))
        return msg.get("data", {})

    async def send(self, mtype, data):
        """Fire and forget (used for the 30 Hz parameter stream)."""
        await self.ws.send(self._envelope(mtype, data, "stream"))


async def authenticate(vts):
    ident = {"pluginName": PLUGIN_NAME, "pluginDeveloper": PLUGIN_DEV}
    token = TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else None
    if token:
        res = await vts.request("AuthenticationRequest", {**ident, "authenticationToken": token})
        if res.get("authenticated"):
            log("authenticated with saved token")
            return
        log("saved token rejected, asking VTS again")
    log("check VTube Studio - click ALLOW on the plugin popup")
    res = await vts.request("AuthenticationTokenRequest", ident, timeout=120)
    token = res["authenticationToken"]
    TOKEN_FILE.write_text(token)
    res = await vts.request("AuthenticationRequest", {**ident, "authenticationToken": token})
    if not res.get("authenticated"):
        raise VTSError("authentication failed")
    log("authenticated, token saved")


# ---------------------------------------------------------------- the body
class Body:
    def __init__(self):
        self.vts = None
        self.available = set()
        self.expressions = []   # [{"name":..., "file":...}]
        self.hotkeys = []       # [{"name":..., "type":...}]
        self.active_expr = None
        self.active_emotion = "neutral"

        self.gaze_target = [0.0, 0.0]
        self.gaze = [0.0, 0.0]
        self.energy = 0.5
        self.talk_until = 0.0
        self.mouth_val = 0.0
        self.mouth_until = 0.0

        self.next_blink = time.time() + 2
        self.blink_start = None
        self.double_blink = False
        self.seed = random.uniform(0, 100)  # so two instances don't sway in sync

    # ---- startup: learn what this model supports
    async def load_model_info(self):
        model = await self.vts.request("CurrentModelRequest")
        if not model.get("modelLoaded"):
            log("no model loaded in VTS - load one, then restart this script")
        else:
            log(f"model: {model.get('modelName')}")

        plist = await self.vts.request("InputParameterListRequest")
        names = {p["name"] for p in plist.get("defaultParameters", []) + plist.get("customParameters", [])}
        self.available = {p for p in PARAMS if p in names}
        missing = set(PARAMS) - self.available
        if missing:
            log(f"VTS doesn't offer these params, skipping: {sorted(missing)}")

        ex = await self.vts.request("ExpressionStateRequest", {"details": False})
        self.expressions = [{"name": e["name"], "file": e["file"]} for e in ex.get("expressions", [])]
        hk = await self.vts.request("HotkeysInCurrentModelRequest")
        self.hotkeys = [{"name": h["name"], "type": h["type"]} for h in hk.get("availableHotkeys", [])]
        log(f"{len(self.expressions)} expressions, {len(self.hotkeys)} hotkeys - type 'list' to see them")

    # ---- one animation frame
    def compute(self, now, dt):
        t = now + self.seed
        e = 0.2 + 0.8 * self.energy  # never fully dead

        # smooth gaze toward target
        k = min(1.0, dt * 4.0)
        for i in range(2):
            self.gaze[i] += (self.gaze_target[i] - self.gaze[i]) * k
        gx, gy = self.gaze[0] * FLIP_X, self.gaze[1]

        # layered sine "noise" for idle sway
        sway_x = e * (5 * math.sin(t * 0.6) + 2.5 * math.sin(t * 1.7 + 1.3))
        sway_y = e * (3 * math.sin(t * 0.45 + 2.1) + 1.5 * math.sin(t * 1.3))
        roll = e * (4 * math.sin(t * 0.35 + 4.0) + 1.5 * math.sin(t * 0.9))
        breath = 0.5 * math.sin(now * 2 * math.pi / 4.0)  # ~15 breaths/min

        eye_open = self._blink(now)

        v = {
            "FaceAngleX": gx * 15 + sway_x,
            "FaceAngleY": gy * 10 + sway_y,
            "FaceAngleZ": roll,
            "FacePositionY": breath,
            "EyeOpenLeft": eye_open,
            "EyeOpenRight": eye_open,
            "EyeLeftX": gx * 0.8, "EyeRightX": gx * 0.8,
            "EyeLeftY": gy * 0.6, "EyeRightY": gy * 0.6,
        }
        # Only own the mouth while talking; otherwise VTS's mic lip sync takes it back.
        if now < self.mouth_until:          # real lip sync from the brain's audio
            v["MouthOpen"] = self.mouth_val
        elif now < self.talk_until:
            m = 0.55 * math.sin(now * 14) + 0.45 * math.sin(now * 23 + 1)
            v["MouthOpen"] = max(0.0, min(1.0, abs(m)))
        return {k_: val for k_, val in v.items() if k_ in self.available}

    def _blink(self, now):
        if self.blink_start is None and now >= self.next_blink:
            self.blink_start = now
        if self.blink_start is None:
            return 1.0
        p = (now - self.blink_start) / 0.16
        if p >= 1:
            self.blink_start = None
            if not self.double_blink and random.random() < 0.15:
                self.double_blink = True
                self.next_blink = now + 0.12
            else:
                self.double_blink = False
                self.next_blink = now + random.uniform(2.0, 6.0)
            return 1.0
        return abs(1 - 2 * p)  # 1 -> 0 -> 1

    # ---- commands
    async def set_emotion(self, name):
        name = name.strip().lower()
        if self.active_expr:
            await self.vts.request("ExpressionActivationRequest",
                                   {"expressionFile": self.active_expr, "active": False})
            self.active_expr = None
        if name in ("neutral", "none", "clear", ""):
            self.active_emotion = "neutral"
            return "emotion: neutral"
        match = next((x for x in self.expressions if name in x["name"].lower()), None)
        if not match:
            self.active_emotion = "neutral"
            opts = ", ".join(x["name"] for x in self.expressions) or "(model has none)"
            return f"no expression matching '{name}'. options: {opts}"
        await self.vts.request("ExpressionActivationRequest",
                               {"expressionFile": match["file"], "active": True, "fadeTime": 0.3})
        self.active_expr = match["file"]
        self.active_emotion = match["name"]
        return f"emotion: {match['name']}"

    async def handle(self, line):
        line = line.strip()
        if not line:
            return ""
        if line.startswith("{"):
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                return "bad JSON"
            cmd, arg = str(j.get("cmd", "")).lower(), str(j.get("arg", ""))
        else:
            parts = line.split(maxsplit=1)
            cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

        if cmd == "help":
            return __doc__.split("Commands come from")[0].strip()
        if cmd == "quit":
            log("bye")
            os._exit(0)
        if cmd == "energy":
            try:
                self.energy = max(0.0, min(1.0, float(arg)))
            except ValueError:
                return "energy needs a number 0-1"
            return f"energy: {self.energy:.2f}"
        if cmd == "look":
            if arg.lower() in GAZE_TARGETS:
                self.gaze_target = list(GAZE_TARGETS[arg.lower()])
            else:
                try:
                    x, y = (float(n) for n in arg.split())
                    self.gaze_target = [max(-1, min(1, x)), max(-1, min(1, y))]
                except ValueError:
                    return f"look needs one of {list(GAZE_TARGETS)} or 'x y'"
            return f"looking at {self.gaze_target}"
        if cmd == "mouth":
            try:
                self.mouth_val = max(0.0, min(1.0, float(arg)))
            except ValueError:
                return "mouth needs 0-1"
            self.mouth_until = time.time() + 0.25
            return "ok"
        if cmd == "talk":
            try:
                secs = float(arg or 2)
            except ValueError:
                return "talk needs seconds"
            self.talk_until = time.time() + secs
            return f"talking {secs:.1f}s"
        if cmd == "status":
            return (f"connected={self.vts is not None} emotion={self.active_emotion} "
                    f"energy={self.energy:.2f} gaze_target={self.gaze_target}")

        # everything below needs VTS
        if self.vts is None:
            return "not connected to VTube Studio yet"
        try:
            if cmd == "emotion":
                return await self.set_emotion(arg)
            if cmd == "anim":
                res = await self.vts.request("HotkeyTriggerRequest", {"hotkeyID": arg})
                return f"fired hotkey {arg} ({res.get('hotkeyID')})"
            if cmd == "list":
                ex = ", ".join(x["name"] for x in self.expressions) or "(none)"
                hk = ", ".join(f"{h['name']} [{h['type']}]" for h in self.hotkeys) or "(none)"
                return f"expressions: {ex}\nhotkeys: {hk}"
        except (VTSError, ConnectionError, asyncio.TimeoutError) as ex_:
            return f"VTS error: {ex_}"
        return f"unknown command '{cmd}' - try help"


# ---------------------------------------------------------------- loops
async def vts_loop(body):
    """Connect, authenticate, stream parameters; reconnect forever if VTS drops."""
    period = 1.0 / TICK_HZ
    while True:
        vts = VTSClient(VTS_URL)
        try:
            log(f"connecting to VTube Studio at {VTS_URL}...")
            await asyncio.wait_for(vts.connect(), 5)
            log(f"connected to {VTS_URL}")
            await authenticate(vts)
            body.vts = vts
            await body.load_model_info()
            last = time.time()
            while True:
                now = time.time()
                values = body.compute(now, now - last)
                last = now
                await vts.send("InjectParameterDataRequest", {
                    "faceFound": True,
                    "mode": "set",
                    "parameterValues": [{"id": k, "value": float(v)} for k, v in values.items()],
                })
                await asyncio.sleep(period)
        except (OSError, websockets.ConnectionClosed, ConnectionError, VTSError, asyncio.TimeoutError) as e:
            body.vts = None
            log(f"VTS not reachable ({e.__class__.__name__}: {e}). retrying in 3s...")
            await vts.close()
            await asyncio.sleep(3)


async def tcp_server(body):
    async def client(reader, writer):
        while line := await reader.readline():
            reply = await body.handle(line.decode(errors="ignore"))
            writer.write((reply.replace("\n", " | ") + "\n").encode())
            await writer.drain()
        writer.close()

    try:
        server = await asyncio.start_server(client, COMMAND_HOST, COMMAND_PORT)
    except OSError:
        log(f"port {COMMAND_PORT} is busy - another vts_body.py is already running.")
        log("fix: run  pkill -f vts_body.py  then start this again.")
        os._exit(1)
    log(f"brain port listening on {COMMAND_HOST}:{COMMAND_PORT}")
    async with server:
        await server.serve_forever()


async def terminal(body):
    while True:
        line = await asyncio.to_thread(input, "> ")
        reply = await body.handle(line)
        if reply:
            print(reply, flush=True)


async def main():
    body = Body()
    log("type 'help' for commands")
    await asyncio.gather(vts_loop(body), tcp_server(body), terminal(body))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        os._exit(0)

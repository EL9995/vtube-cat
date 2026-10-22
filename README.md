# VTube Cat

A fully local AI VTuber for macOS. You talk (or type), a small LLM answers in character, a neural
voice speaks it, and a VTube Studio avatar moves its mouth, face and eyes to match. It can also
watch your screen and react to what you're doing.

Nothing leaves the machine: the LLM, speech-to-text, text-to-speech and vision all run on a laptop.

```
mic ──► ears.py (Whisper, MLX) ──┐
                                 ├──► brain.py ──► Ollama (Hermes 3) ──► streamed reply
screen ──► eyes.py (Qwen2.5-VL) ─┘        │
                                          ├── [emotion:happy] [look:chat] tags ──► vts_body.py ──► VTube Studio
                                          └── sentences ──► voice.py (Kokoro, optional RVC) ──► speakers
                                                                   └── loudness envelope ──► lip sync
```

## What makes it work

- **Talks before it's done thinking.** Replies stream from the model, and each sentence is sent to
  TTS the moment it ends. A two-thread pipeline renders sentence 2 while sentence 1 plays, so the
  first words come out while the model is still writing the rest.
- **Lip sync from the real audio.** The mouth isn't flapping on a timer: `brain.py` builds an
  openness curve from the rendered WAV (RMS per frame, a noise floor, syllable peaks above a local
  average) and streams it to the avatar at 30 fps.
- **The body and the brain are separate processes.** `vts_body.py` keeps the model alive on its own
  (breathing, blinking, head sway, gaze) and takes plain-text or JSON commands over a local TCP port.
  The LLM drives it with inline tags like `[emotion:annoyed] [look:left]`, which are stripped before
  speaking. You can drive the body by hand, or with `test_brain.py`, with no LLM at all.
- **Small-model guardrails.** A 3B model loops and parrots easily, so the brain caps sentences per
  reply, trims history, cuts generation when the model starts writing the user's lines, and drops
  sentences it already said in recent turns. Persona examples are fed as real past messages instead
  of a script, so the model copies the style without continuing the script.
- **Doesn't hear itself.** In voice mode the mic is ignored while the character speaks and for a
  moment after, and anything captured during a reply is thrown away.
- **Screen commentary.** `eyes.py` grabs the screen about once a second, skips frames that didn't
  change, and asks a small vision model for a one-line caption. The character comments on new
  things on its own, with a cooldown, and always answers you first.
- **One-command setup.** The launchers check for missing packages and re-run themselves through
  `uv` on Python 3.12 with exactly what they need. Voice models download on first use.

## Run it

You need macOS on Apple Silicon, [Ollama](https://ollama.com), [VTube Studio](https://denchisoft.com)
with a model loaded, and [uv](https://docs.astral.sh/uv/).

```bash
ollama pull hermes3:3b
pip install websockets
```

In VTube Studio, turn on the Plugin API (port 8001). Then use two terminals:

```bash
python3 vts_body.py        # terminal 1: the body. Click "Allow" in VTube Studio the first time
python3 brain_type.py      # terminal 2: type to chat, it answers out loud
python3 brain_voice.py     # ...or talk to it with your mic
```

Add `--watch` to have it commentate your game, or `--video` to have it heckle what you're watching
(run `ollama pull qwen2.5vl:3b` first).

## Make it your own

- **Persona:** `persona.md` holds the character: who they are, how they talk, and example exchanges.
  The default is Rook, a retired operative stuck in a cat avatar. `personas/` has more; copy one
  over `persona.md` to switch.
- **Settings** sit at the top of each file: the model, voice, speaking speed and lip sync tuning in
  `brain.py`, mic sensitivity in `ears.py`, and the vision model in `eyes.py`.
- **Voice conversion (optional):** set `RVC_MODEL` in `brain.py` to a folder holding an RVC `.pth`
  voice you have the rights to use. Each sentence is then re-voiced through
  [Applio](https://github.com/IAHispano/Applio) in a background worker that keeps the model loaded.

## Files

| File | What it does |
|---|---|
| `brain.py` | Prompting, streaming, tag parsing, the speech pipeline and lip sync |
| `vts_body.py` | VTube Studio WebSocket client, idle animation, and the command port |
| `ears.py` | Mic capture, voice activity detection, on-device Whisper |
| `eyes.py` | Screen capture, change detection, vision captions |
| `voice.py` | Kokoro and macOS TTS engines, RVC worker management |
| `rvc_worker.py` | Background voice conversion over stdin/stdout JSON |
| `launcher.py`, `brain_type.py`, `brain_voice.py` | Entry points that set up dependencies |
| `test_brain.py` | Drives the body with a scripted sequence, with no LLM |

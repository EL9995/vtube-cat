"""
brain_voice.py - TALK to Nova with your mic, she answers out loud.

Run (with vts_body.py already running in another window):
    python3 brain_voice.py           (add --watch to roast your gaming, --video to watch videos with you)

Speech recognition (and the Kokoro voice) need Python 3.12, so this relaunches itself
through uv on Python 3.12 when needed. Settings live in brain.py (model, voice)
and ears.py (mic sensitivity).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import brain     # noqa: E402
import launcher  # noqa: E402

launcher.ensure(brain.required_packages(voice_mode=True))
sys.argv = [sys.argv[0], "--voice"] + [a for a in sys.argv[1:] if a in ("--watch", "--video")]   # mic mode
brain.main()

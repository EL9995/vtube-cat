"""
brain_type.py - TYPE to Nova, she answers out loud.

Run (with vts_body.py already running in another window):
    python3 brain_type.py            (add --watch to roast your gaming, --video to watch videos with you)

Settings (model, voice, lip sync timing) live in brain.py - both launchers share them.
If the voice needs extra packages (Kokoro), this relaunches itself on Python 3.12 via uv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import brain     # noqa: E402
import launcher  # noqa: E402

launcher.ensure(brain.required_packages(voice_mode=False))
sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a in ("--watch", "--video")]   # typing mode
brain.main()

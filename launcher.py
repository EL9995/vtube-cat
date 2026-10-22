"""
launcher.py - shared helper for brain_type.py / brain_voice.py.

Kokoro and Whisper need Python 3.12 (newer Pythons aren't supported yet). If the
needed packages can't be imported here, relaunch the calling script through `uv`
on Python 3.12 with those packages. uv caches everything, so it's fast after the
first run.
"""

import importlib.util
import os
import shutil
import sys
from pathlib import Path


def ensure(packages):
    """packages: {import_name: pip_name}. Returns only if all can be imported."""
    missing = [pip for mod, pip in packages.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if os.environ.get("NOVA_VIA_UV") or not Path(uv).exists():
        print(f"Missing: {', '.join(missing)}.\n"
              "Install uv once with:\n"
              "  curl -LsSf https://astral.sh/uv/install.sh | sh\n"
              "then open a new Terminal window and run this again.")
        sys.exit(1)
    os.environ["NOVA_VIA_UV"] = "1"                    # prevents an endless relaunch loop
    cmd = [uv, "run", "--python", "3.12"]
    for pip in packages.values():
        cmd += ["--with", pip]
    cmd += [str(Path(sys.argv[0]).resolve())] + sys.argv[1:]
    print("switching to Python 3.12 (first time takes a minute)...", flush=True)
    os.execv(uv, cmd)

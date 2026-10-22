"""
test_brain.py - pretends to be Hermes. Sends a scripted sequence of commands to
vts_body.py over its brain port so you can watch the model react.

Run vts_body.py first (in one terminal), then this (in another):
    python test_brain.py
"""

import socket
import time

SCRIPT = [
    ("energy 0.4", 2),
    ("look chat", 2),          # glance at chat
    ("emotion happy", 1),      # needs an expression with 'happy' in its name
    ("talk 3", 3.5),           # "says" something
    ("look left", 1.5),
    ("look right", 1.5),
    ("energy 1", 4),           # gets hyped
    ('{"cmd": "emotion", "arg": "neutral"}', 1),  # JSON form, like a tool call
    ("energy 0.3", 1),
    ("look center", 1),
    ("status", 0),
]

with socket.create_connection(("127.0.0.1", 47865)) as s:
    f = s.makefile("rw")
    for cmd, wait in SCRIPT:
        f.write(cmd + "\n")
        f.flush()
        print(f"brain> {cmd:45s} body> {f.readline().strip()}")
        time.sleep(wait)

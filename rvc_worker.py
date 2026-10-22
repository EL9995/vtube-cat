"""
rvc_worker.py - background voice-conversion worker for Nova (started by voice.py).

Keeps an RVC voice model loaded (using Applio's engine) and converts one sentence at a time:
reads JSON jobs on stdin, writes JSON replies on stdout. You never run this directly.

    stdin : {"in": "kokoro.wav", "out": "rvc.wav", "model": ".pth", "index": ".index",
             "pitch": 0, "index_rate": 0.3, "protect": 0.33, "f0_method": "rmvpe"}
    stdout: {"ok": true, "secs": 0.41}   or   {"ok": false, "error": "..."}
"""

import json
import os
import sys
import time
import traceback
import types

APPLIO = os.path.abspath(sys.argv[1])
PROTO = sys.stdout                 # our replies go here...
sys.stdout = sys.stderr            # ...and Applio's chatter goes to stderr
os.chdir(APPLIO)                   # Applio finds its models relative to its own folder
sys.path.insert(0, APPLIO)
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def reply(**kw):
    PROTO.write(json.dumps(kw) + "\n")
    PROTO.flush()


def cache_faiss_index():
    """Applio re-reads the (often 500 MB) .index file for every clip. Load it once instead."""
    import faiss
    import rvc.infer.pipeline as pipeline

    cache = {}

    class CachedIndex:
        def __init__(self, idx):
            self._idx, self._vectors = idx, None

        def reconstruct_n(self, start, n):
            if self._vectors is None:
                self._vectors = self._idx.reconstruct_n(start, n)
            return self._vectors

        def __getattr__(self, name):
            return getattr(self._idx, name)

    def read_index(path):
        if path not in cache:
            cache[path] = CachedIndex(faiss.read_index(path))
        return cache[path]

    pipeline.faiss = types.SimpleNamespace(read_index=read_index)


def main():
    try:
        from rvc.lib.tools.prerequisites_download import prequisites_download_pipeline
        needed = ["rvc/models/embedders/contentvec/pytorch_model.bin",
                  "rvc/models/predictors/rmvpe.pt"]
        if not all(os.path.exists(p) for p in needed):
            print("downloading RVC helper models (one time, ~400 MB)...")
            prequisites_download_pipeline(pretraineds_hifigan=False, models=True, exe=False)
        cache_faiss_index()
        from rvc.infer.infer import VoiceConverter
        vc = VoiceConverter()
    except Exception as e:
        traceback.print_exc()
        reply(ready=False, error=f"{type(e).__name__}: {e}")
        return
    reply(ready=True, device=str(vc.config.device))

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            job = json.loads(line)
            t0 = time.time()
            vc.convert_audio(
                audio_input_path=job["in"],
                audio_output_path=job["out"],
                model_path=job["model"],
                index_path=job.get("index", ""),
                pitch=int(job.get("pitch", 0)),
                index_rate=float(job.get("index_rate", 0.3)),
                protect=float(job.get("protect", 0.33)),
                f0_method=job.get("f0_method", "rmvpe"),
                volume_envelope=1.0,
                export_format="WAV",
            )
            if not os.path.exists(job["out"]):
                raise RuntimeError("conversion produced no audio")
            reply(ok=True, secs=round(time.time() - t0, 2))
        except Exception as e:
            traceback.print_exc()
            reply(ok=False, error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

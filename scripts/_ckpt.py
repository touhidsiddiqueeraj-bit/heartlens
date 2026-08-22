"""Tiny checkpoint helper — atomic write, no deps."""
import json, os
from pathlib import Path

def atomic_json_write(path: Path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def load_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default

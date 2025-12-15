"""
Backward-compatible shim for the Trainer class.
The implementation now lives in `granted.training.trainer`.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
for p in (SRC_ROOT, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from granted.training.trainer import Trainer  # noqa: F401,E402

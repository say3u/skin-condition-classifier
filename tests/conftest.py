"""
Shared pytest config. Makes `backend` importable regardless of where pytest is
invoked from (project root, inside tests/, in CI, etc.).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

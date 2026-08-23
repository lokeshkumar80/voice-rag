"""
Put the repo root on sys.path so `from src.metrics import ...` resolves.

Needed because bare `pytest` does not add the working directory to sys.path --
only `python -m pytest` does, via the `-m` flag. Without this the suite passes
locally and fails in CI with ModuleNotFoundError, which is a miserable way to
find out. Doing it here makes both invocations work.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

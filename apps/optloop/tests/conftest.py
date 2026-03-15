"""Add scripts/ to sys.path so bench_common can be imported directly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

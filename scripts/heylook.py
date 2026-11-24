#!/usr/bin/env python3
"""Heylook CLI wrapper (`uv run scripts/heylook.py ...` or `uv run heylook ...`)."""

from __future__ import annotations

import sys

from heylook_llm.cli import main

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Deprecated alias — use resolve_pytorch_cu132_versions.py."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.stderr.write("resolve_pytorch_cu128_versions.py is deprecated; using cu132 resolver.\n")
runpy.run_path(str(Path(__file__).with_name("resolve_pytorch_cu132_versions.py")), run_name="__main__")

"""Reload layout modules before importing apply/place helpers."""

from __future__ import annotations

import importlib
import sys

REPO = "/Users/samy/c/touch/samysd/touchdesigner"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import layout_utils
import network_layout

importlib.reload(network_layout)
importlib.reload(layout_utils)

place = layout_utils.place
apply_layout = layout_utils.apply_layout
apply_project_layout = layout_utils.apply_project_layout
collect_positions = layout_utils.collect_positions
managed_roots = layout_utils.managed_roots

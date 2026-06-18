"""
Restore saved node positions only (no rebuild).

    exec(open("/Users/samy/c/touch/samysd/touchdesigner/restore_network_layout.py", encoding="utf-8").read())
"""

try:
    INSTANCE
except NameError:
    INSTANCE = "all"

import sys

REPO = "/Users/samy/c/touch/samysd/touchdesigner"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from instances import INSTANCES, get_instance
from td_layout import apply_layout, apply_project_layout

if INSTANCE == "all":
    keys = list(INSTANCES)
else:
    keys = [get_instance(INSTANCE).key]

placed = apply_project_layout()
for key in keys:
    placed += apply_layout(get_instance(key))

print(f"Restored {placed} node positions from network_layout.py")

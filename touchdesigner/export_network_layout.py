"""
Capture live nodeX/nodeY into network_layout.py.

Run in TouchDesigner after arranging the network:

    exec(open("/Users/samy/c/touch/samysd/touchdesigner/export_network_layout.py", encoding="utf-8").read())

Optional — one instance only:

    INSTANCE = "a"
    exec(open(".../export_network_layout.py", encoding="utf-8").read())
"""

from __future__ import annotations

import datetime
import sys

REPO = "/Users/samy/c/touch/samysd/touchdesigner"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from instances import INSTANCES, get_instance
from td_layout import collect_positions, managed_roots
from network_layout import PROJECT_ROOTS

try:
    INSTANCE
except NameError:
    INSTANCE = "all"

OUTPUT = f"{REPO}/network_layout.py"

if INSTANCE == "all":
    instance_keys = list(INSTANCES)
else:
    instance_keys = [get_instance(INSTANCE).key]

roots: list[str] = list(PROJECT_ROOTS)
for key in instance_keys:
    profile = get_instance(key)
    roots.extend(managed_roots(profile))

positions = collect_positions(roots)
captured_at = datetime.date.today().isoformat()

lines = [
    '"""Frozen TouchDesigner node positions (nodeX, nodeY).',
    "",
    "Regenerate after moving ops in the network:",
    f'    exec(open("{REPO}/export_network_layout.py", encoding="utf-8").read())',
    "",
    'Optional: INSTANCE = "b" before export to capture one instance only (default scans all).',
    '"""',
    "",
    "from __future__ import annotations",
    "",
    f'LAYOUT_CAPTURED = "{captured_at}"',
    "",
    "POSITIONS: dict[str, tuple[int, int]] = {",
]

for path in sorted(positions):
    x, y = positions[path]
    lines.append(f'    "{path}": ({x}, {y}),')

lines.extend(
    [
        "}",
        "",
        "PROJECT_ROOTS: tuple[str, ...] = (",
    ]
)
for path in PROJECT_ROOTS:
    lines.append(f'    "{path}",')
lines.extend([")", ""])

with open(OUTPUT, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines))

print(f"Wrote {len(positions)} positions -> {OUTPUT}")
print(f"  instances: {', '.join(instance_keys)}")
print(f"  captured:  {captured_at}")

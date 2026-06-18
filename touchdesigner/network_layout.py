"""Frozen TouchDesigner node positions (nodeX, nodeY).

Regenerate after moving ops in the network:

    exec(open("/Users/samy/c/touch/samysd/touchdesigner/export_network_layout.py", encoding="utf-8").read())
"""

from __future__ import annotations

LAYOUT_CAPTURED = "seed-from-build-scripts"

# Absolute operator paths -> (nodeX, nodeY)
POSITIONS: dict[str, tuple[int, int]] = {
    # --- /project1 (instance A control + MCP) ---
    "/project1/hal_control": (-200, 200),
    "/project1/hal_control_ui": (500, 200),
    "/project1/mcp_startup_exec": (-600, 400),
    "/project1/null1": (800, 200),
    # --- /project1/vidin ---
    "/project1/vidin/ndiout1": (-400, 0),
    "/project1/vidin/out1": (-200, 0),
    # --- /project1/vidout ---
    "/project1/vidout/ndiin2": (-700, 200),
    "/project1/vidout/info1": (-900, 200),
    "/project1/vidout/in1": (-500, 0),
    "/project1/vidout/out1": (700, 200),
    # --- /project1/vidout/combine ---
    "/project1/vidout/combine/in1": (-600, 200),
    "/project1/vidout/combine/in2": (-400, 200),
    "/project1/vidout/combine/pip_resize": (-500, 200),
    "/project1/vidout/combine/pip_place": (-350, 200),
    "/project1/vidout/combine/comp1": (-200, 200),
    "/project1/vidout/combine/text_prompt": (0, 0),
    "/project1/vidout/combine/comp2": (200, 200),
    "/project1/vidout/combine/finull": (400, 200),
    "/project1/vidout/combine/text_fps": (320, 200),
    "/project1/vidout/combine/fps_xform": (410, 200),
    "/project1/vidout/combine/comp_hud": (500, 200),
    "/project1/vidout/combine/out1": (600, 200),
    # --- instance B (control UI offset; vid paths mirror A until re-exported) ---
    "/project1/hal_control_b": (200, 200),
    "/project1/hal_control_ui_b": (1100, 200),
    "/project1/null1_b": (800, 400),
}

# Top-level /project1 comps scanned by export_network_layout.py
PROJECT_ROOTS: tuple[str, ...] = (
    "/project1/mcp_startup_exec",
    "/project1/mcp_webserver_base",
)

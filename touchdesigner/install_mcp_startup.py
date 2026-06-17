"""
Pin TouchDesigner MCP to a fixed on-disk folder so it survives restarts.

Run once after importing mcp_webserver_base.tox:

    exec(open("/Users/samy/c/touch/samysd/touchdesigner/install_mcp_startup.py", encoding="utf-8").read())

Then save your .toe and set it as TD Default Project (Preferences → General → Project).
"""

import os

MCP_ROOT = "/Users/samy/c/touch/touchmcp"
MCP_TOX = os.path.join(MCP_ROOT, "mcp_webserver_base.tox")
MCP_COMP = "/project1/mcp_webserver_base"

if not os.path.isfile(MCP_TOX):
    raise RuntimeError(f"Missing {MCP_TOX}. Extract touchdesigner-mcp-td.zip there first.")

if not os.path.isdir(os.path.join(MCP_ROOT, "modules", "mcp")):
    raise RuntimeError(f"Missing {MCP_ROOT}/modules/mcp — keep the full zip layout.")

mcp = op(MCP_COMP)
if mcp is None:
    raise RuntimeError(
        f"Import {MCP_TOX} into /project1 as mcp_webserver_base, then run this again."
    )

if hasattr(mcp.par, "externaltox"):
    mcp.par.externaltox = MCP_TOX

for name in ("mcp_webserver", "mpc_webserver"):
    ws = mcp.op(name)
    if ws is not None and hasattr(ws.par, "active"):
        ws.par.active = True
        break
else:
    raise RuntimeError("No mcp_webserver DAT found inside mcp_webserver_base.")

# Optional: project-level Execute DAT so path is repaired even if externaltox drifts.
parent = op("/project1")
exec_dat = parent.op("mcp_startup_exec")
if exec_dat is None:
    exec_dat = parent.create("executeDAT", "mcp_startup_exec")
    exec_dat.nodeX = -600
    exec_dat.nodeY = 400
exec_dat.par.active = True
exec_dat.par.start = True
exec_dat.text = f'''\
def onStart():
    mcp = op("{MCP_COMP}")
    if mcp and hasattr(mcp.par, "externaltox"):
        mcp.par.externaltox = r"{MCP_TOX}"
    ws = mcp.op("mcp_webserver") if mcp else None
    if ws is None and mcp:
        ws = mcp.op("mpc_webserver")
    if ws and hasattr(ws.par, "active"):
        ws.par.active = True
    return
'''

print("MCP pinned.")
print(f"  externaltox -> {MCP_TOX}")
print(f"  webserver active at {ws.path}")
print("Save this .toe, then set TD Preferences → General → Default Project.")

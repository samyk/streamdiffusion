"""
One-shot restore after TD crash/relaunch.

Textport:
    exec(open("/Users/samy/c/touch/samysd/touchdesigner/restore_after_crash.py", encoding="utf-8").read())
"""

try:
    INSTANCE
except NameError:
    INSTANCE = "a"

import sys

REPO = "/Users/samy/c/touch/samysd/touchdesigner"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from instances import get_instance

profile = get_instance(INSTANCE)
_ctx = globals()


def _run(path: str, label: str) -> None:
    print(f"[restore] {label}...")
    exec(open(path, encoding="utf-8").read(), _ctx)


_run(f"{REPO}/upgrade_hal_control.py", "hal_control params + prompt bind")
_run(f"{REPO}/build_ndi_video_path.py", "NDI path + scene idle gate")

ctrl = op(profile.hal_control)
vidin = op(profile.vidin)
for path, node in ((profile.hal_control, ctrl), (profile.vidin, vidin)):
    if node is None:
        print(f"[restore] WARN missing {path}")
        continue
    err = node.errors()
    if err:
        print(f"[restore] ERR {path}: {err}")
    else:
        print(f"[restore] OK {path}")

if vidin and vidin.op("scene_metric_lag"):
    hc = op(profile.hal_control)
    metric = float(vidin.op("scene_metric_lag")[0].eval())
    threshold = float(hc.par.Scenechangethreshold.eval())
    ndi_on = bool(vidin.op("ndiout1").par.active.eval())
    print(
        f"[restore] scene_idle metric={metric:.4f} threshold={threshold:.4f} "
        f"ndi_active={ndi_on} mode={hc.par.Sceneidle.eval()}"
    )

print(f"[restore] done (instance {profile.label}) — save .toe now")

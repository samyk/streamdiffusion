"""
Build one StreamDiffusion instance (control + NDI + combine layout + UI).

In TouchDesigner textport:

    exec(open("/Users/samy/c/touch/samysd/touchdesigner/build_instance.py", encoding="utf-8").read())

Two separate TD apps (recommended for dual live):
    # TD app 1
    INSTANCE = "a"
    TD_LAYOUT = "a_only"
    exec(open(".../build_instance.py", encoding="utf-8").read())

    # TD app 2
    INSTANCE = "b"
    TD_LAYOUT = "b_only"
    exec(open(".../build_instance.py", encoding="utf-8").read())

TD_LAYOUT:
  dual   - keep both instances in one project (default)
  a_only - remove instance B control/sync from this project
  b_only - remove instance A control/sync from this project
"""

try:
    INSTANCE
except NameError:
    INSTANCE = "a"

try:
    TD_LAYOUT
except NameError:
    TD_LAYOUT = "dual"

REPO = "/Users/samy/c/touch/samysd/touchdesigner"

OPPOSITE = {
    "a": {
        "controls": (
            "hal_control_b",
            "hal_remote_sync_b",
            "hal_remote_parexec_b",
            "hal_control_ui_b",
        ),
        "vidin": "vidin_b",
        "vidout": "vidout_b",
        "preview": "null1_b",
    },
    "b": {
        "controls": (
            "hal_control",
            "hal_remote_sync",
            "hal_remote_parexec",
            "hal_control_ui",
        ),
        "vidin": "vidin",
        "vidout": "vidout",
        "preview": "null1",
    },
}


def _run(script_name: str) -> None:
    path = f"{REPO}/{script_name}"
    header = f'INSTANCE = "{INSTANCE}"\n'
    exec(compile(header + open(path, encoding="utf-8").read(), path, "exec"))


def _clone_comp(src_path: str, dst_name: str):
    dst = op(f"/project1/{dst_name}")
    if dst is not None:
        return dst
    src = op(src_path)
    if src is None:
        raise RuntimeError(f"Missing template COMP {src_path}. Create it first.")
    cloned = op("/project1").copy(src)
    cloned.name = dst_name
    return cloned


def _destroy(name: str) -> None:
    node = op(f"/project1/{name}")
    if node is not None:
        node.destroy()


def _apply_layout_cleanup(profile):
    if TD_LAYOUT == "dual":
        return

    other = OPPOSITE[profile.key]
    if TD_LAYOUT == f"{profile.key}_only":
        for name in other["controls"]:
            _destroy(name)
        for name in (other["vidin"], other["vidout"], other["preview"]):
            _destroy(name)
        print(f"TD_LAYOUT={TD_LAYOUT}: removed opposite instance comps from this project.")
        return

    raise ValueError(f"Unknown TD_LAYOUT {TD_LAYOUT!r}. Use dual, a_only, or b_only.")


import sys

if REPO not in sys.path:
    sys.path.insert(0, REPO)
from instances import get_instance
from td_layout import apply_layout, apply_project_layout

profile = get_instance(INSTANCE)

if INSTANCE == "a" and op("/project1/vidin") is None:
    template = op("/project1/vidin_b")
    if template is not None:
        op("/project1").copy(template).name = "vidin"
        print("Cloned /project1/vidin_b -> /project1/vidin")

if INSTANCE == "a" and op("/project1/vidout") is None:
    template = op("/project1/vidout_b")
    if template is not None:
        op("/project1").copy(template).name = "vidout"
        print("Cloned /project1/vidout_b -> /project1/vidout")

if INSTANCE != "a":
    _clone_comp("/project1/vidin", f"vidin{profile.suffix}")
    _clone_comp("/project1/vidout", f"vidout{profile.suffix}")

_apply_layout_cleanup(profile)

_run("build_hal_control.py")
_run("build_ndi_video_path.py")
_run("build_vidout_combine.py")
_run("build_hal_control_ui.py")

placed = apply_layout(profile) + apply_project_layout()
print(
    f"Instance {profile.label} ready (TD_LAYOUT={TD_LAYOUT}):\n"
    f"  NDI send:    {profile.ndi_out}\n"
    f"  NDI return:  {profile.ndi_in}\n"
    f"  Control:     {profile.hal_control} -> {profile.stream_id} :{profile.daydream_port}\n"
    f"  UI:          {profile.hal_control_ui}\n"
    f"  Preview:     {profile.preview}\n"
    f"  HAL screen:  {profile.screen_session}\n"
    f"  restored:    {placed} node positions from network_layout.py\n"
    f"  Rule: only ONE TD project may control {profile.stream_id} on :{profile.daydream_port}"
)

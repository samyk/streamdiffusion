"""
Configure NDI video path for the hal bridge inside vidin/vidout comps.

Run in TD textport:

    exec(open("/Users/samy/c/touch/samysd/touchdesigner/build_ndi_video_path.py", encoding="utf-8").read())

Expected chain (per instance):

  Send (vidin):
    [your source] -> ndiout1 -> hal

  Return (vidout):
    hal -> ndiin2 -> combine/in1 -> out1 -> /project1/null1[_b]
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

PROJECT = op("/project1")
VIDIN_PATH = profile.vidin
VIDOUT_PATH = profile.vidout
HAL_HOST = profile.hal_host
NDI_OUT_NAME = profile.ndi_out
NDI_IN_NAME = profile.ndi_in
PREVIEW_NAME = profile.preview.split("/")[-1]


def _ensure(parent, name, op_type):
    node = parent.op(name)
    if node is None:
        node = parent.create(op_type, name)
    return node


def _connect_if_free(target, source):
    if target.inputs and target.inputs[0] is not None:
        return False
    target.inputConnectors[0].connect(source)
    return True


def _reconnect_ndi_in(ndiin):
    """Bump NDI In after hal bridge restarts (avoids stale last-frame connections)."""
    ndiin.par.active = False
    ndiin.cook(force=True)
    ndiin.par.active = True
    ndiin.cook(force=True)


def _resolve_stream_label(ndi_name: str) -> str:
    if "(" in ndi_name and ndi_name.endswith(")"):
        return ndi_name.rsplit("(", 1)[-1][:-1].strip()
    return ndi_name.strip()


def _deactivate_duplicate_ndi_senders(active_vidin_path: str, stream_name: str) -> None:
    """Only one ndiout in this project should publish a given stream name."""
    for comp in op("/project1").children:
        if comp.OPType != "baseCOMP" or not comp.name.startswith("vidin"):
            continue
        if comp.path == active_vidin_path:
            continue
        ndi = comp.op("ndiout1")
        if ndi is None:
            continue
        if _resolve_stream_label(ndi.par.name.eval()) != stream_name:
            continue
        if ndi.par.active:
            ndi.par.active = False
            print(f"Deactivated duplicate NDI sender: {ndi.path} ({stream_name})")


def _bind_ndi_in_exact(ndiin, stream_name: str, host_ip: str) -> None:
    """Select an NDI In source by exact stream label (not substring)."""
    ndiin.par.extraips = host_ip
    ndiin.par.bandwidth = "high"
    ndiin.par.active = True
    ndiin.cook(force=True)

    menu_names = list(ndiin.par.name.menuNames) if ndiin.par.name.menuNames else []
    exact = None
    for candidate in menu_names:
        if _resolve_stream_label(candidate) == stream_name:
            exact = candidate
            break

    ndiin.par.name = exact if exact else stream_name
    _reconnect_ndi_in(ndiin)


vidin = op(VIDIN_PATH)
vidout = op(VIDOUT_PATH)
if vidin is None or vidout is None:
    raise RuntimeError(
        f"Missing {VIDIN_PATH} and/or {VIDOUT_PATH}. "
        "Run build_instance.py (clones vidin/vidout for instance B)."
    )

# --- Send path (vidin) ---
ndiout = _ensure(vidin, "ndiout1", "ndioutTOP")
ndiout.par.active = True
ndiout.par.name = NDI_OUT_NAME
_deactivate_duplicate_ndi_senders(VIDIN_PATH, NDI_OUT_NAME)

if not ndiout.inputs or ndiout.inputs[0] is None:
    flip = vidin.op("webcam_flip")
    cam = vidin.op("base2")
    if cam is None:
        cam = vidin.op("webcam_in")
    if flip is not None and cam is not None:
        _connect_if_free(flip, cam.op("out1") if cam.OPType == "baseCOMP" else cam)
        _connect_if_free(ndiout, flip)
        send_source = f"{cam.path} -> {flip.path} -> {ndiout.path}"
    elif cam is not None:
        _connect_if_free(ndiout, cam)
        send_source = f"{cam.path} -> {ndiout.path}"
    else:
        send_source = f"(wire a source into {ndiout.path})"
else:
    send_source = f"{ndiout.inputs[0].path} -> {ndiout.path}"

vidin_out = _ensure(vidin, "out1", "outTOP")
if not vidin_out.inputs or vidin_out.inputs[0] is None:
    source = ndiout.inputs[0] if ndiout.inputs and ndiout.inputs[0] else ndiout
    vidin_out.inputConnectors[0].connect(source)

# --- Return path (vidout) ---
ndiin_hal = _ensure(vidout, "ndiin2", "ndiinTOP")
_bind_ndi_in_exact(ndiin_hal, NDI_IN_NAME, HAL_HOST)

vidout_in = _ensure(vidout, "in1", "inTOP")
_connect_if_free(vidout_in, vidin_out)

combine = vidout.op("combine")
if combine is not None:
    combine_in = combine.op("in1")
    if combine_in is not None:
        combine_in.inputConnectors[0].connect(ndiin_hal)

vidout_out = _ensure(vidout, "out1", "outTOP")
preview = PROJECT.op(PREVIEW_NAME)
if preview is None:
    preview = PROJECT.create("nullTOP", PREVIEW_NAME)
if not preview.inputs or preview.inputs[0] is None:
    preview.inputConnectors[0].connect(vidout_out)

display_sink = vidout_out.path
if combine is not None:
    combine_out = combine.op("out1")
    if combine_out is not None:
        display_sink = f"{combine.path} -> {vidout_out.path}"

print(
    f"NDI wired (instance {profile.label}):\n"
    f"  send:    {send_source} ({NDI_OUT_NAME})\n"
    f"  return:  {ndiin_hal.path} ({ndiin_hal.par.name}, extraips={HAL_HOST})\n"
    f"  display: {display_sink} -> {preview.path}"
)

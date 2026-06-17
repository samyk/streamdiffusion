"""
Configure vidout/combine layout:
  in1 (hal) fullscreen | in2 (source) PiP flush bottom-right | text above bottom

Run in TD textport:

    exec(open("/Users/samy/c/touch/samysd/touchdesigner/build_vidout_combine.py", encoding="utf-8").read())
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

VIDOUT_PATH = profile.vidout
COMBINE_PATH = profile.combine
EXEC_PATH = profile.layout_exec
LAYOUT_PATH = f"{REPO}/vidout_combine_layout.py"
HAL_CONTROL_PATH = profile.hal_control
VIDIN_OUT_PATH = f"{profile.vidin}/out1"

VIDOUT = op(VIDOUT_PATH)
COMBINE = op(COMBINE_PATH)

if VIDOUT is None or COMBINE is None:
    raise RuntimeError(f"Missing {VIDOUT_PATH} or {COMBINE_PATH}")


def _recreate(comp, name: str, op_type: str):
    old = comp.op(name)
    if old:
        old.destroy()
    return comp.create(op_type, name)


def _place(node, x: int, y: int) -> None:
    node.nodeX = x
    node.nodeY = y


in1 = COMBINE.op("in1")
in2 = COMBINE.op("in2")
if in1 is None or in2 is None:
    raise RuntimeError("combine needs in1, in2")

_place(in1, -600, 200)
_place(in2, -400, 200)

ndiin2 = VIDOUT.op("ndiin2")
vidout_in1 = VIDOUT.op("in1")
vidin_out = op(VIDIN_OUT_PATH)
if ndiin2 is not None:
    COMBINE.inputConnectors[0].connect(ndiin2)
if vidout_in1 is not None and vidin_out is not None:
    vidout_in1.inputConnectors[0].connect(vidin_out)
    COMBINE.inputConnectors[1].connect(vidout_in1)

for old_name in ("pip_xform",):
    old = COMBINE.op(old_name)
    if old:
        old.destroy()

pip_resize = COMBINE.op("pip_resize")
if pip_resize is None or pip_resize.OPType != "fitTOP":
    if pip_resize:
        pip_resize.destroy()
    pip_resize = COMBINE.create("fitTOP", "pip_resize")
pip_resize.inputConnectors[0].connect(in2)
pip_resize.par.outputresolution = "custom"
pip_resize.par.fit = "fitbest"
pip_resize.par.resolutionw = 192
pip_resize.par.resolutionh = 108
_place(pip_resize, -500, 200)

pip_place = COMBINE.op("pip_place")
if pip_place is None:
    pip_place = COMBINE.create("transformTOP", "pip_place")
pip_place.inputConnectors[0].connect(pip_resize)
pip_place.par.bgcolora = 0
pip_place.par.px = 0
pip_place.par.py = 0
pip_place.par.punit = "fraction"
pip_place.par.tunit = "pixels"
pip_place.par.sx = 1
pip_place.par.sy = 1
_place(pip_place, -350, 200)

comp1 = _recreate(COMBINE, "comp1", "compositeTOP")
comp1.inputConnectors[0].connect(pip_resize)
comp1.inputConnectors[1].connect(in1)
comp1.par.operand = "over"
comp1.par.size = "input2"
comp1.par.prefit = "nativeres"
comp1.par.justifyh = "right"
comp1.par.justifyv = "bottom"
comp1.par.sx = 1
comp1.par.sy = 1
comp1.par.tx = 0
comp1.par.ty = 0
_place(comp1, -200, 200)

prompt = "prompt"
ctrl = op(HAL_CONTROL_PATH)
if ctrl and hasattr(ctrl.par, "Prompt"):
    prompt = ctrl.par.Prompt.eval().strip() or prompt

for index, name in enumerate(("text2", "text3", "text4")):
    _recreate(COMBINE, name, "textTOP")
    text_top = COMBINE.op(name)
    text_top.par.text = prompt
    text_top.par.alignx = "center"
    text_top.par.aligny = "bottom"
    text_top.par.outputresolution = "custom"
    text_top.par.wordwrap = True
    text_top.par.fontautosize = "off"
    text_top.par.fontcolorr = 1
    text_top.par.fontcolorg = 1
    text_top.par.fontcolorb = 1
    text_top.par.keepfontratio = True
    _place(text_top, 0 + index * 40, 0)

text3 = COMBINE.op("text3")

comp2 = _recreate(COMBINE, "comp2", "compositeTOP")
comp2.inputConnectors[0].connect(text3)
comp2.inputConnectors[1].connect(comp1)
comp2.par.operand = "over"
comp2.par.size = "input2"
comp2.par.prefit = "nativeres"
comp2.par.justifyh = "center"
comp2.par.justifyv = "bottom"
comp2.par.px = 0.5
comp2.par.py = 0
comp2.par.tunit = "pixels"
comp2.par.sx = 1
comp2.par.sy = 1
_place(comp2, 200, 200)

finull = COMBINE.op("finull")
if finull is None:
    finull = COMBINE.create("nullTOP", "finull")
finull.inputConnectors[0].connect(comp2)
_place(finull, 400, 200)

text_fps = COMBINE.op("text_fps")
if text_fps is None or text_fps.OPType != "textTOP":
    if text_fps:
        text_fps.destroy()
    text_fps = COMBINE.create("textTOP", "text_fps")
text_fps.par.text = "NDI -- fps"
text_fps.par.alignx = "left"
text_fps.par.aligny = "bottom"
text_fps.par.outputresolution = "custom"
text_fps.par.fontautosize = "off"
text_fps.par.fontcolorr = 1
text_fps.par.fontcolorg = 1
text_fps.par.fontcolorb = 1
text_fps.par.keepfontratio = True
_place(text_fps, 320, 200)

comp_hud = COMBINE.op("comp_hud")
if comp_hud is None or comp_hud.OPType != "compositeTOP":
    if comp_hud:
        comp_hud.destroy()
    comp_hud = COMBINE.create("compositeTOP", "comp_hud")
comp_hud.inputConnectors[0].connect(text_fps)
comp_hud.inputConnectors[1].connect(finull)
comp_hud.par.operand = "over"
comp_hud.par.size = "input2"
comp_hud.par.prefit = "nativeres"
comp_hud.par.justifyh = "left"
comp_hud.par.justifyv = "bottom"
comp_hud.par.tx = 12
comp_hud.par.ty = 12
_place(comp_hud, 500, 200)

out1 = COMBINE.op("out1")
if out1 is not None:
    out1.inputConnectors[0].connect(comp_hud)
    _place(out1, 600, 200)

layout_dat = VIDOUT.op("combine_layout")
if layout_dat is None:
    layout_dat = VIDOUT.create("textDAT", "combine_layout")
layout_dat.text = open(LAYOUT_PATH, encoding="utf-8").read()
_place(layout_dat, -600, 0)

old_exec = op(EXEC_PATH)
if old_exec and old_exec.parent != VIDOUT:
    old_exec.destroy()

exec_dat = VIDOUT.op("combine_layout_exec")
if exec_dat is None:
    exec_dat = VIDOUT.create("executeDAT", "combine_layout_exec")
exec_dat.par.active = True
exec_dat.par.start = True
exec_dat.par.framestart = True
exec_dat.par.file = layout_dat.path
exec_dat.text = (
    "def onFrameStart(frame):\n"
    f"    op('{layout_dat.path}').module.update_layout('{VIDOUT_PATH}')\n"
    "    return\n"
)
_place(exec_dat, -400, 0)

combine_parexec = VIDOUT.op("combine_parexec")
if combine_parexec:
    combine_parexec.destroy()

vidout_ui = op(f"/project1/vidout_ui{profile.suffix}")
if vidout_ui:
    vidout_ui.destroy()

sync_dat = op(profile.sync_dat)
if sync_dat is not None:
    sync_body = open(f"{REPO}/hal_remote_sync.py", encoding="utf-8").read()
    sync_body = sync_body.replace(
        'CONTROL_PATH = "/project1/hal_control"',
        f'CONTROL_PATH = "{profile.hal_control}"',
    )
    sync_body = sync_body.replace("REMOTE_PORT = 8780", f"REMOTE_PORT = {profile.daydream_port}")
    sync_body = sync_body.replace(
        'STREAM_ID = "remote-1"',
        f'STREAM_ID = "{profile.stream_id}"',
    )
    sync_dat.text = sync_body

op(layout_dat.path).module.update_layout(VIDOUT_PATH)

print(f"vidout/combine layout updated (instance {profile.label}).")

"""
Configure vidout/combine layout (expression-driven, no per-frame layout script):
  in1 (hal) fullscreen | in2 (source) PiP flush bottom-right | prompt + FPS HUD

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
from td_layout import apply_layout, place

profile = get_instance(INSTANCE)

VIDOUT_PATH = profile.vidout
COMBINE_PATH = profile.combine
SYNC_PATH = profile.sync_dat
HAL_CONTROL_PATH = profile.hal_control
VIDIN_OUT_PATH = f"{profile.vidin}/out1"

INFO_PATH = f"{VIDOUT_PATH}/info1"
INFO_FPS_CHANNEL = "receive_fps"
TEXT_FONT_BASE = 28.0
FPS_FONT_BASE = 22.0
FPS_MARGIN_PX = 16

VIDOUT = op(VIDOUT_PATH)
COMBINE = op(COMBINE_PATH)

if VIDOUT is None or COMBINE is None:
    raise RuntimeError(f"Missing {VIDOUT_PATH} or {COMBINE_PATH}")


def _recreate(comp, name: str, op_type: str):
    old = comp.op(name)
    if old:
        old.destroy()
    return comp.create(op_type, name)


def _hal() -> str:
    return f"op('{HAL_CONTROL_PATH}')"


def _set_expr(target, par_name: str, expression: str) -> None:
    par = getattr(target.par, par_name)
    par.expr = expression


def _set_const(target, par_name: str, value) -> None:
    par = getattr(target.par, par_name)
    par.expr = ""
    par.val = value


in1 = COMBINE.op("in1")
in2 = COMBINE.op("in2")
if in1 is None or in2 is None:
    raise RuntimeError("combine needs in1, in2")

place(in1)
place(in2)

ndiin2 = VIDOUT.op("ndiin2")
vidout_in1 = VIDOUT.op("in1")
vidin_out = op(VIDIN_OUT_PATH)
if ndiin2 is not None:
    COMBINE.inputConnectors[0].connect(ndiin2)
if vidout_in1 is not None and vidin_out is not None:
    vidout_in1.inputConnectors[0].connect(vidin_out)
    COMBINE.inputConnectors[1].connect(vidout_in1)

for old_name in ("pip_xform", "text2", "text3", "text4"):
    old = COMBINE.op(old_name)
    if old:
        old.destroy()

for old_name in ("combine_layout", "combine_layout_exec", "fps_timer", "fps_chop_exec"):
    old = VIDOUT.op(old_name)
    if old:
        old.destroy()

info1 = VIDOUT.op("info1")
if info1 is None or info1.OPType != "infoCHOP":
    if info1:
        info1.destroy()
    info1 = VIDOUT.create("infoCHOP", "info1")
if ndiin2 is not None:
    info1.par.op = ndiin2
if hasattr(info1.par, "infotype"):
    _set_const(info1, "infotype", "all")
if hasattr(info1.par, "iscope"):
    _set_const(info1, "iscope", "*")
_set_const(info1, "passive", False)
place(info1)

pip_resize = COMBINE.op("pip_resize")
if pip_resize is None or pip_resize.OPType != "fitTOP":
    if pip_resize:
        pip_resize.destroy()
    pip_resize = COMBINE.create("fitTOP", "pip_resize")
pip_resize.inputConnectors[0].connect(in2)
_set_const(pip_resize, "outputresolution", "custom")
_set_const(pip_resize, "fit", "fitbest")
_set_expr(pip_resize, "resolutionw", f"max(1, int(op('in2').width * {_hal()}.par.Pipscale))")
_set_expr(pip_resize, "resolutionh", f"max(1, int(op('in2').height * {_hal()}.par.Pipscale))")
place(pip_resize)

pip_place = COMBINE.op("pip_place")
if pip_place is None:
    pip_place = COMBINE.create("transformTOP", "pip_place")
pip_place.inputConnectors[0].connect(pip_resize)
_set_const(pip_place, "bgcolora", 0)
_set_const(pip_place, "px", 0)
_set_const(pip_place, "py", 0)
_set_const(pip_place, "punit", "fraction")
_set_const(pip_place, "tunit", "pixels")
_set_const(pip_place, "sx", 1)
_set_const(pip_place, "sy", 1)
place(pip_place)

comp1 = _recreate(COMBINE, "comp1", "compositeTOP")
comp1.inputConnectors[0].connect(pip_resize)
comp1.inputConnectors[1].connect(in1)
_set_const(comp1, "operand", "over")
_set_const(comp1, "size", "input2")
_set_const(comp1, "prefit", "nativeres")
_set_const(comp1, "justifyh", "right")
_set_const(comp1, "justifyv", "bottom")
_set_const(comp1, "sx", 1)
_set_const(comp1, "sy", 1)
_set_const(comp1, "tx", 0)
_set_const(comp1, "ty", 0)
place(comp1)

text_prompt = COMBINE.op("text_prompt")
if text_prompt is None or text_prompt.OPType != "textTOP":
    if text_prompt:
        text_prompt.destroy()
    text_prompt = COMBINE.create("textTOP", "text_prompt")
_set_expr(text_prompt, "text", f"{_hal()}.par.Prompt")
_set_const(text_prompt, "alignx", "center")
_set_const(text_prompt, "aligny", "bottom")
_set_const(text_prompt, "outputresolution", "custom")
_set_const(text_prompt, "wordwrap", True)
_set_const(text_prompt, "fontautosize", "off")
_set_const(text_prompt, "fontsizexunit", "points")
_set_const(text_prompt, "fontsizeyunit", "points")
_set_const(text_prompt, "linespacingunit", "points")
_set_const(text_prompt, "keepfontratio", True)
_set_expr(text_prompt, "fontsizex", f"max(12, {TEXT_FONT_BASE} * {_hal()}.par.Textscale)")
_set_expr(text_prompt, "fontsizey", f"max(12, {TEXT_FONT_BASE} * {_hal()}.par.Textscale)")
_set_expr(text_prompt, "linespacing", f"max(2, {TEXT_FONT_BASE} * {_hal()}.par.Textscale * 0.12)")
_set_expr(text_prompt, "resolutionw", "max(64, int(op('in1').width))")
_set_expr(
    text_prompt,
    "resolutionh",
    f"max(48, int({TEXT_FONT_BASE} * {_hal()}.par.Textscale * 3.5))",
)
_set_const(text_prompt, "fontcolorr", 1)
_set_const(text_prompt, "fontcolorg", 1)
_set_const(text_prompt, "fontcolorb", 1)
place(text_prompt)

comp2 = _recreate(COMBINE, "comp2", "compositeTOP")
comp2.inputConnectors[0].connect(text_prompt)
comp2.inputConnectors[1].connect(comp1)
_set_const(comp2, "operand", "over")
_set_const(comp2, "size", "input2")
_set_const(comp2, "prefit", "nativeres")
_set_const(comp2, "justifyh", "center")
_set_const(comp2, "justifyv", "bottom")
_set_const(comp2, "px", 0.5)
_set_const(comp2, "py", 0)
_set_const(comp2, "tunit", "pixels")
_set_const(comp2, "sx", 1)
_set_const(comp2, "sy", 1)
_set_expr(comp2, "ty", f"{_hal()}.par.Textlift")
place(comp2)

finull = COMBINE.op("finull")
if finull is None:
    finull = COMBINE.create("nullTOP", "finull")
finull.inputConnectors[0].connect(comp2)
place(finull)

text_fps = COMBINE.op("text_fps")
if text_fps is None or text_fps.OPType != "textTOP":
    if text_fps:
        text_fps.destroy()
    text_fps = COMBINE.create("textTOP", "text_fps")
_set_expr(
    text_fps,
    "text",
    f"str(int(max(0, op('../info1')['{INFO_FPS_CHANNEL}'])))",
)
_set_const(text_fps, "alignx", "left")
_set_const(text_fps, "aligny", "center")
_set_const(text_fps, "outputresolution", "custom")
_set_const(text_fps, "fontautosize", "off")
_set_const(text_fps, "fontsizexunit", "points")
_set_const(text_fps, "fontsizeyunit", "points")
_set_const(text_fps, "keepfontratio", True)
_set_const(text_fps, "fontsizex", FPS_FONT_BASE)
_set_const(text_fps, "fontsizey", FPS_FONT_BASE)
_set_const(text_fps, "resolutionw", max(48, int(FPS_FONT_BASE * 2.2)))
_set_const(text_fps, "resolutionh", max(40, int(FPS_FONT_BASE * 1.6)))
_set_const(text_fps, "fontcolorr", 1)
_set_const(text_fps, "fontcolorg", 1)
_set_const(text_fps, "fontcolorb", 1)
place(text_fps)

fps_xform = COMBINE.op("fps_xform")
if fps_xform is None or fps_xform.OPType != "transformTOP":
    if fps_xform:
        fps_xform.destroy()
    fps_xform = COMBINE.create("transformTOP", "fps_xform")
fps_xform.inputConnectors[0].connect(text_fps)
_set_const(fps_xform, "bgcolora", 0)
_set_const(fps_xform, "px", FPS_MARGIN_PX)
_set_const(fps_xform, "py", FPS_MARGIN_PX)
_set_const(fps_xform, "punit", "pixels")
_set_const(fps_xform, "tunit", "pixels")
_set_const(fps_xform, "sx", 1)
_set_const(fps_xform, "sy", 1)
place(fps_xform)

comp_hud = COMBINE.op("comp_hud")
if comp_hud is None or comp_hud.OPType != "compositeTOP":
    if comp_hud:
        comp_hud.destroy()
    comp_hud = COMBINE.create("compositeTOP", "comp_hud")
comp_hud.inputConnectors[0].connect(fps_xform)
comp_hud.inputConnectors[1].connect(finull)
_set_const(comp_hud, "operand", "over")
_set_const(comp_hud, "size", "input2")
_set_const(comp_hud, "prefit", "nativeres")
_set_const(comp_hud, "justifyh", "left")
_set_const(comp_hud, "justifyv", "bottom")
_set_const(comp_hud, "tx", 0)
_set_const(comp_hud, "ty", 0)
place(comp_hud)

out1 = COMBINE.op("out1")
if out1 is not None:
    out1.inputConnectors[0].connect(comp_hud)
    place(out1)

combine_parexec = VIDOUT.op("combine_parexec")
if combine_parexec:
    combine_parexec.destroy()

vidout_ui = op(f"/project1/vidout_ui{profile.suffix}")
if vidout_ui:
    vidout_ui.destroy()

sync_dat = op(SYNC_PATH)
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

placed = apply_layout(profile)
print(f"vidout/combine wired with expressions (instance {profile.label}).")
print(f"  restored {placed} saved node positions from network_layout.py")
print("  No per-frame layout script — edit Text TOP colors directly in the network.")
print(f"  FPS: op('../info1')['{INFO_FPS_CHANNEL}'] → text_fps")

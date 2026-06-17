"""
Floating panel for hal_control (HAL + display layout).

    exec(open("/Users/samy/c/touch/samysd/touchdesigner/build_hal_control_ui.py", encoding="utf-8").read())
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

CONTROL = profile.hal_control
UI_PATH = profile.hal_control_ui
VIDOUT = profile.vidout


def _ensure_display_params(ctrl):
    if hasattr(ctrl.par, "Pipscale"):
        return

    pg = ctrl.appendCustomPage("Display")
    pip = pg.appendFloat("Pipscale", label="PiP Size")
    pip.normMin = 0.05
    pip.normMax = 1.0
    text = pg.appendFloat("Textscale", label="Text Size")
    text.normMin = 0.25
    text.normMax = 4.0
    lift = pg.appendFloat("Textlift", label="Text Lift (px)")
    lift.normMin = 0
    lift.normMax = 200

    ctrl.par.Pipscale = 0.25
    ctrl.par.Textscale = 1.0
    ctrl.par.Textlift = 36


existing = op(UI_PATH)
if existing:
    existing.destroy()

vidout_ui = op(f"/project1/vidout_ui{profile.suffix}")
if vidout_ui:
    vidout_ui.destroy()

ctrl = op(CONTROL)
if ctrl is None:
    raise RuntimeError(f"Missing {CONTROL}. Run build_instance.py first.")

_ensure_display_params(ctrl)

ui = op("/project1").create("parameterCOMP", f"hal_control_ui{profile.suffix}")
ui.nodeX = 500 if profile.key == "a" else 1100
ui.nodeY = 200
ui.par.op = ctrl.path
ui.par.header = True
ui.par.custom = True
ui.par.builtin = False
ui.par.pagescope = "*"
ui.par.parscope = (
    "Remotehost Remoteport Streamid Pushall "
    "Prompt Negativeprompt Prompt2 Prompt2weight Promptinterp "
    "Denoise Step2 Step3 Step4 "
    "Preset Qualitymode Modelid Sdmode Acceleration "
    "Width Height Guidance Delta Seed Usetinyvae Vaeid "
    "Lora1path Lora1scale Lora2path Lora2scale Lora3path Lora3scale "
    "Pipscale Textscale Textlift "
    "Filterthreshold Filterskip Pausestream "
    "Ipimagepath Ipscale Controlnetmodel Controlnetscale"
)
ui.par.allowexpand = True
ui.par.inputeditor = True
ui.par.labels = True
ui.par.separators = True
ui.par.compress = True
ui.par.pvscrollbar = True
ui.par.phscrollbar = False
ui.par.crop = False
ui.par.fit = False
ui.par.spacing = 2
ui.par.marginl = 8
ui.par.marginr = 8
ui.par.margint = 6
ui.par.marginb = 6
ui.par.w = 520
ui.par.h = 720
ui.par.hmode = "fixed"
ui.par.vmode = "fixed"
ui.par.fixedaspect = False
ui.par.bgcolorr = 0.1
ui.par.bgcolorg = 0.11
ui.par.bgcolorb = 0.14
ui.par.bgalpha = 1
ui.par.display = True
ui.par.enable = True

layout = op(f"{VIDOUT}/combine_layout")
if layout is not None:
    layout.module.update_layout(VIDOUT)

print(f"Created {ui.path} — instance {profile.label} controls.")

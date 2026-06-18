"""
HAL control UI — parameterCOMP in /project1 (click node → see all controls).

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

import importlib

import hal_control_ui_build

importlib.reload(hal_control_ui_build)
from hal_control_ui_build import build_hal_control_ui
import importlib

import td_hal_defs

importlib.reload(td_hal_defs)
from td_hal_defs import HAL_CONTROL_PAGE
from instances import get_instance
from td_layout import apply_layout

profile = get_instance(INSTANCE)
CONTROL = profile.hal_control


def _hal_page(ctrl):
    for page in ctrl.customPages:
        if page.name == HAL_CONTROL_PAGE:
            return page
    for page in ctrl.customPages:
        if page.name == "Display":
            return page
    return ctrl.appendCustomPage(HAL_CONTROL_PAGE)


def _ensure_display_params(ctrl):
    if hasattr(ctrl.par, "Pipscale"):
        return
    pg = _hal_page(ctrl)
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


vidout_ui = op(f"/project1/vidout_ui{profile.suffix}")
if vidout_ui:
    vidout_ui.destroy()

ctrl = op(CONTROL)
if ctrl is None:
    raise RuntimeError(f"Missing {CONTROL}. Run build_instance.py first.")

_ensure_display_params(ctrl)

ui = build_hal_control_ui(op("/project1"), profile, ctrl)
if isinstance(ui, tuple):
    ui = ui[0]
apply_layout(profile)

print(f"Created {ui.path} (instance {profile.label}) → {CONTROL}")
print("  Click hal_control_ui in /project1 to select; scrollable HAL panel shows all sections.")

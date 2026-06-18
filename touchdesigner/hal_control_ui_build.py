"""Build hal_control_ui as a single parameterCOMP in /project1."""

from __future__ import annotations

import td

from td_hal_defs import HAL_CONTROL_PAGE
from td_layout import place


def configure_hal_control_ui(ui, control_path: str, *, height: int = 960) -> None:
    ui.par.op = control_path
    ui.par.header = True
    ui.par.custom = True
    ui.par.builtin = False
    ui.par.pagescope = HAL_CONTROL_PAGE
    ui.par.parscope = "*"
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
    ui.par.h = height
    ui.par.hmode = "fixed"
    ui.par.vmode = "fixed"
    ui.par.fixedaspect = False
    ui.par.bgcolorr = 0.1
    ui.par.bgcolorg = 0.11
    ui.par.bgcolorb = 0.14
    ui.par.bgalpha = 1
    ui.par.display = True
    ui.par.enable = True


def build_hal_control_ui(parent, profile, ctrl, *, panel_height: int = 960):
    """Return parameterCOMP at /project1/hal_control_ui[_b]."""
    existing = td.op(profile.hal_control_ui)
    if existing is not None:
        existing.destroy()

    ui = parent.create("parameterCOMP", f"hal_control_ui{profile.suffix}")
    configure_hal_control_ui(ui, ctrl.path, height=panel_height)
    place(ui)
    return ui

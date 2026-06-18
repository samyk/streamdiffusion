"""
Add missing HAL remote params to an existing hal_control COMP (no destroy).

Run in TouchDesigner (or via touchmcp execute_python_script):

    exec(open("/Users/samy/c/touch/samysd/touchdesigner/upgrade_hal_control.py", encoding="utf-8").read())
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

import hal_control_defs

importlib.reload(hal_control_defs)

from hal_control_defs import (
    HAL_CONTROL_PAGE,
    HAL_SYNC_PARSCOPE,
    TD_HAL_DEFAULTS,
    ATTENTION_BACKEND_LABELS,
    ATTENTION_BACKEND_NAMES,
    PRESET_MENU_LABELS,
    PRESET_MENU_NAMES,
    SEGMENTATION_BACKEND_LABELS,
    SEGMENTATION_BACKEND_NAMES,
    UPSCALE_FACTOR_LABELS,
    UPSCALE_FACTOR_NAMES,
    UPSCALE_MAXINE_QUALITY_LABELS,
    UPSCALE_MAXINE_QUALITY_NAMES,
    UPSCALE_METHOD_LABELS,
    UPSCALE_METHOD_NAMES,
    apply_td_hal_defaults,
)
from instances import get_instance
from td_layout import apply_layout

profile = get_instance(INSTANCE)
SYNC_PATH = f"{REPO}/hal_remote_sync.py"
ctrl = op(profile.hal_control)
if ctrl is None:
    raise RuntimeError(f"Missing {profile.hal_control}. Run build_hal_control.py first.")


def _page(name: str):
    for page in ctrl.customPages:
        if page.name == HAL_CONTROL_PAGE:
            return page
    for page in ctrl.customPages:
        if page.name == name:
            return page
    return ctrl.appendCustomPage(name)


def _ensure_toggle(page, name: str, label: str, default: bool) -> None:
    if hasattr(ctrl.par, name):
        return
    page.appendToggle(name, label=label)
    setattr(ctrl.par, name, default)


def _ensure_menu(page, name: str, label: str, names, labels, default: str) -> None:
    if hasattr(ctrl.par, name):
        par = getattr(ctrl.par, name)
        par.menuNames = names
        par.menuLabels = labels
        return
    page.appendMenu(name, label=label)
    par = getattr(ctrl.par, name)
    par.menuNames = names
    par.menuLabels = labels
    setattr(ctrl.par, name, default)


def _ensure_str(page, name: str, label: str, default: str = "") -> None:
    if hasattr(ctrl.par, name):
        return
    page.appendStr(name, label=label)
    setattr(ctrl.par, name, default)


def _ensure_int(page, name: str, label: str, default: int, norm_min: int, norm_max: int) -> None:
    if hasattr(ctrl.par, name):
        return
    par = page.appendInt(name, label=label)
    par.normMin = norm_min
    par.normMax = norm_max
    setattr(ctrl.par, name, default)


def _ensure_float(page, name: str, label: str, default: float) -> None:
    if hasattr(ctrl.par, name):
        return
    page.appendFloat(name, label=label)
    setattr(ctrl.par, name, default)


# --- Upscale (hal-only before this upgrade) ---
upscale = _page("Upscale")
_ensure_toggle(upscale, "Upscaleenabled", "Upscale Enabled", TD_HAL_DEFAULTS["Upscaleenabled"])
_ensure_menu(
    upscale,
    "Upscalefactor",
    "Upscale Factor",
    UPSCALE_FACTOR_NAMES,
    UPSCALE_FACTOR_LABELS,
    TD_HAL_DEFAULTS["Upscalefactor"],
)
_ensure_menu(
    upscale,
    "Upscalemethod",
    "Upscale Method",
    UPSCALE_METHOD_NAMES,
    UPSCALE_METHOD_LABELS,
    TD_HAL_DEFAULTS["Upscalemethod"],
)
_ensure_toggle(upscale, "Upscalehalf", "Real-ESRGAN FP16 (half)", TD_HAL_DEFAULTS["Upscalehalf"])
_ensure_menu(
    upscale,
    "Upscalemaxinequality",
    "Maxine Quality",
    UPSCALE_MAXINE_QUALITY_NAMES,
    UPSCALE_MAXINE_QUALITY_LABELS,
    TD_HAL_DEFAULTS["Upscalemaxinequality"],
)
_ensure_str(upscale, "Upscalemodel", "Custom Upscale Model (.pth)")

# --- V2V / Person segmentation ---
segment = _page("V2V / Segmentation")
_ensure_toggle(segment, "Segmentenabled", "Person Segmentation (CUDA / Maxine)", TD_HAL_DEFAULTS["Segmentenabled"])
_ensure_toggle(segment, "Persononly", "Person Only (style people, keep camera bg)", TD_HAL_DEFAULTS["Persononly"])
_ensure_toggle(segment, "Cutbackground", "Cut Background (replace bg color)", TD_HAL_DEFAULTS["Cutbackground"])
_ensure_float(segment, "Segmentfeather", "Mask Feather (px)", TD_HAL_DEFAULTS["Segmentfeather"])
_ensure_str(segment, "Backgroundcolor", "Background Color (#RRGGBB or R,G,B)", TD_HAL_DEFAULTS["Backgroundcolor"])
_ensure_menu(
    segment,
    "Segmentbackend",
    "Segmentation Backend",
    SEGMENTATION_BACKEND_NAMES,
    SEGMENTATION_BACKEND_LABELS,
    TD_HAL_DEFAULTS["Segmentbackend"],
)
if hasattr(ctrl.par, "Sdmode"):
    ctrl.par.Sdmode.menuLabels = ["img2img", "txt2img", "v2v (temporal img2img)", "passthrough"]

# --- Params that may be missing on older builds ---
quality = _page("Quality")
_ensure_int(quality, "Framebatch", "Frame Batch Count", TD_HAL_DEFAULTS["Framebatch"], 1, 8)
_ensure_toggle(
    quality,
    "Fluxtransformerengine",
    "FLUX Blackwell Transformer Engine",
    TD_HAL_DEFAULTS["Fluxtransformerengine"],
)

advanced = _page("Advanced")
_ensure_str(advanced, "Ipmodel", "IP-Adapter Model (HF id)", "h94/IP-Adapter")

display = _page("Display")
_ensure_float(display, "Pipscale", "PiP Size", TD_HAL_DEFAULTS["Pipscale"])
_ensure_float(display, "Textscale", "Text Size", TD_HAL_DEFAULTS["Textscale"])
_ensure_float(display, "Textlift", "Text Lift (px)", TD_HAL_DEFAULTS["Textlift"])

model = _page("Model")
if hasattr(ctrl.par, "Preset"):
    ctrl.par.Preset.menuNames = PRESET_MENU_NAMES
    ctrl.par.Preset.menuLabels = PRESET_MENU_LABELS
_ensure_menu(
    model,
    "Attentionbackend",
    "Attention Backend (FLUX / SD3.5 DiT)",
    ATTENTION_BACKEND_NAMES,
    ATTENTION_BACKEND_LABELS,
    TD_HAL_DEFAULTS["Attentionbackend"],
)
if hasattr(ctrl.par, "Acceleration"):
    ctrl.par.Acceleration.label = "Acceleration (SD Turbo / SDXL only)"
_ensure_toggle(
    quality,
    "Modeloptenabled",
    "ModelOpt Quant (SD3.5 / DiT, optional)",
    bool(TD_HAL_DEFAULTS["Modeloptenabled"]),
)
_ensure_str(
    quality,
    "Modeloptcheckpoint",
    "ModelOpt Checkpoint (.pt path on hal)",
    TD_HAL_DEFAULTS["Modeloptcheckpoint"],
)
if hasattr(ctrl.par, "Fluxtransformerengine"):
    ctrl.par.Fluxtransformerengine.label = "DiT/FLUX Blackwell Compile (torch.compile)"

denoise_page = _page("Denoise")
if denoise_page is None:
    denoise_page = _page("AI")
if hasattr(ctrl.par, "Denoise"):
    ctrl.par.Denoise.label = "Steps (Klein/SD3.5 1-6) / T-index 1 (Turbo 1-49)"
    if hasattr(ctrl.par.Denoise, "normMin"):
        ctrl.par.Denoise.normMin = 1
    if hasattr(ctrl.par.Denoise, "normMax"):
        ctrl.par.Denoise.normMax = 49
for name, label in (
    ("Step2", "Extra step (Klein/SD3.5) / T-index 2 (Turbo, 0=off)"),
    ("Step3", "Extra step (Klein/SD3.5) / T-index 3 (Turbo)"),
    ("Step4", "Extra step (Klein/SD3.5) / T-index 4 (Turbo)"),
):
    if hasattr(ctrl.par, name):
        getattr(ctrl.par, name).label = label
        if hasattr(getattr(ctrl.par, name), "normMax"):
            getattr(ctrl.par, name).normMax = 49

apply_td_hal_defaults(ctrl)

ctrl.par.Remotehost = profile.hal_host
ctrl.par.Remoteport = profile.daydream_port
ctrl.par.Streamid = profile.stream_id

# --- Refresh sync DAT from repo ---
sync_dat = op(profile.sync_dat)
if sync_dat is None:
    raise RuntimeError(f"Missing {profile.sync_dat}")

sync_body = open(SYNC_PATH, encoding="utf-8").read()
sync_body = sync_body.replace(
    'CONTROL_PATH = "/project1/hal_control"',
    f'CONTROL_PATH = "{profile.hal_control}"',
)
sync_body = sync_body.replace(
    'SYNC_DAT_PATH = "/project1/hal_remote_sync"',
    f'SYNC_DAT_PATH = "{profile.sync_dat}"',
)
sync_body = sync_body.replace("REMOTE_PORT = 8780", f"REMOTE_PORT = {profile.daydream_port}")
sync_body = sync_body.replace('STREAM_ID = "remote-1"', f'STREAM_ID = "{profile.stream_id}"')
sync_dat.text = sync_body

parexec = op(profile.parexec)
if parexec is not None:
    parexec.par.pars = HAL_SYNC_PARSCOPE
    parexec.par.file = sync_dat.path
    parexec.par.valuechange = True
    parexec.par.onpulse = True

# --- Rebuild UI container (panel inside /project1/hal_control_ui) ---
exec(
    compile(
        f'INSTANCE = "{INSTANCE}"\n'
        + open(f"{REPO}/build_hal_control_ui.py", encoding="utf-8").read(),
        f"{REPO}/build_hal_control_ui.py",
        "exec",
    )
)

# --- Ensure HUD nodes + refresh layout wiring ---
exec(
    compile(
        f'INSTANCE = "{INSTANCE}"\n' + open(f"{REPO}/build_vidout_combine.py", encoding="utf-8").read(),
        f"{REPO}/build_vidout_combine.py",
        "exec",
    )
)

sync_dat.module.push_params(force=True)
placed = apply_layout(profile)
ui = op(profile.hal_control_ui)
print(f"Upgraded {ctrl.path} + {ui.path} (instance {profile.label}).")
if ui:
    print(f"  UI: click {ui.path} in /project1")
print(f"  restored {placed} saved node positions from network_layout.py")

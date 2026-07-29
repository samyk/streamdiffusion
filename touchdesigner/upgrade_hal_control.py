"""
Add missing HAL remote params to an existing hal_control COMP (no destroy).

Run in TouchDesigner (or via touchmcp execute_python_script):

    exec(open("/Users/samy/c/touch/samysd/touchdesigner/upgrade_hal_control.py", encoding="utf-8").read())
"""

try:
    INSTANCE
except NameError:
    INSTANCE = "a"

# Set True to force UI + vidout rebuild (slow; don't paste repeatedly with this on).
FORCE_FULL = False

import sys

REPO = "/Users/samy/c/touch/samysd/touchdesigner"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import importlib

import td_hal_defs

importlib.reload(td_hal_defs)

from td_hal_defs import (
    HAL_CONTROL_PAGE,
    HAL_SYNC_PARSCOPE,
    TD_HAL_DEFAULTS,
    ATTENTION_BACKEND_LABELS,
    ATTENTION_BACKEND_NAMES,
    PRESET_MENU_LABELS,
    PRESET_MENU_NAMES,
    SCENE_IDLE_MODE_LABELS,
    SCENE_IDLE_MODE_NAMES,
    SEGMENTATION_BACKEND_LABELS,
    SEGMENTATION_BACKEND_NAMES,
    UPSCALE_FACTOR_LABELS,
    UPSCALE_FACTOR_NAMES,
    UPSCALE_MAXINE_QUALITY_LABELS,
    UPSCALE_MAXINE_QUALITY_NAMES,
    UPSCALE_METHOD_LABELS,
    UPSCALE_METHOD_NAMES,
)
from instances import get_instance
from td_layout import apply_layout

profile = get_instance(INSTANCE)
SYNC_PATH = f"{REPO}/hal_remote_sync.py"
ctrl = op(profile.hal_control)
if ctrl is None:
    raise RuntimeError(f"Missing {profile.hal_control}. Run build_hal_control.py first.")

PROMPT_WIDGET = "/project1/prompt/textprompt"
PROMPT_EXPR = f"op('{PROMPT_WIDGET}').par.text"


def _repair_prompt_expr() -> None:
    """Fix broken Prompt bindings to the prompt textCOMP (common typo: .par.textpp)."""
    if not hasattr(ctrl.par, "Prompt"):
        return
    widget = op(PROMPT_WIDGET)
    if widget is None:
        return
    par = ctrl.par.Prompt
    expr = par.expr or ""
    if "textpp" in expr or (expr and "textprompt" in expr and expr.strip() != PROMPT_EXPR):
        par.expr = PROMPT_EXPR
    ctrl.cook(force=True)
    err = ctrl.errors()
    if err:
        raise RuntimeError(f"{ctrl.path} Prompt repair failed: {err}")

# Second run is params-only: skip UI/vidout rebuild (those caused textport freezes).
_LIGHT = (
    not FORCE_FULL
    and hasattr(ctrl.par, "Sceneidle")
    and hasattr(ctrl.par, "Scenechangethreshold")
)

parexec = op(profile.parexec)
sync_dat = op(profile.sync_dat)
if sync_dat is None:
    raise RuntimeError(f"Missing {profile.sync_dat}")

if parexec is not None:
    parexec.par.active = False
    parexec.par.valuechange = False

try:
    sync_dat.module.set_sync_muted(True)
except Exception:
    pass


def _hal_page():
    for page in ctrl.customPages:
        if page.name == HAL_CONTROL_PAGE:
            return page
    return ctrl.appendCustomPage(HAL_CONTROL_PAGE)


def _page(name: str):
    # Single-page HAL layout: all sections live on HAL_CONTROL_PAGE.
    return _hal_page()


def _is_menu(par) -> bool:
    try:
        return bool(par.isMenu)
    except Exception:
        return False


def _ensure_toggle(page, name: str, label: str, default: bool) -> None:
    if hasattr(ctrl.par, name):
        return
    page.appendToggle(name, label=label)
    getattr(ctrl.par, name).val = int(bool(default))


def _ensure_menu(page, name: str, label: str, names, labels, default: str) -> None:
    if hasattr(ctrl.par, name):
        par = getattr(ctrl.par, name)
        if not _is_menu(par):
            print(f"[upgrade_hal_control] skip {name}: exists but is not a menu")
            return
        par.menuNames = names
        par.menuLabels = labels
        par.label = label
        return
    page.appendMenu(name, label=label)
    par = getattr(ctrl.par, name)
    par.menuNames = names
    par.menuLabels = labels
    par.val = default


def _ensure_str(page, name: str, label: str, default: str = "") -> None:
    if hasattr(ctrl.par, name):
        return
    page.appendStr(name, label=label)
    getattr(ctrl.par, name).val = default


def _ensure_int(page, name: str, label: str, default: int, norm_min: int, norm_max: int) -> None:
    if hasattr(ctrl.par, name):
        return
    page.appendInt(name, label=label)
    par = getattr(ctrl.par, name)
    par.normMin = norm_min
    par.normMax = norm_max
    par.val = default


def _ensure_float(
    page,
    name: str,
    label: str,
    default: float,
    *,
    norm_min: float | None = None,
    norm_max: float | None = None,
) -> None:
    if hasattr(ctrl.par, name):
        par = getattr(ctrl.par, name)
        if norm_min is not None:
            par.normMin = norm_min
        if norm_max is not None:
            par.normMax = norm_max
        return
    page.appendFloat(name, label=label)
    par = getattr(ctrl.par, name)
    if norm_min is not None:
        par.normMin = norm_min
    if norm_max is not None:
        par.normMax = norm_max
    par.val = default


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
_ensure_menu(
    advanced,
    "Sceneidle",
    "Scene Idle (skip static input)",
    SCENE_IDLE_MODE_NAMES,
    SCENE_IDLE_MODE_LABELS,
    TD_HAL_DEFAULTS["Sceneidle"],
)
_ensure_float(
    advanced,
    "Scenechangethreshold",
    "Min Scene Change (0.02-0.08 noisy cam)",
    float(TD_HAL_DEFAULTS["Scenechangethreshold"]),
    norm_min=0.0,
    norm_max=0.25,
)
_ensure_toggle(
    advanced,
    "Sceneidlendi",
    "Also gate NDI send from TouchDesigner",
    bool(TD_HAL_DEFAULTS["Sceneidlendi"]),
)
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

ctrl.par.Remotehost.val = profile.hal_host
ctrl.par.Remoteport.val = profile.daydream_port
ctrl.par.Streamid.val = profile.stream_id

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
if sync_dat.text != sync_body:
    sync_dat.text = sync_body

if parexec is not None:
    parexec.par.pars = HAL_SYNC_PARSCOPE
    parexec.par.file = sync_dat.path
    parexec.par.onpulse = True

if not _LIGHT:
    try:
        exec(
            compile(
                f'INSTANCE = "{INSTANCE}"\n'
                + open(f"{REPO}/build_hal_control_ui.py", encoding="utf-8").read(),
                f"{REPO}/build_hal_control_ui.py",
                "exec",
            )
        )
    except Exception as exc:
        print(f"[upgrade_hal_control] UI rebuild skipped: {exc}")

    try:
        exec(
            compile(
                f'INSTANCE = "{INSTANCE}"\n'
                + open(f"{REPO}/build_vidout_combine.py", encoding="utf-8").read(),
                f"{REPO}/build_vidout_combine.py",
                "exec",
            )
        )
    except Exception as exc:
        print(f"[upgrade_hal_control] vidout combine skipped: {exc}")
else:
    print("[upgrade_hal_control] light upgrade (params + sync only; UI/vidout unchanged)")

try:
    sync_dat.module.cancel_pending_push()
    sync_dat.module.set_sync_muted(False)
    sync_dat.module.push_params(force=True)
except Exception as exc:
    print(f"[upgrade_hal_control] push_params failed: {exc}")

_repair_prompt_expr()

if parexec is not None:
    parexec.par.valuechange = True
    parexec.par.active = True

placed = apply_layout(profile)
ui = op(profile.hal_control_ui)
mode = "light" if _LIGHT else "full"
print(f"Upgraded {ctrl.path} ({mode}, instance {profile.label}).")
if ui:
    print(f"  UI: click {ui.path} in /project1")
print(f"  restored {placed} saved node positions from network_layout.py")

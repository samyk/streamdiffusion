"""
Build the HAL remote control COMP (replaces StreamDiffusionTD UI).

Run in TouchDesigner:

    exec(open("/Users/samy/c/touch/samysd/touchdesigner/build_hal_control.py", encoding="utf-8").read())
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
    SEGMENTATION_BACKEND_LABELS,
    SEGMENTATION_BACKEND_NAMES,
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
from td_layout import apply_layout, place

profile = get_instance(INSTANCE)
ROOT = profile.hal_control
SYNC_PATH = f"{REPO}/hal_remote_sync.py"
REMOTE_HOST = profile.hal_host
REMOTE_PORT = profile.daydream_port
STREAM_ID = profile.stream_id


def _delete(path):
    node = op(path)
    if node:
        node.destroy()


_delete(ROOT)
_delete(profile.sync_dat)
_delete(profile.parexec)

parent = op("/project1")
ctrl = parent.create("baseCOMP", f"hal_control{profile.suffix}")
place(ctrl)

pg = ctrl.appendCustomPage(HAL_CONTROL_PAGE)


def _section(label: str) -> None:
    slug = "".join(c.lower() for c in label if c.isalnum())
    pg.appendHeader(f"Hdr{slug}", label=label)


def _start_section(par_name: str) -> None:
    getattr(ctrl.par, par_name).startSection = True


# --- Connection ---
_section("Connection")
pg.appendStr("Remotehost", label="HAL Host")
ctrl.par.Remotehost = REMOTE_HOST
_start_section("Remotehost")
pg.appendInt("Remoteport", label="API Port")
ctrl.par.Remoteport = REMOTE_PORT
pg.appendStr("Streamid", label="Stream ID")
ctrl.par.Streamid = STREAM_ID
pg.appendPulse("Pushall", label="Push All")

# --- Prompt ---
_section("Prompt")
pg.appendStr("Prompt", label="Prompt")
ctrl.par.Prompt = TD_HAL_DEFAULTS["Prompt"]
_start_section("Prompt")
pg.appendStr("Negativeprompt", label="Negative Prompt")
ctrl.par.Negativeprompt = TD_HAL_DEFAULTS["Negativeprompt"]
pg.appendStr("Prompt2", label="Prompt 2")
ctrl.par.Prompt2 = ""
pg.appendFloat("Prompt2weight", label="Prompt 2 Weight")
ctrl.par.Prompt2weight = 0.0
pg.appendMenu("Promptinterp", label="Prompt Interpolation")
ctrl.par.Promptinterp.menuNames = ["average", "slerp"]
ctrl.par.Promptinterp.menuLabels = ["Average", "Slerp (avg fallback)"]
ctrl.par.Promptinterp = "average"

# --- Denoise ---
_section("Denoise")
denoise = pg.appendInt("Denoise", label="Steps (Klein/SD3.5 1-6) / T-index 1 (Turbo 1-49)")
denoise.normMin = 1
denoise.normMax = 49
ctrl.par.Denoise = TD_HAL_DEFAULTS["Denoise"]
_start_section("Denoise")
for name, label in (
    ("Step2", "Extra step (Klein/SD3.5) / T-index 2 (Turbo, 0=off)"),
    ("Step3", "Extra step (Klein/SD3.5) / T-index 3 (Turbo)"),
    ("Step4", "Extra step (Klein/SD3.5) / T-index 4 (Turbo)"),
):
    step = pg.appendInt(name, label=label)
    step.normMin = 0
    step.normMax = 49
    setattr(ctrl.par, name, TD_HAL_DEFAULTS[name])

# --- Model ---
_section("Model")
pg.appendMenu("Preset", label="Preset")
ctrl.par.Preset.menuNames = PRESET_MENU_NAMES
ctrl.par.Preset.menuLabels = PRESET_MENU_LABELS
ctrl.par.Preset = TD_HAL_DEFAULTS["Preset"]
_start_section("Preset")
pg.appendStr("Modelid", label="Custom Model (HF id or .safetensors)")
ctrl.par.Modelid = ""
pg.appendMenu("Sdmode", label="Mode")
ctrl.par.Sdmode.menuNames = ["img2img", "txt2img", "v2v", "passthrough"]
ctrl.par.Sdmode.menuLabels = ["img2img", "txt2img", "v2v (temporal img2img)", "passthrough"]
ctrl.par.Sdmode = TD_HAL_DEFAULTS["Sdmode"]
pg.appendMenu("Acceleration", label="Acceleration (SD Turbo / SDXL only)")
ctrl.par.Acceleration.menuNames = ["none", "xformers", "tensorrt"]
ctrl.par.Acceleration.menuLabels = ["none", "xformers", "tensorrt (default)"]
ctrl.par.Acceleration = TD_HAL_DEFAULTS["Acceleration"]
pg.appendMenu("Attentionbackend", label="Attention Backend (FLUX / SD3.5 DiT)")
ctrl.par.Attentionbackend.menuNames = ATTENTION_BACKEND_NAMES
ctrl.par.Attentionbackend.menuLabels = ATTENTION_BACKEND_LABELS
ctrl.par.Attentionbackend = TD_HAL_DEFAULTS["Attentionbackend"]

# --- Quality ---
_section("Quality")
pg.appendInt("Width", label="Width")
ctrl.par.Width = TD_HAL_DEFAULTS["Width"]
_start_section("Width")
pg.appendInt("Height", label="Height")
ctrl.par.Height = TD_HAL_DEFAULTS["Height"]
framebatch = pg.appendInt("Framebatch", label="Frame Batch (turbo v2v: keep 1; multi-step = temporal)")
framebatch.normMin = 1
framebatch.normMax = 8
ctrl.par.Framebatch = TD_HAL_DEFAULTS["Framebatch"]
pg.appendToggle("Fluxtransformerengine", label="DiT/FLUX Blackwell Compile (torch.compile)")
ctrl.par.Fluxtransformerengine = TD_HAL_DEFAULTS["Fluxtransformerengine"]
pg.appendToggle("Modeloptenabled", label="ModelOpt Quant (SD3.5 / DiT, optional)")
ctrl.par.Modeloptenabled = TD_HAL_DEFAULTS["Modeloptenabled"]
pg.appendStr("Modeloptcheckpoint", label="ModelOpt Checkpoint (.pt path on hal)")
ctrl.par.Modeloptcheckpoint = TD_HAL_DEFAULTS["Modeloptcheckpoint"]
pg.appendFloat("Guidance", label="Guidance Scale")
ctrl.par.Guidance = TD_HAL_DEFAULTS["Guidance"]
pg.appendFloat("Delta", label="Delta")
ctrl.par.Delta = TD_HAL_DEFAULTS["Delta"]
pg.appendInt("Seed", label="Seed")
ctrl.par.Seed = TD_HAL_DEFAULTS["Seed"]
pg.appendToggle("Usetinyvae", label="Use Tiny VAE (off = sharper, slower)")
ctrl.par.Usetinyvae = TD_HAL_DEFAULTS["Usetinyvae"]
pg.appendStr("Vaeid", label="Tiny VAE id (blank = auto taesd/taesdxl)")
ctrl.par.Vaeid = ""

# --- LoRA ---
_section("LoRA")
pg.appendStr("Lora1path", label="LoRA 1 Path")
ctrl.par.Lora1path = ""
_start_section("Lora1path")
pg.appendFloat("Lora1scale", label="LoRA 1 Scale")
ctrl.par.Lora1scale = 1.0
pg.appendStr("Lora2path", label="LoRA 2 Path")
ctrl.par.Lora2path = ""
pg.appendFloat("Lora2scale", label="LoRA 2 Scale")
ctrl.par.Lora2scale = 1.0
pg.appendStr("Lora3path", label="LoRA 3 Path")
ctrl.par.Lora3path = ""
pg.appendFloat("Lora3scale", label="LoRA 3 Scale")
ctrl.par.Lora3scale = 1.0

# --- Upscale ---
_section("Upscale")
pg.appendToggle("Upscaleenabled", label="Upscale Enabled")
ctrl.par.Upscaleenabled = TD_HAL_DEFAULTS["Upscaleenabled"]
_start_section("Upscaleenabled")
pg.appendMenu("Upscalefactor", label="Upscale Factor")
ctrl.par.Upscalefactor.menuNames = UPSCALE_FACTOR_NAMES
ctrl.par.Upscalefactor.menuLabels = UPSCALE_FACTOR_LABELS
ctrl.par.Upscalefactor = TD_HAL_DEFAULTS["Upscalefactor"]
pg.appendMenu("Upscalemethod", label="Upscale Method")
ctrl.par.Upscalemethod.menuNames = UPSCALE_METHOD_NAMES
ctrl.par.Upscalemethod.menuLabels = UPSCALE_METHOD_LABELS
ctrl.par.Upscalemethod = TD_HAL_DEFAULTS["Upscalemethod"]
pg.appendToggle("Upscalehalf", label="Real-ESRGAN FP16 (half)")
ctrl.par.Upscalehalf = TD_HAL_DEFAULTS["Upscalehalf"]
pg.appendMenu("Upscalemaxinequality", label="Maxine Quality")
ctrl.par.Upscalemaxinequality.menuNames = UPSCALE_MAXINE_QUALITY_NAMES
ctrl.par.Upscalemaxinequality.menuLabels = UPSCALE_MAXINE_QUALITY_LABELS
ctrl.par.Upscalemaxinequality = TD_HAL_DEFAULTS["Upscalemaxinequality"]
pg.appendStr("Upscalemodel", label="Custom Upscale Model (.pth)")
ctrl.par.Upscalemodel = ""

# --- V2V + Person Segmentation ---
_section("V2V / Segmentation")
pg.appendToggle("Segmentenabled", label="Person Segmentation (CUDA / Maxine)")
ctrl.par.Segmentenabled = TD_HAL_DEFAULTS["Segmentenabled"]
_start_section("Segmentenabled")
pg.appendToggle("Persononly", label="Person Only (style people, keep camera bg)")
ctrl.par.Persononly = TD_HAL_DEFAULTS["Persononly"]
pg.appendToggle("Cutbackground", label="Cut Background (replace bg color)")
ctrl.par.Cutbackground = TD_HAL_DEFAULTS["Cutbackground"]
feather = pg.appendFloat("Segmentfeather", label="Mask Feather (px)")
feather.normMin = 0
feather.normMax = 20
ctrl.par.Segmentfeather = TD_HAL_DEFAULTS["Segmentfeather"]
pg.appendStr("Backgroundcolor", label="Background Color (#RRGGBB or R,G,B)")
ctrl.par.Backgroundcolor = TD_HAL_DEFAULTS["Backgroundcolor"]
pg.appendMenu("Segmentbackend", label="Segmentation Backend")
ctrl.par.Segmentbackend.menuNames = SEGMENTATION_BACKEND_NAMES
ctrl.par.Segmentbackend.menuLabels = SEGMENTATION_BACKEND_LABELS
ctrl.par.Segmentbackend = TD_HAL_DEFAULTS["Segmentbackend"]

# --- Display ---
_section("Display")
pip = pg.appendFloat("Pipscale", label="PiP Size")
pip.normMin = 0.05
pip.normMax = 1.0
ctrl.par.Pipscale = TD_HAL_DEFAULTS["Pipscale"]
_start_section("Pipscale")
text = pg.appendFloat("Textscale", label="Text Size")
text.normMin = 0.25
text.normMax = 4.0
ctrl.par.Textscale = TD_HAL_DEFAULTS["Textscale"]
lift = pg.appendFloat("Textlift", label="Text Lift (px)")
lift.normMin = 0
lift.normMax = 200
ctrl.par.Textlift = TD_HAL_DEFAULTS["Textlift"]

# --- Advanced ---
_section("Advanced")
flt = pg.appendFloat("Filterthreshold", label="Similar Filter (0=off)")
flt.normMin = 0
flt.normMax = 0.99
ctrl.par.Filterthreshold = TD_HAL_DEFAULTS["Filterthreshold"]
_start_section("Filterthreshold")
pg.appendInt("Filterskip", label="Similar Filter Max Skip")
ctrl.par.Filterskip = TD_HAL_DEFAULTS["Filterskip"]
pg.appendToggle("Pausestream", label="Pause / Passthrough")
ctrl.par.Pausestream = False
pg.appendStr("Ipimagepath", label="IP-Adapter Image Path (TRT only)")
ctrl.par.Ipimagepath = ""
pg.appendFloat("Ipscale", label="IP-Adapter Scale")
ctrl.par.Ipscale = 0.5
pg.appendStr("Ipmodel", label="IP-Adapter Model (HF id)")
ctrl.par.Ipmodel = "h94/IP-Adapter"
pg.appendStr("Controlnetmodel", label="ControlNet Model (TRT only)")
ctrl.par.Controlnetmodel = ""
pg.appendFloat("Controlnetscale", label="ControlNet Scale")
ctrl.par.Controlnetscale = 0.5

readme = ctrl.create("textDAT", "readme")
place(readme)
readme.text = (
    f"HAL remote control for StreamDiffusion bridge (instance {profile.label}).\n"
    f"NDI send: {profile.ndi_out} | return: {profile.ndi_in}\n"
    f"Control: REST PATCH to {REMOTE_HOST}:{REMOTE_PORT}/v1/streams/{STREAM_ID}\n"
    f"Default: {TD_HAL_DEFAULTS['Preset']} @ {TD_HAL_DEFAULTS['Width']}x{TD_HAL_DEFAULTS['Height']}, "
    f"denoise {TD_HAL_DEFAULTS['Denoise']}, tiny VAE on, 2x Maxine upscale.\n"
)

sync_dat = parent.create("textDAT", f"hal_remote_sync{profile.suffix}")
place(sync_dat)
sync_body = open(SYNC_PATH, encoding="utf-8").read()
sync_body = sync_body.replace(
    'CONTROL_PATH = "/project1/hal_control"',
    f'CONTROL_PATH = "{profile.hal_control}"',
)
sync_body = sync_body.replace(
    'SYNC_DAT_PATH = "/project1/hal_remote_sync"',
    f'SYNC_DAT_PATH = "{profile.sync_dat}"',
)
sync_body = sync_body.replace("REMOTE_PORT = 8780", f"REMOTE_PORT = {REMOTE_PORT}")
sync_body = sync_body.replace('STREAM_ID = "remote-1"', f'STREAM_ID = "{STREAM_ID}"')
sync_dat.text = sync_body

parexec = parent.create("parameterexecuteDAT", f"hal_remote_parexec{profile.suffix}")
place(parexec)
parexec.par.op = ctrl.path
parexec.par.file = sync_dat.path
parexec.text = f'''\
def onValueChange(par, prev):
    try:
        op("{sync_dat.path}").module.onValueChange(par, prev)
    except Exception as exc:
        print(f"[hal_remote_parexec] {{exc}}")
    return

def onPulse(par):
    try:
        op("{sync_dat.path}").module.onPulse(par)
    except Exception as exc:
        print(f"[hal_remote_parexec] {{exc}}")
    return
'''
parexec.par.executeloc = "here"
parexec.par.fromop = "/project1"
parexec.par.valuechange = True
parexec.par.onpulse = True
parexec.par.custom = True
parexec.par.builtin = False
parexec.par.pars = HAL_SYNC_PARSCOPE

# Disable legacy paths
for path in (
    "/project1/streamdiffusion_bridge/websocket",
    "/project1/sdtd_remote_parexec",
):
    node = op(path)
    if node and hasattr(node.par, "active"):
        node.par.active = False

sync_dat.module.push_params(force=True)
placed = apply_layout(profile)
print(f"HAL control wired (instance {profile.label}).")
print(f"  restored {placed} saved node positions from network_layout.py")
print(f"  COMP: {ctrl.path}")
print(f"  API:  http://{REMOTE_HOST}:{REMOTE_PORT}/v1/streams/{STREAM_ID}")
print(f"  Run build_hal_control_ui.py or build_instance.py for the UI container.")

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
from hal_control_defs import (
    TD_HAL_DEFAULTS,
    UPSCALE_FACTOR_LABELS,
    UPSCALE_FACTOR_NAMES,
    UPSCALE_MAXINE_QUALITY_LABELS,
    UPSCALE_MAXINE_QUALITY_NAMES,
    UPSCALE_METHOD_LABELS,
    UPSCALE_METHOD_NAMES,
    apply_td_hal_defaults,
)
from instances import get_instance

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
ctrl.nodeX = -200 if profile.key == "a" else 200
ctrl.nodeY = 200

# --- Connection ---
pg = ctrl.appendCustomPage("Connection")
pg.appendStr("Remotehost", label="HAL Host")
ctrl.par.Remotehost = REMOTE_HOST
pg.appendInt("Remoteport", label="API Port")
ctrl.par.Remoteport = REMOTE_PORT
pg.appendStr("Streamid", label="Stream ID")
ctrl.par.Streamid = STREAM_ID
pg.appendPulse("Pushall", label="Push All")

# --- Prompt ---
pg = ctrl.appendCustomPage("Prompt")
pg.appendStr("Prompt", label="Prompt")
ctrl.par.Prompt = "cybernetic botanical glass sculpture"
pg.appendStr("Negativeprompt", label="Negative Prompt")
ctrl.par.Negativeprompt = "blurry, low detail, artifacts, watermark"
pg.appendStr("Prompt2", label="Prompt 2")
ctrl.par.Prompt2 = ""
pg.appendFloat("Prompt2weight", label="Prompt 2 Weight")
ctrl.par.Prompt2weight = 0.0
pg.appendMenu("Promptinterp", label="Prompt Interpolation")
ctrl.par.Promptinterp.menuNames = ["average", "slerp"]
ctrl.par.Promptinterp.menuLabels = ["Average", "Slerp (avg fallback)"]
ctrl.par.Promptinterp = "average"

# --- Denoise ---
pg = ctrl.appendCustomPage("Denoise")
denoise = pg.appendInt("Denoise", label="Step 1 (1-49)")
denoise.normMin = 1
denoise.normMax = 49
ctrl.par.Denoise = TD_HAL_DEFAULTS["Denoise"]
for name, label in (("Step2", "Step 2 (0=off)"), ("Step3", "Step 3"), ("Step4", "Step 4")):
    step = pg.appendInt(name, label=label)
    step.normMin = 0
    step.normMax = 49
    setattr(ctrl.par, name, TD_HAL_DEFAULTS[name])

# --- Model ---
pg = ctrl.appendCustomPage("Model")
pg.appendMenu("Preset", label="Preset")
ctrl.par.Preset.menuNames = [
    "sdxl_turbo_fast",
    "sdxl_turbo_quality",
    "sd_turbo_fast",
    "sd_turbo_quality",
    "lcm_lora_style",
    "flux2_klein_fast",
    "flux2_klein_quality",
    "flux2_klein_9b",
    "passthrough",
]
ctrl.par.Preset.menuLabels = [
    "SDXL Turbo Fast",
    "SDXL Turbo Quality",
    "SD Turbo Fast",
    "SD Turbo Quality",
    "SD1.5 LCM LoRA",
    "FLUX.2 Klein Fast (4B)",
    "FLUX.2 Klein Quality (4B)",
    "FLUX.2 Klein 9B",
    "Passthrough",
]
ctrl.par.Preset = TD_HAL_DEFAULTS["Preset"]
pg.appendMenu("Qualitymode", label="Quality Mode")
ctrl.par.Qualitymode.menuNames = ["fast", "quality"]
ctrl.par.Qualitymode.menuLabels = ["Fast", "Quality"]
ctrl.par.Qualitymode = TD_HAL_DEFAULTS["Qualitymode"]
pg.appendStr("Modelid", label="Custom Model (HF id or .safetensors)")
ctrl.par.Modelid = ""
pg.appendMenu("Sdmode", label="Mode")
ctrl.par.Sdmode.menuNames = ["img2img", "txt2img", "v2v", "passthrough"]
ctrl.par.Sdmode.menuLabels = ["img2img", "txt2img", "v2v (TRT only)", "passthrough"]
ctrl.par.Sdmode = "img2img"
pg.appendMenu("Acceleration", label="Acceleration")
ctrl.par.Acceleration.menuNames = ["none", "xformers", "tensorrt"]
ctrl.par.Acceleration.menuLabels = ["none (Blackwell)", "xformers", "tensorrt"]
ctrl.par.Acceleration = "none"

# --- Quality ---
pg = ctrl.appendCustomPage("Quality")
pg.appendInt("Width", label="Width")
ctrl.par.Width = TD_HAL_DEFAULTS["Width"]
pg.appendInt("Height", label="Height")
ctrl.par.Height = TD_HAL_DEFAULTS["Height"]
framebatch = pg.appendInt("Framebatch", label="Frame Batch Count")
framebatch.normMin = 1
framebatch.normMax = 8
ctrl.par.Framebatch = TD_HAL_DEFAULTS["Framebatch"]
pg.appendToggle("Fluxtransformerengine", label="FLUX Blackwell Transformer Engine")
ctrl.par.Fluxtransformerengine = TD_HAL_DEFAULTS["Fluxtransformerengine"]
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
pg = ctrl.appendCustomPage("LoRA")
for index in (1, 2, 3):
    pg.appendStr(f"Lora{index}path", label=f"LoRA {index} Path")
    pg.appendFloat(f"Lora{index}scale", label=f"LoRA {index} Scale")
    setattr(ctrl.par, f"Lora{index}path", "")
    setattr(ctrl.par, f"Lora{index}scale", 1.0)

# --- Upscale ---
pg = ctrl.appendCustomPage("Upscale")
pg.appendToggle("Upscaleenabled", label="Upscale Enabled")
ctrl.par.Upscaleenabled = TD_HAL_DEFAULTS["Upscaleenabled"]
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

# --- Display ---
pg = ctrl.appendCustomPage("Display")
pip = pg.appendFloat("Pipscale", label="PiP Size")
pip.normMin = 0.05
pip.normMax = 1.0
ctrl.par.Pipscale = TD_HAL_DEFAULTS["Pipscale"]
text = pg.appendFloat("Textscale", label="Text Size")
text.normMin = 0.25
text.normMax = 4.0
ctrl.par.Textscale = TD_HAL_DEFAULTS["Textscale"]
lift = pg.appendFloat("Textlift", label="Text Lift (px)")
lift.normMin = 0
lift.normMax = 200
ctrl.par.Textlift = TD_HAL_DEFAULTS["Textlift"]

# --- Advanced ---
pg = ctrl.appendCustomPage("Advanced")
flt = pg.appendFloat("Filterthreshold", label="Similar Filter (0=off)")
flt.normMin = 0
flt.normMax = 0.99
ctrl.par.Filterthreshold = 0
pg.appendInt("Filterskip", label="Similar Filter Max Skip")
ctrl.par.Filterskip = 10
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
readme.text = (
    f"HAL remote control for StreamDiffusion bridge (instance {profile.label}).\n"
    f"NDI send: {profile.ndi_out} | return: {profile.ndi_in}\n"
    f"Control: REST PATCH to {REMOTE_HOST}:{REMOTE_PORT}/v1/streams/{STREAM_ID}\n"
    "SDXL-Turbo default. IP-Adapter / ControlNet / V2V need TensorRT (not on Blackwell yet).\n"
)

sync_dat = parent.create("textDAT", f"hal_remote_sync{profile.suffix}")
sync_body = open(SYNC_PATH, encoding="utf-8").read()
sync_body = sync_body.replace(
    'CONTROL_PATH = "/project1/hal_control"',
    f'CONTROL_PATH = "{profile.hal_control}"',
)
sync_body = sync_body.replace("REMOTE_PORT = 8780", f"REMOTE_PORT = {REMOTE_PORT}")
sync_body = sync_body.replace('STREAM_ID = "remote-1"', f'STREAM_ID = "{STREAM_ID}"')
sync_dat.text = sync_body

parexec = parent.create("parameterexecuteDAT", f"hal_remote_parexec{profile.suffix}")
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
parexec.par.pars = "*"

# Disable legacy paths
for path in (
    "/project1/streamdiffusion_bridge/websocket",
    "/project1/sdtd_remote_parexec",
):
    node = op(path)
    if node and hasattr(node.par, "active"):
        node.par.active = False

sync_dat.module.push_params(force=True)
print(f"HAL control wired (instance {profile.label}).")
print(f"  COMP: {ctrl.path}")
print(f"  API:  http://{REMOTE_HOST}:{REMOTE_PORT}/v1/streams/{STREAM_ID}")
print(f"  Run build_hal_control_ui.py or build_instance.py for the panel.")

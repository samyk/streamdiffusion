"""Shared HAL control parameter menus (build + live upgrade)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from streamdiffusion_td_bridge.defaults import (  # noqa: E402
    DAYDREAM_PORT,
    HAL_BRIDGE_LAUNCH_DEFAULTS,
    HAL_HOST,
    STREAM_ID,
)

UPSCALE_FACTOR_NAMES = ["1", "2", "4"]
UPSCALE_FACTOR_LABELS = ["1× (off when disabled)", "2×", "4×"]

UPSCALE_METHOD_NAMES = ["maxine-vsr", "realesrgan", "bicubic"]
UPSCALE_METHOD_LABELS = ["Maxine VSR (NVIDIA)", "Real-ESRGAN", "Bicubic (fast)"]

UPSCALE_MAXINE_QUALITY_NAMES = [
    "low",
    "medium",
    "high",
    "ultra",
    "highbitrate_low",
    "highbitrate_medium",
    "highbitrate_high",
    "highbitrate_ultra",
]
UPSCALE_MAXINE_QUALITY_LABELS = [
    "Low",
    "Medium",
    "High",
    "Ultra",
    "High bitrate — Low",
    "High bitrate — Medium",
    "High bitrate — High",
    "High bitrate — Ultra",
]

FLUX_KLEIN_PRESETS = {
    "flux2_klein_fast",
    "flux2_klein_quality",
    "flux2_klein_9b",
}

KLEIN_MAX_STEPS = 6

# Instance A locations (see instances.py for B):
#   TD control:  /project1/hal_control
#   TD sync DAT: /project1/hal_remote_sync
#   TD vidout:   /project1/vidout/combine
#   hal API:     http://{HAL_HOST}:{DAYDREAM_PORT}/v1/streams/{STREAM_ID}
#   NDI → hal:   td_streamdiffusion_in
#   NDI ← hal:   streamdiffusion_out

PRESET_DENOISE_STEPS = {
    "sdxl_turbo_fast": [35],
    "sdxl_turbo_quality": [32, 45],
    "sd_turbo_fast": [28],
    "sd_turbo_quality": [32, 45],
    "lcm_lora_style": [0, 16, 32, 45],
    "flux2_klein_fast": [1, 2, 3, 4],
    "flux2_klein_quality": [1, 2, 3, 4],
    "flux2_klein_9b": [1, 2, 3, 4],
    "passthrough": [35],
}


def denoise_steps_for_preset(preset: str) -> list[int]:
    return list(PRESET_DENOISE_STEPS.get(preset, [35]))


def klein_steps_from_denoise(
    denoise: int,
    *,
    step2: int = 0,
    step3: int = 0,
    step4: int = 0,
) -> list[int]:
    """Klein only uses len(t_index_list) as num_inference_steps; values are placeholders."""
    count = max(1, min(KLEIN_MAX_STEPS, int(denoise)))
    for value in (step2, step3, step4):
        if int(value) > 0:
            count += 1
    count = min(KLEIN_MAX_STEPS, count)
    return list(range(1, count + 1))


def turbo_steps_from_denoise(
    denoise: int,
    *,
    step2: int = 0,
    step3: int = 0,
    step4: int = 0,
) -> list[int]:
    steps = [max(1, min(49, int(denoise)))]
    for value in (step2, step3, step4):
        if int(value) > 0:
            steps.append(max(1, min(49, int(value))))
    return steps


def denoise_steps_from_control(ctrl, preset: str | None = None) -> list[int]:
    preset = preset or (ctrl.par.Preset.eval() if hasattr(ctrl.par, "Preset") else "")
    denoise = int(ctrl.par.Denoise)
    step2 = int(ctrl.par.Step2)
    step3 = int(ctrl.par.Step3)
    step4 = int(ctrl.par.Step4)
    if preset in FLUX_KLEIN_PRESETS:
        return klein_steps_from_denoise(denoise, step2=step2, step3=step3, step4=step4)
    steps = turbo_steps_from_denoise(denoise, step2=step2, step3=step3, step4=step4)
    if not steps:
        return denoise_steps_for_preset(preset)
    if preset != "lcm_lora_style" and min(steps) < 15:
        return [max(15, int(v)) for v in steps]
    return steps


def denoise_steps_match_preset(preset: str, steps: list[int]) -> bool:
    if not steps:
        return False
    if preset in FLUX_KLEIN_PRESETS:
        return 1 <= len(steps) <= KLEIN_MAX_STEPS
    return min(steps) >= 15


_bridge = HAL_BRIDGE_LAUNCH_DEFAULTS
_denoise = _bridge["t_index_list"][0] if len(_bridge["t_index_list"]) == 1 else _bridge["t_index_list"][0]
_step2 = _bridge["t_index_list"][1] if len(_bridge["t_index_list"]) > 1 else 0

# TouchDesigner hal_control defaults (synced from live instance A).
TD_HAL_DEFAULTS = {
    "Remotehost": HAL_HOST,
    "Remoteport": DAYDREAM_PORT,
    "Streamid": STREAM_ID,
    "Prompt": _bridge["prompt"],
    "Negativeprompt": _bridge["negative_prompt"],
    "Prompt2": "",
    "Prompt2weight": 0.0,
    "Promptinterp": "average",
    "Preset": _bridge["preset"],
    "Width": _bridge["width"],
    "Height": _bridge["height"],
    "Denoise": _denoise,
    "Step2": _step2,
    "Step3": 0,
    "Step4": 0,
    "Sdmode": _bridge["sdmode"],
    "Acceleration": _bridge["acceleration"],
    "Framebatch": _bridge["frame_buffer_size"],
    "Fluxtransformerengine": _bridge["flux_transformer_engine"],
    "Guidance": _bridge["guidance_scale"],
    "Delta": _bridge["delta"],
    "Seed": _bridge["seed"],
    "Usetinyvae": _bridge["use_tiny_vae"],
    "Upscaleenabled": _bridge["upscale_enabled"],
    "Upscalefactor": str(_bridge["upscale_factor"]),
    "Upscalemethod": _bridge["upscale_method"],
    "Upscalehalf": _bridge["upscale_half"],
    "Upscalemaxinequality": _bridge["upscale_maxine_quality"],
    "Pipscale": 0.25,
    "Textscale": 1.0,
    "Textlift": 36,
    "Filterthreshold": _bridge["similar_image_filter_threshold"],
    "Filterskip": _bridge["similar_image_filter_max_skip_frame"],
    "Pausestream": False,
}


_CONNECTION_KEYS = frozenset({"Remotehost", "Remoteport", "Streamid"})


def apply_td_hal_defaults(ctrl, *, include_connection: bool = False) -> None:
    """Apply TD_HAL_DEFAULTS to an existing hal_control COMP."""
    for name, value in TD_HAL_DEFAULTS.items():
        if not include_connection and name in _CONNECTION_KEYS:
            continue
        if not hasattr(ctrl.par, name):
            continue
        getattr(ctrl.par, name).val = value


# Full floating-panel parscope (order = UI layout).
HAL_CONTROL_PARSCOPE = (
    "Remotehost Remoteport Streamid Pushall "
    "Prompt Negativeprompt Prompt2 Prompt2weight Promptinterp "
    "Denoise Step2 Step3 Step4 "
    "Preset Modelid Sdmode Acceleration "
    "Width Height Framebatch Fluxtransformerengine Guidance Delta Seed "
    "Usetinyvae Vaeid "
    "Lora1path Lora1scale Lora2path Lora2scale Lora3path Lora3scale "
    "Upscaleenabled Upscalefactor Upscalemethod Upscalehalf "
    "Upscalemaxinequality Upscalemodel "
    "Pipscale Textscale Textlift "
    "Filterthreshold Filterskip Pausestream "
    "Ipimagepath Ipscale Ipmodel Controlnetmodel Controlnetscale"
)

# Parameter execute DAT: sync to hal on change (connection pars excluded).
HAL_SYNC_PARSCOPE = (
    "Prompt Negativeprompt Prompt2 Prompt2weight Promptinterp "
    "Denoise Step2 Step3 Step4 "
    "Preset Modelid Sdmode Acceleration "
    "Width Height Framebatch Fluxtransformerengine Guidance Delta Seed "
    "Usetinyvae Vaeid "
    "Lora1path Lora1scale Lora2path Lora2scale Lora3path Lora3scale "
    "Upscaleenabled Upscalefactor Upscalemethod Upscalehalf "
    "Upscalemaxinequality Upscalemodel "
    "Filterthreshold Filterskip Pausestream "
    "Ipimagepath Ipscale Ipmodel Controlnetmodel Controlnetscale Pushall"
)

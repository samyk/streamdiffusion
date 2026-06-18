"""Shared HAL control parameter menus (build + live upgrade)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamdiffusion_td_bridge.defaults as _bridge_defaults  # noqa: E402

importlib.reload(_bridge_defaults)
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

SEGMENTATION_BACKEND_NAMES = ["auto", "maxine", "cuda"]
SEGMENTATION_BACKEND_LABELS = [
    "Auto (Maxine AIGS → CUDA DeepLab)",
    "Maxine AI Green Screen",
    "CUDA DeepLab (Blackwell)",
]
ATTENTION_BACKEND_NAMES = ["auto", "flash", "sage", "xformers", "sdpa", "none"]
ATTENTION_BACKEND_LABELS = [
    "Auto (flash → sage → xformers → sdpa)",
    "FlashAttention",
    "SageAttention",
    "xFormers",
    "PyTorch SDPA",
    "None (eager)",
]

FLUX_KLEIN_PRESETS = {
    "flux2_klein_fast",
    "flux2_klein_quality",
    "flux2_klein_9b",
}

DIT_PRESETS = {
    "sd35_medium_fast",
    "sd35_medium_quality",
    "sd35_large_fast",
}

TRANSFORMER_PRESETS = FLUX_KLEIN_PRESETS | DIT_PRESETS

PRESET_MENU_NAMES = [
    "sdxl_turbo_fast",
    "sdxl_turbo_quality",
    "sd_turbo_fast",
    "sd_turbo_quality",
    "lcm_lora_style",
    "flux2_klein_fast",
    "flux2_klein_quality",
    "flux2_klein_9b",
    "sd35_medium_fast",
    "sd35_medium_quality",
    "sd35_large_fast",
    "passthrough",
]

PRESET_MENU_LABELS = [
    "SDXL Turbo Fast",
    "SDXL Turbo Quality",
    "SD Turbo Fast",
    "SD Turbo Quality",
    "SD1.5 LCM LoRA",
    "FLUX.2 Klein Fast (4B)",
    "FLUX.2 Klein Quality (4B)",
    "FLUX.2 Klein 9B",
    "SD3.5 Medium Fast",
    "SD3.5 Medium Quality",
    "SD3.5 Large Fast",
    "Passthrough",
]

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
    "sd35_medium_fast": [1, 2, 3, 4],
    "sd35_medium_quality": [1, 2, 3, 4, 5, 6],
    "sd35_large_fast": [1, 2, 3, 4],
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
    if preset in TRANSFORMER_PRESETS or preset.startswith("sd3_"):
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
    if preset in TRANSFORMER_PRESETS or preset.startswith("sd3_"):
        return 1 <= len(steps) <= KLEIN_MAX_STEPS
    return min(steps) >= 15


_bridge = HAL_BRIDGE_LAUNCH_DEFAULTS
_denoise = _bridge["t_index_list"][0] if len(_bridge["t_index_list"]) == 1 else _bridge["t_index_list"][0]
_step2 = _bridge["t_index_list"][1] if len(_bridge["t_index_list"]) > 1 else 0


def _bridge_get(key: str, default):
    return _bridge.get(key, default)


# TouchDesigner hal_control defaults (synced from live instance A).
TD_HAL_DEFAULTS = {
    "Remotehost": HAL_HOST,
    "Remoteport": DAYDREAM_PORT,
    "Streamid": STREAM_ID,
    "Prompt": _bridge_get("prompt", ""),
    "Negativeprompt": _bridge_get("negative_prompt", ""),
    "Prompt2": "",
    "Prompt2weight": 0.0,
    "Promptinterp": "average",
    "Preset": _bridge_get("preset", "sd_turbo_fast"),
    "Width": _bridge_get("width", 960),
    "Height": _bridge_get("height", 536),
    "Denoise": _denoise,
    "Step2": _step2,
    "Step3": 0,
    "Step4": 0,
    "Sdmode": _bridge_get("sdmode", "img2img"),
    "Acceleration": _bridge_get("acceleration", "tensorrt"),
    "Attentionbackend": _bridge_get("attention_backend", "auto"),
    "Modeloptenabled": _bridge_get("modelopt_enabled", False),
    "Modeloptcheckpoint": _bridge_get("modelopt_checkpoint", "") or "",
    "Framebatch": _bridge_get("frame_buffer_size", 1),
    "Fluxtransformerengine": _bridge_get("flux_transformer_engine", True),
    "Guidance": _bridge_get("guidance_scale", 1.1),
    "Delta": _bridge_get("delta", 1.0),
    "Seed": _bridge_get("seed", 2),
    "Usetinyvae": _bridge_get("use_tiny_vae", True),
    "Upscaleenabled": _bridge_get("upscale_enabled", True),
    "Upscalefactor": str(_bridge_get("upscale_factor", 2)),
    "Upscalemethod": _bridge_get("upscale_method", "maxine-vsr"),
    "Upscalehalf": _bridge_get("upscale_half", True),
    "Upscalemaxinequality": _bridge_get("upscale_maxine_quality", "high"),
    "Pipscale": 0.25,
    "Textscale": 1.0,
    "Textlift": 36,
    "Filterthreshold": _bridge_get("similar_image_filter_threshold", 0.0),
    "Filterskip": _bridge_get("similar_image_filter_max_skip_frame", 10),
    "Pausestream": False,
    "Segmentenabled": _bridge_get("segmentation_enabled", False),
    "Persononly": _bridge_get("person_only", False),
    "Cutbackground": _bridge_get("cut_background", False),
    "Segmentfeather": _bridge_get("segmentation_feather", 3.0),
    "Backgroundcolor": _bridge_get("background_color", "#000000"),
    "Segmentbackend": _bridge_get("segmentation_backend", "auto"),
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


# Single scrollable parameter page (sections via appendHeader in build_hal_control).
HAL_CONTROL_PAGE = "HAL"

# Legacy explicit parscope order (unused when UI uses pagescope=HAL + parscope=*).
HAL_CONTROL_PARSCOPE = (
    "Remotehost Remoteport Streamid Pushall "
    "Prompt Negativeprompt Prompt2 Prompt2weight Promptinterp "
    "Denoise Step2 Step3 Step4 "
    "Preset Modelid Sdmode Acceleration Attentionbackend "
    "Width Height Framebatch Fluxtransformerengine Guidance Delta Seed "
    "Modeloptenabled Modeloptcheckpoint "
    "Usetinyvae Vaeid "
    "Lora1path Lora1scale Lora2path Lora2scale Lora3path Lora3scale "
    "Upscaleenabled Upscalefactor Upscalemethod Upscalehalf "
    "Upscalemaxinequality Upscalemodel "
    "Pipscale Textscale Textlift "
    "Filterthreshold Filterskip Pausestream "
    "Segmentenabled Persononly Cutbackground Segmentfeather Backgroundcolor Segmentbackend "
    "Ipimagepath Ipscale Ipmodel Controlnetmodel Controlnetscale"
)

# Parameter execute DAT: sync to hal on change (connection pars excluded).
HAL_SYNC_PARSCOPE = (
    "Prompt Negativeprompt Prompt2 Prompt2weight Promptinterp "
    "Denoise Step2 Step3 Step4 "
    "Preset Modelid Sdmode Acceleration Attentionbackend "
    "Width Height Framebatch Fluxtransformerengine Guidance Delta Seed "
    "Modeloptenabled Modeloptcheckpoint "
    "Usetinyvae Vaeid "
    "Lora1path Lora1scale Lora2path Lora2scale Lora3path Lora3scale "
    "Upscaleenabled Upscalefactor Upscalemethod Upscalehalf "
    "Upscalemaxinequality Upscalemodel "
    "Filterthreshold Filterskip Pausestream "
    "Segmentenabled Persononly Cutbackground Segmentfeather Backgroundcolor Segmentbackend "
    "Ipimagepath Ipscale Ipmodel Controlnetmodel Controlnetscale Pushall"
)

from __future__ import annotations

from typing import Any

MODEL_PRESETS: dict[str, str] = {
    "stabilityai/sd-turbo": "sd_turbo_fast",
    "stabilityai/sdxl-turbo": "sdxl_turbo_fast",
    "runwayml/stable-diffusion-v1-5": "lcm_lora_style",
    "black-forest-labs/FLUX.2-klein-4B": "flux2_klein_fast",
    "black-forest-labs/FLUX.2-klein-9B": "flux2_klein_9b",
    "stabilityai/stable-diffusion-3.5-medium": "sd35_medium_fast",
    "stabilityai/stable-diffusion-3.5-large": "sd35_large_fast",
}

PRESET_NAMES = set(
    [
        "passthrough",
        "sd_turbo_fast",
        "sd_turbo_quality",
        "sdxl_turbo_fast",
        "sdxl_turbo_quality",
        "lcm_lora_style",
        "flux2_klein_fast",
        "flux2_klein_quality",
        "flux2_klein_9b",
        "sd35_medium_fast",
        "sd35_medium_quality",
        "sd35_large_fast",
    ]
)


def normalize_stream_params(
    params: dict[str, Any],
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pass through TD params; validation happens in the worker."""
    return dict(params)


def _model_fields_changed(params: dict[str, Any], previous: dict[str, Any] | None) -> bool:
    keys = (
        "model_id",
        "preset",
        "acceleration",
        "mode",
        "sdmode",
        "frame_buffer_size",
        "flux_transformer_engine",
        "attention_backend",
        "modelopt_enabled",
        "modelopt_checkpoint",
    )
    if not previous:
        return any(key in params for key in keys)
    for key in keys:
        if key in params and params.get(key) != previous.get(key):
            return True
    return False


def _field_changed(
    params: dict[str, Any],
    previous: dict[str, Any] | None,
    key: str,
    *,
    default: Any = None,
) -> bool:
    if key not in params:
        return False
    if not previous:
        return True
    return params.get(key) != previous.get(key, default)


def _resolution_changed(params: dict[str, Any], previous: dict[str, Any] | None) -> bool:
    if not previous:
        return "width" in params or "height" in params
    width_changed = "width" in params and params.get("width") != previous.get("width")
    height_changed = "height" in params and params.get("height") != previous.get("height")
    return width_changed or height_changed


def _resolve_preset(params: dict[str, Any]) -> str | None:
    preset = params.get("preset")
    if isinstance(preset, str) and preset in PRESET_NAMES:
        return preset
    model_id = str(params.get("model_id", "")).strip()
    if model_id in MODEL_PRESETS:
        quality = params.get("quality_mode") or params.get("qualitypreset")
        if quality in ("quality", "sd_turbo_quality", "sdxl_turbo_quality"):
            if MODEL_PRESETS[model_id] == "sd_turbo_fast":
                return "sd_turbo_quality"
            if MODEL_PRESETS[model_id] == "sdxl_turbo_fast":
                return "sdxl_turbo_quality"
        return MODEL_PRESETS[model_id]
    return None


def _acceleration_only_changed(params: dict[str, Any], previous: dict[str, Any] | None) -> bool:
    if not _field_changed(params, previous, "acceleration"):
        return False
    for key in ("model_id", "preset", "mode", "sdmode", "frame_buffer_size", "flux_transformer_engine"):
        if key in params and (not previous or params.get(key) != previous.get(key)):
            return False
    return True


def daydream_params_to_commands(
    params: dict[str, Any],
    *,
    previous: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []

    if params.get("paused") is True:
        commands.append({"type": "set_mode", "mode": "passthrough"})
    elif previous and previous.get("paused") is True and params.get("paused") is False:
        mode = str(params.get("mode", params.get("sdmode", previous.get("mode", "img2img"))))
        commands.append({"type": "set_mode", "mode": mode})

    if _resolution_changed(params, previous):
        commands.append(
            {
                "type": "set_resolution",
                "width": int(params.get("width", previous.get("width", 512) if previous else 512)),
                "height": int(
                    params.get("height", previous.get("height", 512) if previous else 512)
                ),
            }
        )

    if _acceleration_only_changed(params, previous):
        commands.append({"type": "set_acceleration", "acceleration": str(params["acceleration"])})
    elif _field_changed(params, previous, "attention_backend"):
        commands.append(
            {
                "type": "set_attention_backend",
                "attention_backend": str(params["attention_backend"]),
            }
        )
    elif _model_fields_changed(params, previous):
        preset = _resolve_preset(params)
        model_id = str(params.get("model_id", "")).strip()
        load_command: dict[str, Any] = {
            "type": "load_model",
            "mode": str(params.get("mode", params.get("sdmode", "img2img"))),
            "acceleration": str(params.get("acceleration", "tensorrt")),
            "attention_backend": str(params.get("attention_backend", "auto")),
        }
        if preset:
            load_command["preset"] = preset
        elif model_id:
            load_command["model"] = model_id
            load_command["name"] = "custom"
        if params.get("t_index_list"):
            load_command["t_index_list"] = [int(v) for v in params["t_index_list"]]
        if params.get("width") and params.get("height"):
            load_command["width"] = int(params["width"])
            load_command["height"] = int(params["height"])
        if params.get("frame_buffer_size") is not None:
            load_command["frame_buffer_size"] = max(1, int(params["frame_buffer_size"]))
        if params.get("flux_transformer_engine") is not None:
            load_command["flux_transformer_engine"] = bool(params["flux_transformer_engine"])
        if params.get("modelopt_enabled") is not None:
            load_command["modelopt_enabled"] = bool(params["modelopt_enabled"])
        if params.get("modelopt_checkpoint"):
            load_command["modelopt_checkpoint"] = str(params["modelopt_checkpoint"])
        commands.append(load_command)

    if _field_changed(params, previous, "frame_buffer_size") and not _model_fields_changed(params, previous):
        commands.append(
            {
                "type": "set_frame_buffer",
                "frame_buffer_size": max(1, int(params["frame_buffer_size"])),
            }
        )

    if _field_changed(params, previous, "flux_transformer_engine") and not _model_fields_changed(
        params, previous
    ):
        commands.append(
            {
                "type": "set_flux_transformer_engine",
                "enabled": bool(params["flux_transformer_engine"]),
            }
        )

    prompt_entries = []
    if "prompts" in params:
        for entry in params["prompts"]:
            if isinstance(entry, dict):
                text = str(entry.get("text", entry.get("prompt", ""))).strip()
                if text:
                    prompt_entries.append({"text": text, "weight": float(entry.get("weight", 1.0))})
    elif "prompt" in params and params["prompt"] is not None:
        text = str(params["prompt"]).strip()
        if text:
            prompt_entries.append({"text": text, "weight": 1.0})

    prompts_changed = _field_changed(params, previous, "prompts") or _field_changed(
        params, previous, "prompt"
    ) or _field_changed(params, previous, "prompt_interpolation_method")
    if prompt_entries and prompts_changed:
        interpolation = str(params.get("prompt_interpolation_method", "average"))
        if interpolation == "slerp":
            interpolation = "average"
        if len(prompt_entries) == 1:
            commands.append({"type": "set_prompt", "prompt": prompt_entries[0]["text"]})
        else:
            commands.append(
                {
                    "type": "set_prompts",
                    "prompts": prompt_entries,
                    "interpolation": interpolation,
                }
            )

    if _field_changed(params, previous, "negative_prompt") and params.get("negative_prompt") is not None:
        commands.append(
            {
                "type": "set_negative_prompt",
                "negative_prompt": str(params["negative_prompt"]),
            }
        )

    if _field_changed(params, previous, "t_index_list") and params.get("t_index_list"):
        denoise_cmd: dict[str, Any] = {
            "type": "set_denoise",
            "steps": [int(v) for v in params["t_index_list"]],
        }
        preset = _resolve_preset(params)
        if preset:
            denoise_cmd["preset"] = preset
        commands.append(denoise_cmd)

    if _field_changed(params, previous, "guidance_scale"):
        preset = _resolve_preset(params) or str(
            (previous or {}).get("preset", "")
        )
        if not (preset and preset.startswith("flux2_klein")):
            commands.append({"type": "set_guidance_scale", "value": float(params["guidance_scale"])})

    if _field_changed(params, previous, "delta"):
        commands.append({"type": "set_delta", "value": float(params["delta"])})

    if _field_changed(params, previous, "seed"):
        commands.append({"type": "set_seed", "seed": int(params["seed"])})

    if _field_changed(params, previous, "loras"):
        commands.append({"type": "set_loras", "loras": params.get("loras") or []})

    vae_changed = _field_changed(params, previous, "use_tiny_vae", default=True) or _field_changed(
        params, previous, "vae_id"
    )
    if vae_changed:
        commands.append(
            {
                "type": "set_vae",
                "use_tiny_vae": bool(params.get("use_tiny_vae", True)),
                "vae_id": params.get("vae_id"),
            }
        )

    filter_changed = any(
        _field_changed(params, previous, key)
        for key in (
            "enable_similar_image_filter",
            "similar_image_filter_threshold",
            "similar_image_filter_max_skip_frame",
        )
    )
    if filter_changed:
        if params.get("enable_similar_image_filter"):
            commands.append(
                {
                    "type": "set_filter",
                    "threshold": float(params.get("similar_image_filter_threshold", 0.98)),
                    "max_skip_frame": int(params.get("similar_image_filter_max_skip_frame", 10)),
                }
            )
        else:
            commands.append({"type": "set_filter", "threshold": 0.0, "max_skip_frame": 10})

    mode = str(params.get("sdmode", params.get("mode", "img2img")))
    if _field_changed(params, previous, "sdmode") or _field_changed(params, previous, "mode"):
        if not _model_fields_changed(params, previous):
            commands.append({"type": "set_mode", "mode": mode})

    if params.get("ipadapter_image") or params.get("ipadapter_scale"):
        commands.append(
            {
                "type": "set_ipadapter",
                "image": params.get("ipadapter_image"),
                "scale": float(params.get("ipadapter_scale", 0.5)),
                "model": params.get("ipadapter_model", "h94/IP-Adapter"),
            }
        )

    if params.get("controlnet_model"):
        commands.append(
            {
                "type": "set_controlnet",
                "model": params.get("controlnet_model"),
                "scale": float(params.get("controlnet_scale", 0.5)),
            }
        )

    upscale_changed = any(
        _field_changed(params, previous, key)
        for key in (
            "upscale_enabled",
            "upscale_factor",
            "upscale_method",
            "upscale_half",
            "upscale_maxine_quality",
            "upscale_model",
        )
    )
    if upscale_changed:
        commands.append(
            {
                "type": "set_upscale",
                "enabled": bool(params.get("upscale_enabled", False)),
                "factor": int(params.get("upscale_factor", 2)),
                "method": str(params.get("upscale_method", "maxine-vsr")),
                "use_half": bool(params.get("upscale_half", True)),
                "maxine_quality": str(params.get("upscale_maxine_quality", "medium")),
                "model_path": params.get("upscale_model"),
            }
        )

    return commands

"""Canonical launch/runtime defaults (synced from live TD + hal instance A)."""

from __future__ import annotations

# Instance A — paths/ports also in touchdesigner/instances.py
HAL_HOST = "192.168.0.90"
DAYDREAM_PORT = 8780
CONTROL_PORT = 8765
STREAM_ID = "remote-1"
NDI_IN = "td_streamdiffusion_in"
NDI_OUT = "streamdiffusion_out"

# Captured 2026-06-17: sd_turbo_fast @ 960x540, TensorRT, tiny VAE, 2x Maxine.
HAL_BRIDGE_LAUNCH_DEFAULTS = {
    "preset": "sd_turbo_fast",
    "width": 960,
    "height": 536,
    "acceleration": "tensorrt",
    "attention_backend": "auto",
    "prompt": "paper comic halftone hero, Ben-Day dots, speech bubble pop-art",
    "negative_prompt": "blurry, low detail, artifacts, watermark",
    "guidance_scale": 1.1,
    "delta": 1.0,
    "seed": 2,
    "frame_buffer_size": 1,
    "flux_transformer_engine": True,
    "modelopt_enabled": False,
    "modelopt_checkpoint": None,
    "use_tiny_vae": True,
    "upscale_enabled": True,
    "upscale_factor": 2,
    "upscale_method": "maxine-vsr",
    "upscale_half": True,
    "upscale_maxine_quality": "high",
    "t_index_list": [28],
    "sdmode": "img2img",
    "similar_image_filter_threshold": 0.0,
    "similar_image_filter_max_skip_frame": 10,
}

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Mode = Literal["passthrough", "img2img", "txt2img", "v2v"]
Acceleration = Literal["none", "xformers", "tensorrt"]


@dataclass(frozen=True)
class ModelPreset:
    name: str
    model_id_or_path: str
    t_index_list: list[int]
    mode: Mode = "img2img"
    acceleration: Acceleration = "tensorrt"
    frame_buffer_size: int = 1
    use_lcm_lora: bool = False
    use_tiny_vae: bool = True
    cfg_type: str = "none"
    use_denoising_batch: bool = True
    warmup: int = 10
    lora_dict: dict[str, float] | None = None


PRESETS: dict[str, ModelPreset] = {
    "passthrough": ModelPreset(
        name="passthrough",
        model_id_or_path="",
        t_index_list=[35],
        mode="passthrough",
        acceleration="none",
        use_tiny_vae=False,
    ),
    "sd_turbo_fast": ModelPreset(
        name="sd_turbo_fast",
        model_id_or_path="stabilityai/sd-turbo",
        t_index_list=[35],
        mode="img2img",
        acceleration="tensorrt",
        use_lcm_lora=False,
        cfg_type="none",
    ),
    "sd_turbo_quality": ModelPreset(
        name="sd_turbo_quality",
        model_id_or_path="stabilityai/sd-turbo",
        t_index_list=[32, 45],
        mode="img2img",
        acceleration="tensorrt",
        use_lcm_lora=False,
        cfg_type="none",
    ),
    "lcm_lora_style": ModelPreset(
        name="lcm_lora_style",
        model_id_or_path="runwayml/stable-diffusion-v1-5",
        t_index_list=[0, 16, 32, 45],
        mode="img2img",
        acceleration="tensorrt",
        use_lcm_lora=True,
        cfg_type="self",
    ),
    "sdxl_turbo_fast": ModelPreset(
        name="sdxl_turbo_fast",
        model_id_or_path="stabilityai/sdxl-turbo",
        t_index_list=[35],
        mode="img2img",
        acceleration="tensorrt",
        use_lcm_lora=False,
        cfg_type="none",
    ),
    "sdxl_turbo_quality": ModelPreset(
        name="sdxl_turbo_quality",
        model_id_or_path="stabilityai/sdxl-turbo",
        t_index_list=[32, 45],
        mode="img2img",
        acceleration="tensorrt",
        use_lcm_lora=False,
        cfg_type="none",
    ),
}


@dataclass
class BridgeConfig:
    width: int = 512
    height: int = 512
    input_name: str | None = "td_streamdiffusion_in"
    output_name: str = "streamdiffusion_out"
    control_host: str = "0.0.0.0"
    control_port: int = 8765
    daydream_host: str = "0.0.0.0"
    daydream_port: int = 8780
    stream_id: str = "remote-1"
    preset: str = "sd_turbo_fast"
    prompt: str = ""
    negative_prompt: str = ""
    guidance_scale: float = 1.1
    delta: float = 1.0
    seed: int = 2
    engine_dir: str = "engines"
    video_backend: Literal["ndi", "mock"] = "ndi"
    drop_stale_frames: bool = True
    acceleration: Acceleration | None = None
    upscale_enabled: bool = False
    upscale_factor: int = 2
    upscale_method: Literal["bicubic", "realesrgan", "maxine-vsr"] = "maxine-vsr"
    upscale_half: bool = True
    upscale_maxine_quality: str = "medium"
    upscale_model: str | None = None


@dataclass
class RuntimeState:
    status: str = "starting"
    preset: str = "sd_turbo_fast"
    mode: Mode = "img2img"
    model_id_or_path: str = "stabilityai/sd-turbo"
    prompt: str = ""
    negative_prompt: str = ""
    guidance_scale: float = 1.1
    delta: float = 1.0
    strength: float = 0.55
    seed: int = 2
    t_index_list: list[int] = field(default_factory=lambda: [0])
    fps_in: float = 0.0
    fps_out: float = 0.0
    latency_ms: float = 0.0
    last_error: str | None = None
    frame_count: int = 0
    loading: bool = False
    similar_image_filter_threshold: float = 0.0
    similar_image_filter_max_skip_frame: int = 10
    width: int = 512
    height: int = 512
    extra: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "status",
            "status": self.status,
            "preset": self.preset,
            "mode": self.mode,
            "model": self.model_id_or_path,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "guidance_scale": self.guidance_scale,
            "delta": self.delta,
            "strength": self.strength,
            "seed": self.seed,
            "t_index_list": self.t_index_list,
            "fps_in": round(self.fps_in, 2),
            "fps_out": round(self.fps_out, 2),
            "latency_ms": round(self.latency_ms, 2),
            "frame_count": self.frame_count,
            "loading": self.loading,
            "last_error": self.last_error,
            "similar_image_filter_threshold": self.similar_image_filter_threshold,
            "similar_image_filter_max_skip_frame": self.similar_image_filter_max_skip_frame,
            "width": self.width,
            "height": self.height,
            "extra": self.extra,
        }


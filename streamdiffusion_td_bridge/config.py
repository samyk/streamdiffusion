from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Mode = Literal["passthrough", "img2img", "txt2img", "v2v"]
Acceleration = Literal["none", "xformers", "tensorrt"]
PipelineKind = Literal["streamdiffusion", "flux2_klein"]

FLUX2_KLEIN_4B = "black-forest-labs/FLUX.2-klein-4B"
FLUX2_KLEIN_9B = "black-forest-labs/FLUX.2-klein-9B"


def is_flux_preset(*, pipeline: PipelineKind | str, name: str = "") -> bool:
    return pipeline == "flux2_klein" or name.startswith("flux2_klein")


@dataclass(frozen=True)
class ModelPreset:
    name: str
    model_id_or_path: str
    t_index_list: list[int]
    mode: Mode = "img2img"
    pipeline: PipelineKind = "streamdiffusion"
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
    "flux2_klein_fast": ModelPreset(
        name="flux2_klein_fast",
        model_id_or_path="black-forest-labs/FLUX.2-klein-4B",
        t_index_list=[1, 2, 3, 4],
        mode="img2img",
        pipeline="flux2_klein",
        acceleration="none",
        frame_buffer_size=1,
        use_lcm_lora=False,
        use_tiny_vae=False,
        cfg_type="none",
        use_denoising_batch=False,
        warmup=2,
    ),
    "flux2_klein_quality": ModelPreset(
        name="flux2_klein_quality",
        model_id_or_path="black-forest-labs/FLUX.2-klein-4B",
        t_index_list=[1, 2, 3, 4, 5, 6],
        mode="img2img",
        pipeline="flux2_klein",
        acceleration="none",
        frame_buffer_size=2,
        use_lcm_lora=False,
        use_tiny_vae=False,
        cfg_type="none",
        use_denoising_batch=False,
        warmup=3,
    ),
    "flux2_klein_9b": ModelPreset(
        name="flux2_klein_9b",
        model_id_or_path="black-forest-labs/FLUX.2-klein-9B",
        t_index_list=[1, 2, 3, 4],
        mode="img2img",
        pipeline="flux2_klein",
        acceleration="none",
        frame_buffer_size=1,
        use_lcm_lora=False,
        use_tiny_vae=False,
        cfg_type="none",
        use_denoising_batch=False,
        warmup=2,
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
    frame_buffer_size: int | None = None
    flux_transformer_engine: bool = True
    upscale_enabled: bool = False
    upscale_factor: int = 2
    upscale_method: Literal["bicubic", "realesrgan", "maxine-vsr"] = "maxine-vsr"
    upscale_half: bool = True
    upscale_maxine_quality: str = "medium"
    upscale_model: str | None = None

    def effective_frame_buffer_size(self) -> int:
        if self.frame_buffer_size is not None:
            return self.frame_buffer_size
        preset = PRESETS.get(self.preset)
        return preset.frame_buffer_size if preset else 1

    def effective_acceleration(self) -> str:
        if self.acceleration is not None:
            return self.acceleration
        preset = PRESETS.get(self.preset)
        return preset.acceleration if preset else "tensorrt"

    def output_resolution(self) -> tuple[int, int]:
        if self.upscale_enabled:
            return self.width * self.upscale_factor, self.height * self.upscale_factor
        return self.width, self.height

    def print_startup_settings(self) -> None:
        preset_entry = PRESETS.get(self.preset)
        pipeline = preset_entry.pipeline if preset_entry else "unknown"
        model = preset_entry.model_id_or_path if preset_entry else "unknown"
        frame_buffer = self.effective_frame_buffer_size()
        frame_buffer_src = "explicit" if self.frame_buffer_size is not None else "preset default"
        acceleration = self.effective_acceleration()
        out_w, out_h = self.output_resolution()

        if self.flux_transformer_engine:
            flux_engine = "on (Blackwell bfloat16 + torch.compile)"
        else:
            flux_engine = "off (float16 eager)"

        if self.upscale_enabled:
            upscale = f"on x{self.upscale_factor} ({self.upscale_method}"
            if self.upscale_method == "maxine-vsr":
                upscale += f", quality {self.upscale_maxine_quality}"
            elif self.upscale_method == "realesrgan" and self.upscale_half:
                upscale += ", fp16"
            upscale += ")"
        else:
            upscale = "off"

        lines = [
            "",
            "=== sdtd-bridge settings ===",
            f"  preset:              {self.preset}",
            f"  pipeline:            {pipeline}",
            f"  model:               {model or '(passthrough)'}",
            f"  mode:                {preset_entry.mode if preset_entry else 'unknown'}",
            f"  prompt:              {self.prompt}",
            f"  infer resolution:    {self.width} x {self.height}",
            f"  output resolution:   {out_w} x {out_h}",
            f"  frame_buffer_size:   {frame_buffer} ({frame_buffer_src})",
            f"  flux_transformer:    {flux_engine}",
            f"  acceleration:        {acceleration}",
            f"  upscale:             {upscale}",
            f"  video backend:       {self.video_backend}",
            f"  NDI in:              {self.input_name}",
            f"  NDI out:             {self.output_name}",
            f"  stream id:           {self.stream_id}",
            f"  REST API:            :{self.daydream_port}/v1/streams/{self.stream_id}",
            f"  WebSocket control:   :{self.control_port}/control",
            "",
        ]
        if preset_entry and preset_entry.t_index_list:
            lines.insert(7, f"  t_index_list:        {preset_entry.t_index_list}")
        print("\n".join(lines), flush=True)


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


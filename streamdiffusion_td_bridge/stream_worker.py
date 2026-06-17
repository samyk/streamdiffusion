from __future__ import annotations

import queue
import time
import traceback
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from PIL import Image

from .accel import resolve_acceleration
from .config import PRESETS, BridgeConfig, ModelPreset, RuntimeState
from .control import normalize_resolution, parse_prompt_entries, parse_t_index_list
from .deps import load_wrapper_class
from .frames import LatestFrameQueue, SharedState, VideoFrame
from .ndi_io import resize_rgb
from .upscaler import Upscaler, create_upscaler


class StreamWorker:
    def __init__(
        self,
        config: BridgeConfig,
        input_queue: LatestFrameQueue,
        output_queue: LatestFrameQueue,
        command_queue: "queue.Queue[dict[str, Any]]",
        state: SharedState,
    ) -> None:
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.command_queue = command_queue
        self.state = state
        self.active_preset = PRESETS[config.preset]
        self.wrapper = None
        self.prompt = config.prompt
        self.prompt_entries: list[dict[str, Any]] = (
            [{"text": config.prompt, "weight": 1.0}] if config.prompt else []
        )
        self.negative_prompt = config.negative_prompt
        self.guidance_scale = config.guidance_scale
        self.delta = config.delta
        self.seed = config.seed
        self.prompt_interpolation = "average"
        self.use_tiny_vae = self.active_preset.use_tiny_vae
        self.vae_id: str | None = None
        self._needs_reload = self.active_preset.mode != "passthrough"
        self._stop = False
        self.upscaler: Upscaler | None = None
        self._reload_upscaler()

        self.state.mutate(self._sync_state)

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        self.state.update(status="running")
        while not self._stop:
            self._drain_commands()
            if self._needs_reload:
                self._load_model()

            frame = self.input_queue.get(timeout=0.05)
            if frame is None:
                continue

            started = time.perf_counter()
            try:
                output = self._process(frame)
                latency_ms = (time.perf_counter() - frame.timestamp) * 1000
                self.output_queue.put(
                    VideoFrame(data=output, timestamp=frame.timestamp, sequence=frame.sequence)
                )
                self.state.mutate(
                    lambda state: _set_many(
                        state,
                        status="running",
                        last_error=None,
                        latency_ms=latency_ms,
                        frame_count=state.frame_count + 1,
                    )
                )
            except Exception as exc:  # keep the service alive during live shows
                self.state.update(status="error", last_error=f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
                time.sleep(0.2)
            finally:
                elapsed = time.perf_counter() - started
                if elapsed > 0:
                    self.state.update(fps_out=1.0 / elapsed)

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                return
            self._apply_command(command)

    def _apply_command(self, command: dict[str, Any]) -> None:
        ctype = command.get("type")
        if ctype == "set_prompt":
            self.prompt_entries = [{"text": str(command.get("prompt", "")), "weight": 1.0}]
            self.prompt = self.prompt_entries[0]["text"]
            self._apply_prompt_embeddings()
            self.state.update(prompt=self.prompt)
            return

        if ctype == "set_prompts":
            self.prompt_entries = parse_prompt_entries(command)
            self.prompt_interpolation = str(command.get("interpolation", "average"))
            self.prompt = " | ".join(entry["text"] for entry in self.prompt_entries)
            self._apply_prompt_embeddings()
            self.state.update(prompt=self.prompt)
            return

        if ctype == "set_negative_prompt":
            self.negative_prompt = str(command.get("negative_prompt", command.get("value", "")))
            self._prepare_current_prompt()
            self.state.update(negative_prompt=self.negative_prompt)
            return

        if ctype in ("set_denoise", "set_t_index_list"):
            self._update_t_index_list(parse_t_index_list(command))
            return

        if ctype == "set_strength":
            self._update_t_index_list(
                parse_t_index_list(
                    {
                        "value": float(command["value"]),
                        "scale": "normalized" if float(command["value"]) <= 1.0 else "steps",
                    }
                )
            )
            self.state.update(strength=float(command["value"]))
            return

        if ctype == "set_guidance_scale":
            self.guidance_scale = float(command["value"])
            self._prepare_current_prompt()
            self.state.update(guidance_scale=self.guidance_scale)
            return

        if ctype == "set_delta":
            self.delta = float(command["value"])
            self._prepare_current_prompt()
            self.state.update(delta=self.delta)
            return

        if ctype == "set_seed":
            self.seed = int(command.get("seed", command.get("value", self.seed)))
            self._prepare_current_prompt()
            self.state.update(seed=self.seed)
            return

        if ctype == "set_filter":
            threshold = float(command.get("threshold", 0.0))
            max_skip = int(command.get("max_skip_frame", 10))
            self.state.update(
                similar_image_filter_threshold=threshold,
                similar_image_filter_max_skip_frame=max_skip,
            )
            if self.wrapper is not None and hasattr(self.wrapper, "stream"):
                stream = self.wrapper.stream
                if threshold <= 0.0:
                    stream.disable_similar_image_filter()
                else:
                    stream.enable_similar_image_filter(threshold, max_skip)
            return

        if ctype == "set_mode":
            mode = command["mode"]
            if mode == self.active_preset.mode:
                return
            self.active_preset = replace(self.active_preset, mode=mode)
            self._needs_reload = mode != "passthrough"
            if mode == "passthrough":
                self.wrapper = None
            self.state.update(mode=mode)
            return

        if ctype == "set_lora":
            path = str(command["path"])
            scale = float(command.get("scale", 1.0))
            loras = dict(self.active_preset.lora_dict or {})
            loras[path] = scale
            self.active_preset = replace(self.active_preset, lora_dict=loras)
            self._needs_reload = self.active_preset.mode != "passthrough"
            return

        if ctype == "set_loras":
            loras_list = command.get("loras") or []
            loras = {}
            for entry in loras_list:
                if not isinstance(entry, dict):
                    continue
                path = str(entry.get("path", entry.get("name", ""))).strip()
                if path:
                    loras[path] = float(entry.get("scale", 1.0))
            self.active_preset = replace(self.active_preset, lora_dict=loras or None)
            self.wrapper = None
            self._needs_reload = self.active_preset.mode != "passthrough"
            return

        if ctype == "set_vae":
            use_tiny_vae = bool(command.get("use_tiny_vae", True))
            vae_id = command.get("vae_id") or None
            if use_tiny_vae == self.use_tiny_vae and vae_id == self.vae_id:
                return
            self.use_tiny_vae = use_tiny_vae
            self.vae_id = vae_id
            self.wrapper = None
            self._needs_reload = self.active_preset.mode != "passthrough"
            self.state.update(
                extra={
                    **self.state.snapshot().get("extra", {}),
                    "use_tiny_vae": self.use_tiny_vae,
                    "vae_id": self.vae_id,
                }
            )
            return

        if ctype in ("set_ipadapter", "set_controlnet"):
            extra = dict(self.state.snapshot().get("extra", {}))
            extra[ctype] = command
            extra[f"{ctype}_status"] = (
                "queued (requires TensorRT; unavailable on Blackwell acceleration=none)"
            )
            self.state.update(extra=extra)
            print(
                f"[stream_worker] {ctype} requested but advanced processors need TensorRT. "
                "Ignored on Blackwell until TRT supports sm_120."
            )
            return

        if ctype == "set_resolution":
            width, height = normalize_resolution(
                int(command.get("width", self.config.width)),
                int(command.get("height", self.config.height)),
            )
            if width == self.config.width and height == self.config.height:
                return
            self.config.width = width
            self.config.height = height
            self.wrapper = None
            self._needs_reload = self.active_preset.mode != "passthrough"
            self.state.mutate(self._sync_state)
            return

        if ctype == "set_upscale":
            enabled = bool(command.get("enabled", command.get("upscale_enabled", True)))
            factor = int(command.get("factor", command.get("upscale_factor", self.config.upscale_factor)))
            method = str(command.get("method", command.get("upscale_method", self.config.upscale_method)))
            use_half = bool(command.get("use_half", command.get("upscale_half", self.config.upscale_half)))
            maxine_quality = str(
                command.get("maxine_quality", command.get("upscale_maxine_quality", self.config.upscale_maxine_quality))
            )
            model_path = command.get("model_path", command.get("upscale_model"))
            changed = (
                enabled != self.config.upscale_enabled
                or factor != self.config.upscale_factor
                or method != self.config.upscale_method
                or use_half != self.config.upscale_half
                or maxine_quality != self.config.upscale_maxine_quality
                or (model_path is not None and str(model_path) != (self.config.upscale_model or ""))
            )
            if not changed:
                return
            self.config.upscale_enabled = enabled
            self.config.upscale_factor = factor
            self.config.upscale_method = method  # type: ignore[assignment]
            self.config.upscale_half = use_half
            self.config.upscale_maxine_quality = maxine_quality
            if model_path is not None:
                self.config.upscale_model = str(model_path) if model_path else None
            self._reload_upscaler()
            self.state.mutate(self._sync_state)
            return

        if ctype == "load_model":
            if command.get("width") is not None and command.get("height") is not None:
                width, height = normalize_resolution(
                    int(command["width"]),
                    int(command["height"]),
                )
                self.config.width = width
                self.config.height = height
                self.state.update(width=width, height=height)
            preset_name = command.get("preset")
            if preset_name:
                if preset_name not in PRESETS:
                    raise ValueError(f"Unknown preset {preset_name!r}. Known: {sorted(PRESETS)}")
                self.active_preset = PRESETS[preset_name]
                if command.get("t_index_list"):
                    self.active_preset = replace(
                        self.active_preset,
                        t_index_list=[int(v) for v in command["t_index_list"]],
                    )
            elif command.get("model"):
                self.active_preset = replace(
                    self.active_preset,
                    name=str(command.get("name", "custom")),
                    model_id_or_path=str(command["model"]),
                    mode=command.get("mode", self.active_preset.mode),
                    acceleration=command.get("acceleration", self.active_preset.acceleration),
                    t_index_list=list(command.get("t_index_list", self.active_preset.t_index_list)),
                )
            else:
                self.active_preset = replace(
                    self.active_preset,
                    mode=command.get("mode", self.active_preset.mode),
                    acceleration=command.get("acceleration", self.active_preset.acceleration),
                )
            self.use_tiny_vae = self.active_preset.use_tiny_vae
            self.wrapper = None
            self._needs_reload = self.active_preset.mode != "passthrough"
            self.state.mutate(self._sync_state)
            return

        if ctype == "ping":
            return

        raise ValueError(f"Unknown command type: {ctype!r}")

    def _update_t_index_list(self, t_index_list: list[int]) -> None:
        if not t_index_list:
            raise ValueError("t_index_list must not be empty")
        self.active_preset = replace(self.active_preset, t_index_list=t_index_list)
        self.state.update(t_index_list=t_index_list)
        if self.wrapper is None:
            return
        stream = self.wrapper.stream
        stream.t_list = t_index_list
        stream.denoising_steps_num = len(t_index_list)
        self._prepare_current_prompt()

    def _load_model(self) -> None:
        if self.active_preset.mode == "passthrough":
            self.wrapper = None
            self._needs_reload = False
            self.state.update(loading=False, status="running")
            return

        self.state.update(loading=True, status="loading", last_error=None)
        wrapper_cls = load_wrapper_class()
        acceleration = resolve_acceleration(self.config.acceleration, self.active_preset.acceleration)
        extra = dict(self.state.snapshot().get("extra", {}))
        extra["acceleration"] = acceleration
        self.state.update(extra=extra)
        self.wrapper = wrapper_cls(
            model_id_or_path=self.active_preset.model_id_or_path,
            t_index_list=self.active_preset.t_index_list,
            lora_dict=self.active_preset.lora_dict,
            mode="img2img" if self.active_preset.mode == "v2v" else self.active_preset.mode,
            output_type="pil",
            frame_buffer_size=self.active_preset.frame_buffer_size,
            width=self.config.width,
            height=self.config.height,
            warmup=self.active_preset.warmup,
            acceleration=acceleration,
            use_lcm_lora=self.active_preset.use_lcm_lora,
            use_tiny_vae=self.use_tiny_vae,
            vae_id=self.vae_id,
            use_denoising_batch=self.active_preset.use_denoising_batch,
            cfg_type=self.active_preset.cfg_type,
            seed=self.seed,
            engine_dir=self.config.engine_dir,
        )
        self._prepare_current_prompt()
        self._apply_prompt_embeddings()
        self._needs_reload = False
        self.state.update(loading=False, status="running")

    def _prepare_current_prompt(self) -> None:
        if self.wrapper is None:
            return
        if hasattr(self.wrapper, "prepare"):
            self.wrapper.prepare(
                self.prompt,
                negative_prompt=self.negative_prompt,
                guidance_scale=self.guidance_scale,
                delta=self.delta,
                seed=self.seed,
            )
            if hasattr(self.wrapper.stream, "generator") and self.wrapper.stream.generator is not None:
                self.wrapper.stream.generator.manual_seed(self.seed)

    def _apply_prompt_embeddings(self) -> None:
        if self.wrapper is None or not self.prompt_entries:
            return
        if len(self.prompt_entries) == 1:
            self.wrapper.stream.update_prompt(self.prompt_entries[0]["text"])
            return

        stream = self.wrapper.stream
        embeds = []
        pooled_embeds = []
        total_weight = 0.0
        for entry in self.prompt_entries:
            weight = float(entry.get("weight", 1.0))
            if getattr(stream, "_sdxl_patched", False):
                from .vendor.sdxl_patch import _repeat_added, encode_prompt_entry

                encoded, pooled = encode_prompt_entry(stream, entry["text"])
                embeds.append(encoded * weight)
                pooled_embeds.append(pooled * weight)
            else:
                encoded = stream.pipe.encode_prompt(
                    prompt=entry["text"],
                    device=stream.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )[0]
                embeds.append(encoded * weight)
            total_weight += weight
        if total_weight <= 0:
            return
        blended = torch.stack(embeds, dim=0).sum(dim=0) / total_weight
        stream.prompt_embeds = blended.repeat(stream.batch_size, 1, 1)
        if getattr(stream, "_sdxl_patched", False) and pooled_embeds:
            from .vendor.sdxl_patch import _repeat_added

            blended_pooled = torch.stack(pooled_embeds, dim=0).sum(dim=0) / total_weight
            _repeat_added(stream, blended_pooled.to(device=stream.device, dtype=stream.dtype))

    def _reload_upscaler(self) -> None:
        self.upscaler = create_upscaler(
            enabled=self.config.upscale_enabled,
            factor=self.config.upscale_factor,
            method=self.config.upscale_method,
            use_half=self.config.upscale_half,
            model_path=self.config.upscale_model,
            engine_dir=self.config.engine_dir,
            maxine_quality=self.config.upscale_maxine_quality,
        )

    def _process(self, frame: VideoFrame) -> np.ndarray:
        if self.active_preset.mode == "passthrough" or self.wrapper is None:
            rgb = np.ascontiguousarray(frame.data[:, :, :3])
        else:
            image = Image.fromarray(frame.data[:, :, :3], "RGB")
            result = self.wrapper(image=image, prompt=self.prompt)
            rgb = self._result_to_rgb(result)

        if self.upscaler is not None:
            rgb = self.upscaler.upscale_rgb(rgb)
        return rgb

    def _result_to_rgb(self, result: Any) -> np.ndarray:
        if isinstance(result, list):
            result = result[-1]
        if isinstance(result, Image.Image):
            rgb = np.ascontiguousarray(np.array(result.convert("RGB"), dtype=np.uint8))
            return resize_rgb(rgb, self.config.width, self.config.height)
        if isinstance(result, np.ndarray):
            if result.ndim == 4:
                result = result[-1]
            if result.dtype != np.uint8:
                result = np.clip(result * 255 if result.max() <= 1.0 else result, 0, 255).astype(
                    np.uint8
                )
            rgb = np.ascontiguousarray(result[:, :, :3])
            return resize_rgb(rgb, self.config.width, self.config.height)
        raise TypeError(f"Unsupported StreamDiffusion output type: {type(result)!r}")

    def _sync_state(self, state: RuntimeState) -> None:
        state.preset = self.active_preset.name
        state.mode = self.active_preset.mode
        state.model_id_or_path = self.active_preset.model_id_or_path
        state.prompt = self.prompt
        state.negative_prompt = self.negative_prompt
        state.guidance_scale = self.guidance_scale
        state.delta = self.delta
        state.seed = self.seed
        state.t_index_list = list(self.active_preset.t_index_list)
        state.width = self.config.width
        state.height = self.config.height
        extra = dict(state.extra)
        extra["upscale_enabled"] = self.config.upscale_enabled
        extra["upscale_factor"] = self.config.upscale_factor
        extra["upscale_method"] = self.config.upscale_method
        extra["upscale_half"] = self.config.upscale_half
        extra["upscale_maxine_quality"] = self.config.upscale_maxine_quality
        extra["upscale_runtime"] = self.upscaler.method if self.upscaler else "off"
        if self.upscaler is not None:
            out_w, out_h = self.upscaler.output_size(self.config.width, self.config.height)
            extra["output_width"] = out_w
            extra["output_height"] = out_h
        state.extra = extra


def _set_many(obj, **values):  # noqa: ANN001, ANN003, ANN201
    for key, value in values.items():
        setattr(obj, key, value)

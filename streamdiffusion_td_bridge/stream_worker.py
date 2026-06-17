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
from .config import PRESETS, BridgeConfig, ModelPreset, RuntimeState, is_flux_preset
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
        self.config.frame_buffer_size = config.frame_buffer_size or self.active_preset.frame_buffer_size
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
        self._last_loaded_signature: tuple[Any, ...] | None = None
        self._pending_t_index: list[int] | None = None
        self._pending_t_index_at = 0.0
        self._t_index_debounce_s = 0.2
        self._ignored_td: set[tuple[str, str, str]] = set()
        self.upscaler: Upscaler | None = None
        self._reload_upscaler()

        self.state.mutate(self._sync_state)

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        self.state.update(status="running")
        while not self._stop:
            self._drain_commands()
            self._maybe_apply_pending_t_index()
            if self._needs_reload:
                signature = self._load_signature()
                if signature == self._last_loaded_signature and self.wrapper is not None:
                    self._needs_reload = False
                else:
                    self._load_model()
                    if self.wrapper is not None or self.active_preset.mode == "passthrough":
                        self._last_loaded_signature = signature

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
            try:
                self._update_t_index_list(parse_t_index_list(command))
            except ValueError as exc:
                self._ignore_td("denoise", command, str(exc))
            return

        if ctype == "set_strength":
            try:
                self._update_t_index_list(
                    parse_t_index_list(
                        {
                            "value": float(command["value"]),
                            "scale": "normalized" if float(command["value"]) <= 1.0 else "steps",
                        }
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                self._ignore_td("strength", command, str(exc))
                return
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

        if ctype == "set_frame_buffer":
            frame_buffer_size = max(1, int(command.get("frame_buffer_size", command.get("value", 1))))
            if not self._valid_frame_buffer_request(frame_buffer_size):
                self._ignore_td(
                    "frame_buffer_size",
                    frame_buffer_size,
                    "needs multiple denoise steps on StreamDiffusion (use Framebatch=1 for turbo)",
                )
                return
            if frame_buffer_size == self.config.frame_buffer_size:
                return
            self.config.frame_buffer_size = frame_buffer_size
            self.active_preset = replace(self.active_preset, frame_buffer_size=frame_buffer_size)
            if self._reconfigure_streamdiffusion_batch(self.active_preset.t_index_list, frame_buffer_size):
                self.state.mutate(self._sync_state)
            return

        if ctype == "set_flux_transformer_engine":
            enabled = bool(command.get("enabled", command.get("value", True)))
            if enabled == self.config.flux_transformer_engine:
                return
            self.config.flux_transformer_engine = enabled
            if is_flux_preset(
                pipeline=self.active_preset.pipeline,
                name=self.active_preset.name,
            ):
                self.wrapper = None
                self._needs_reload = True
            self.state.mutate(self._sync_state)
            return

        if ctype == "load_model":
            if not self._apply_load_model(command):
                return
            return

        if ctype == "ping":
            return

        raise ValueError(f"Unknown command type: {ctype!r}")

    def _update_t_index_list(self, t_index_list: list[int]) -> None:
        if not self._valid_t_index_list(t_index_list):
            self._ignore_td("t_index_list", t_index_list, "expected 1-4 steps in range 1-49")
            return
        if t_index_list == list(self.active_preset.t_index_list):
            return
        self.active_preset = replace(self.active_preset, t_index_list=t_index_list)
        self.state.update(t_index_list=t_index_list)
        if self.wrapper is None:
            return
        if is_flux_preset(pipeline=self.active_preset.pipeline, name=self.active_preset.name):
            self.wrapper.set_t_index_list(t_index_list)
            return
        self._pending_t_index = list(t_index_list)
        self._pending_t_index_at = time.monotonic()

    def _maybe_apply_pending_t_index(self) -> None:
        if self._pending_t_index is None:
            return
        if time.monotonic() - self._pending_t_index_at < self._t_index_debounce_s:
            return
        t_index_list = self._pending_t_index
        self._pending_t_index = None
        if self._reconfigure_streamdiffusion_batch(
            t_index_list,
            self.config.frame_buffer_size,
        ):
            self.state.mutate(self._sync_state)

    def _reconfigure_streamdiffusion_batch(
        self,
        t_index_list: list[int],
        frame_buffer_size: int,
    ) -> bool:
        if self.wrapper is None:
            return False
        if is_flux_preset(pipeline=self.active_preset.pipeline, name=self.active_preset.name):
            return False
        if not hasattr(self.wrapper, "reconfigure_batch"):
            return False
        if not self._valid_t_index_list(t_index_list):
            return False
        requested = max(1, int(frame_buffer_size))
        if not self._valid_frame_buffer_request(requested, t_index_list=t_index_list):
            return False
        self.config.frame_buffer_size = requested
        self.active_preset = replace(
            self.active_preset,
            t_index_list=list(t_index_list),
            frame_buffer_size=requested,
        )
        try:
            self.wrapper.reconfigure_batch(t_index_list, requested)
            self._prepare_current_prompt()
        except Exception as exc:  # noqa: BLE001
            self._ignore_td("stream_batch", (t_index_list, requested), f"{type(exc).__name__}: {exc}")
            return False
        return True

    def _ignore_td(self, key: str, value: Any, reason: str) -> None:
        token = (key, str(value), reason)
        if token in self._ignored_td:
            return
        self._ignored_td.add(token)
        print(f"[stream_worker] ignoring TD {key}={value!r}: {reason}")

    def _valid_t_index_list(self, t_index_list: list[int]) -> bool:
        if not t_index_list or len(t_index_list) > 4:
            return False
        return all(1 <= int(v) <= 49 for v in t_index_list)

    def _valid_frame_buffer_request(
        self,
        requested: int,
        *,
        t_index_list: list[int] | None = None,
    ) -> bool:
        requested = max(1, int(requested))
        if requested > 8:
            return False
        if is_flux_preset(pipeline=self.active_preset.pipeline, name=self.active_preset.name):
            return True
        steps = t_index_list if t_index_list is not None else list(self.active_preset.t_index_list)
        if len(steps) <= 1 and requested > 1:
            return False
        return True

    def _apply_load_model(self, command: dict[str, Any]) -> bool:
        """Apply load_model only when something valid actually requires a reload."""
        reload_needed = False

        if command.get("width") is not None and command.get("height") is not None:
            width, height = normalize_resolution(int(command["width"]), int(command["height"]))
            if width != self.config.width or height != self.config.height:
                self.config.width = width
                self.config.height = height
                self.state.update(width=width, height=height)
                reload_needed = True

        preset_name = command.get("preset")
        if preset_name:
            if preset_name not in PRESETS:
                self._ignore_td("preset", preset_name, f"unknown preset (known: {sorted(PRESETS)})")
            else:
                previous_t_index = list(self.active_preset.t_index_list)
                next_preset = PRESETS[preset_name]
                t_index_list = command.get("t_index_list") or previous_t_index
                if command.get("t_index_list") and not self._valid_t_index_list(
                    [int(v) for v in t_index_list]
                ):
                    self._ignore_td("t_index_list", t_index_list, "expected 1-4 steps in range 1-49")
                    t_index_list = previous_t_index
                if (
                    preset_name != self.active_preset.name
                    or next_preset.model_id_or_path != self.active_preset.model_id_or_path
                    or list(t_index_list) != list(self.active_preset.t_index_list)
                ):
                    self.active_preset = replace(
                        next_preset,
                        t_index_list=[int(v) for v in t_index_list],
                    )
                    reload_needed = True
        elif command.get("model"):
            pipeline = command.get("pipeline")
            if pipeline is None and str(command["model"]).lower().find("flux.2-klein") >= 0:
                pipeline = "flux2_klein"
            t_index_list = list(command.get("t_index_list", self.active_preset.t_index_list))
            if command.get("t_index_list") and not self._valid_t_index_list(t_index_list):
                self._ignore_td("t_index_list", t_index_list, "expected 1-4 steps in range 1-49")
                t_index_list = list(self.active_preset.t_index_list)
            next_preset = replace(
                self.active_preset,
                name=str(command.get("name", "custom")),
                model_id_or_path=str(command["model"]),
                mode=command.get("mode", self.active_preset.mode),
                acceleration=command.get("acceleration", self.active_preset.acceleration),
                t_index_list=t_index_list,
                pipeline=pipeline or self.active_preset.pipeline,
            )
            if next_preset.model_id_or_path != self.active_preset.model_id_or_path:
                self.active_preset = next_preset
                reload_needed = True
        else:
            next_mode = command.get("mode", self.active_preset.mode)
            next_accel = command.get("acceleration", self.active_preset.acceleration)
            if next_mode != self.active_preset.mode or next_accel != self.active_preset.acceleration:
                self.active_preset = replace(
                    self.active_preset,
                    mode=next_mode,
                    acceleration=next_accel,
                )
                reload_needed = True

        if command.get("frame_buffer_size") is not None:
            frame_buffer_size = max(1, int(command["frame_buffer_size"]))
            if not self._valid_frame_buffer_request(frame_buffer_size):
                self._ignore_td(
                    "frame_buffer_size",
                    frame_buffer_size,
                    "needs multiple denoise steps on StreamDiffusion (use Framebatch=1 for turbo)",
                )
            elif frame_buffer_size != self.config.frame_buffer_size:
                self.config.frame_buffer_size = frame_buffer_size
                self.active_preset = replace(self.active_preset, frame_buffer_size=frame_buffer_size)
                if self.wrapper is not None and self._reconfigure_streamdiffusion_batch(
                    self.active_preset.t_index_list,
                    frame_buffer_size,
                ):
                    reload_needed = False
                else:
                    reload_needed = True

        if command.get("flux_transformer_engine") is not None:
            enabled = bool(command["flux_transformer_engine"])
            if enabled != self.config.flux_transformer_engine:
                self.config.flux_transformer_engine = enabled
                if is_flux_preset(pipeline=self.active_preset.pipeline, name=self.active_preset.name):
                    reload_needed = True

        if not reload_needed:
            return False

        self.use_tiny_vae = self.active_preset.use_tiny_vae
        self.wrapper = None
        self._needs_reload = self.active_preset.mode != "passthrough"
        self.state.mutate(self._sync_state)
        return True

    def _effective_frame_buffer_size(self, requested: int | None = None) -> int:
        size = self.config.frame_buffer_size if requested is None else max(1, int(requested))
        if self._valid_frame_buffer_request(size):
            return size
        return 1

    def _load_signature(self) -> tuple[Any, ...]:
        acceleration = resolve_acceleration(self.config.acceleration, self.active_preset.acceleration)
        return (
            self.active_preset.name,
            self.active_preset.model_id_or_path,
            tuple(self.active_preset.t_index_list),
            self._effective_frame_buffer_size(),
            self.config.width,
            self.config.height,
            self.use_tiny_vae,
            self.vae_id,
            acceleration,
            self.config.flux_transformer_engine,
            self.active_preset.mode,
        )

    def _load_model(self) -> None:
        if self.active_preset.mode == "passthrough":
            self.wrapper = None
            self._needs_reload = False
            self.state.update(loading=False, status="running")
            return

        self.state.update(loading=True, status="loading", last_error=None)
        wrapper_cls = load_wrapper_class(self.active_preset)
        acceleration = resolve_acceleration(self.config.acceleration, self.active_preset.acceleration)
        frame_buffer_size = self.config.frame_buffer_size
        if not self._valid_frame_buffer_request(frame_buffer_size):
            frame_buffer_size = 1
            self.config.frame_buffer_size = 1
        extra = dict(self.state.snapshot().get("extra", {}))
        extra["acceleration"] = acceleration
        extra["pipeline"] = self.active_preset.pipeline
        extra["frame_buffer_size"] = self.config.frame_buffer_size
        extra["flux_transformer_engine"] = self.config.flux_transformer_engine
        self.state.update(extra=extra)
        wrapper_kwargs = dict(
            model_id_or_path=self.active_preset.model_id_or_path,
            t_index_list=self.active_preset.t_index_list,
            lora_dict=self.active_preset.lora_dict,
            mode="img2img" if self.active_preset.mode == "v2v" else self.active_preset.mode,
            output_type="pil",
            frame_buffer_size=frame_buffer_size,
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
        if is_flux_preset(pipeline=self.active_preset.pipeline, name=self.active_preset.name):
            wrapper_kwargs = dict(
                model_id_or_path=self.active_preset.model_id_or_path,
                t_index_list=self.active_preset.t_index_list,
                width=self.config.width,
                height=self.config.height,
                frame_buffer_size=frame_buffer_size,
                guidance_scale=self.guidance_scale,
                seed=self.seed,
                flux_transformer_engine=self.config.flux_transformer_engine,
                warmup=self.active_preset.warmup,
            )
        self.wrapper = wrapper_cls(**wrapper_kwargs)
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
        if is_flux_preset(pipeline=self.active_preset.pipeline, name=self.active_preset.name):
            return
        if hasattr(self.wrapper, "stream") and hasattr(self.wrapper.stream, "generator"):
            if self.wrapper.stream.generator is not None:
                self.wrapper.stream.generator.manual_seed(self.seed)

    def _apply_prompt_embeddings(self) -> None:
        if self.wrapper is None or not self.prompt_entries:
            return
        if is_flux_preset(pipeline=self.active_preset.pipeline, name=self.active_preset.name):
            self.wrapper.prompt = " | ".join(entry["text"] for entry in self.prompt_entries)
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
        extra["pipeline"] = self.active_preset.pipeline
        extra["frame_buffer_size"] = self.config.frame_buffer_size
        extra["flux_transformer_engine"] = self.config.flux_transformer_engine
        if self.upscaler is not None:
            out_w, out_h = self.upscaler.output_size(self.config.width, self.config.height)
            extra["output_width"] = out_w
            extra["output_height"] = out_h
        state.extra = extra


def _set_many(obj, **values):  # noqa: ANN001, ANN003, ANN201
    for key, value in values.items():
        setattr(obj, key, value)

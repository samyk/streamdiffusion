from __future__ import annotations

import gc
import traceback
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

from streamdiffusion_td_bridge.attention import apply_attention, normalize_attention_backend
from streamdiffusion_td_bridge.blackwell import tune_cuda_for_inference

from .sdxl_patch import patch_stream_for_sdxl

def _is_sdxl(model_id_or_path: str) -> bool:
    lower = (model_id_or_path or "").lower()
    return "sdxl" in lower or "xl-turbo" in lower


def _default_tiny_vae_id(model_id_or_path: str) -> str:
    return "madebyollin/taesdxl" if _is_sdxl(model_id_or_path) else "madebyollin/taesd"


def _configure_full_vae(stream: StreamDiffusion, model_id_or_path: str) -> None:
    """Use the pipeline VAE; SDXL full decode needs fp32 upcast (diffusers force_upcast)."""
    vae = stream.pipe.vae
    stream.vae = vae
    for name in ("enable_tiling", "enable_slicing"):
        method = getattr(vae, name, None)
        if callable(method):
            method()

    needs_upcast = getattr(vae.config, "force_upcast", False) or _is_sdxl(model_id_or_path)
    if not needs_upcast:
        return

    compute_dtype = stream.dtype
    original_decode = stream.decode_image

    def decode_image_upcast(x_0_pred_out: torch.Tensor) -> torch.Tensor:
        latent = x_0_pred_out / vae.config.scaling_factor
        if vae.dtype != torch.float32:
            latent = latent.float()
            vae.to(dtype=torch.float32)
            try:
                return vae.decode(latent, return_dict=False)[0]
            finally:
                vae.to(dtype=compute_dtype)
        return original_decode(x_0_pred_out)

    stream.decode_image = decode_image_upcast


torch.set_grad_enabled(False)
tune_cuda_for_inference()


class StreamDiffusionWrapper:
    """Compact version of StreamDiffusion's demo wrapper.

    The upstream project keeps this helper under `utils/wrapper.py`; vendoring a
    small compatible version avoids requiring the whole repository checkout.
    """

    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: list[int],
        lora_dict: dict[str, float] | None = None,
        mode: Literal["img2img", "txt2img"] = "img2img",
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        lcm_lora_id: str | None = None,
        vae_id: str | None = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        attention_backend: str = "auto",
        do_add_noise: bool = True,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: str | Path = "engines",
        **_unused,
    ) -> None:
        if mode == "txt2img" and cfg_type != "none":
            raise ValueError("txt2img mode accepts only cfg_type='none'")
        if mode == "img2img" and not use_denoising_batch:
            raise NotImplementedError("img2img mode currently requires denoising batch")

        self.sd_turbo = "turbo" in (model_id_or_path or "").lower()
        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.mode = mode
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        self.use_denoising_batch = use_denoising_batch
        self.batch_size = (
            len(t_index_list) * frame_buffer_size if use_denoising_batch else frame_buffer_size
        )

        self.stream = self._load_model(
            model_id_or_path=model_id_or_path,
            t_index_list=t_index_list,
            lora_dict=lora_dict,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            acceleration=acceleration,
            attention_backend=attention_backend,
            warmup=warmup,
            do_add_noise=do_add_noise,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
        )

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(
                similar_image_filter_threshold,
                similar_image_filter_max_skip_frame,
            )

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        seed: int = 2,
    ) -> None:
        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
            generator=torch.manual_seed(seed),
            seed=seed,
        )

    def trt_unet_batch_size_for(self, t_index_list: list[int], frame_buffer_size: int) -> int:
        """Compute StreamDiffusion TRT UNet batch size for the given schedule."""
        steps = len(t_index_list)
        frame_buffer_size = max(1, int(frame_buffer_size))
        stream = self.stream
        if not self.use_denoising_batch:
            return frame_buffer_size
        if stream.cfg_type == "initialize":
            return (steps + 1) * frame_buffer_size
        if stream.cfg_type == "full":
            return 2 * steps * frame_buffer_size
        return steps * frame_buffer_size

    def trt_batch_matches(self, t_index_list: list[int], frame_buffer_size: int) -> bool:
        active = getattr(self.stream, "_sdtd_acceleration_active", None)
        if active != "tensorrt":
            return True
        needed = self.trt_unet_batch_size_for(t_index_list, frame_buffer_size)
        return needed == int(self.stream.trt_unet_batch_size)

    def reconfigure_batch(self, t_index_list: list[int], frame_buffer_size: int) -> None:
        """Hot-update denoise steps / frame buffer without reloading the pipeline."""
        if not t_index_list:
            raise ValueError("t_index_list must not be empty")
        frame_buffer_size = max(1, int(frame_buffer_size))
        if not self.trt_batch_matches(t_index_list, frame_buffer_size):
            needed = self.trt_unet_batch_size_for(t_index_list, frame_buffer_size)
            current = int(self.stream.trt_unet_batch_size)
            raise RuntimeError(
                f"TensorRT UNet batch size {current} != required {needed}; reload model to rebuild engines"
            )
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = (
            len(t_index_list) * frame_buffer_size
            if self.use_denoising_batch
            else frame_buffer_size
        )

        stream = self.stream
        stream.t_list = list(t_index_list)
        stream.denoising_steps_num = len(t_index_list)
        stream.frame_bff_size = frame_buffer_size
        if stream.use_denoising_batch:
            stream.batch_size = stream.denoising_steps_num * frame_buffer_size
            if stream.cfg_type == "initialize":
                stream.trt_unet_batch_size = (stream.denoising_steps_num + 1) * stream.frame_bff_size
            elif stream.cfg_type == "full":
                stream.trt_unet_batch_size = 2 * stream.denoising_steps_num * stream.frame_bff_size
            else:
                stream.trt_unet_batch_size = stream.denoising_steps_num * frame_buffer_size
        else:
            stream.trt_unet_batch_size = frame_buffer_size
            stream.batch_size = frame_buffer_size

        self._rebuild_stream_schedule(stream)

    @staticmethod
    def _resize_batch_tensor(
        tensor: torch.Tensor | None,
        batch_size: int,
        stream: StreamDiffusion,
        *,
        fill_zeros: bool = False,
    ) -> torch.Tensor:
        target_shape = (batch_size, 4, stream.latent_height, stream.latent_width)
        if tensor is not None and tuple(tensor.shape) == target_shape:
            return tensor
        if tensor is None or tensor.shape[0] == 0:
            if fill_zeros:
                return torch.zeros(target_shape, dtype=stream.dtype, device=stream.device)
            generator = stream.generator
            return torch.randn(
                target_shape,
                generator=generator,
                device=stream.device,
                dtype=stream.dtype,
            )
        if tensor.shape[0] < batch_size:
            extra = batch_size - tensor.shape[0]
            pad = (
                torch.zeros(
                    (extra, 4, stream.latent_height, stream.latent_width),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                if fill_zeros
                else tensor[-1:].repeat(extra, 1, 1, 1)
            )
            return torch.cat([tensor, pad], dim=0)
        return tensor[:batch_size]

    def _rebuild_stream_schedule(self, stream: StreamDiffusion) -> None:
        """Rebuild timestep/batch tensors after t_index changes (no prompt re-encode)."""
        old_init = getattr(stream, "init_noise", None)
        old_stock = getattr(stream, "stock_noise", None)
        old_embeds = getattr(stream, "prompt_embeds", None)

        if stream.denoising_steps_num > 1:
            stream.x_t_latent_buffer = torch.zeros(
                (
                    (stream.denoising_steps_num - 1) * stream.frame_bff_size,
                    4,
                    stream.latent_height,
                    stream.latent_width,
                ),
                dtype=stream.dtype,
                device=stream.device,
            )
        else:
            stream.x_t_latent_buffer = None

        if old_embeds is not None and old_embeds.shape[0] != stream.batch_size:
            stream.prompt_embeds = old_embeds[:1].repeat(stream.batch_size, 1, 1)

        stream.scheduler.set_timesteps(50, stream.device)
        stream.timesteps = stream.scheduler.timesteps.to(stream.device)

        stream.sub_timesteps = [stream.timesteps[t] for t in stream.t_list]
        sub_timesteps_tensor = torch.tensor(
            stream.sub_timesteps,
            dtype=torch.long,
            device=stream.device,
        )
        stream.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=stream.frame_bff_size if stream.use_denoising_batch else 1,
            dim=0,
        )

        stream.init_noise = self._resize_batch_tensor(old_init, stream.batch_size, stream)
        stream.stock_noise = self._resize_batch_tensor(
            old_stock,
            stream.batch_size,
            stream,
            fill_zeros=True,
        )

        c_skip_list = []
        c_out_list = []
        for timestep in stream.sub_timesteps:
            c_skip, c_out = stream.scheduler.get_scalings_for_boundary_condition_discrete(
                timestep
            )
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        stream.c_skip = (
            torch.stack(c_skip_list)
            .view(len(stream.t_list), 1, 1, 1)
            .to(dtype=stream.dtype, device=stream.device)
        )
        stream.c_out = (
            torch.stack(c_out_list)
            .view(len(stream.t_list), 1, 1, 1)
            .to(dtype=stream.dtype, device=stream.device)
        )

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in stream.sub_timesteps:
            alpha_prod_t_sqrt_list.append(stream.scheduler.alphas_cumprod[timestep].sqrt())
            beta_prod_t_sqrt_list.append(
                (1 - stream.scheduler.alphas_cumprod[timestep]).sqrt()
            )
        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(stream.t_list), 1, 1, 1)
            .to(dtype=stream.dtype, device=stream.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(stream.t_list), 1, 1, 1)
            .to(dtype=stream.dtype, device=stream.device)
        )
        stream.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=stream.frame_bff_size if stream.use_denoising_batch else 1,
            dim=0,
        )
        stream.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=stream.frame_bff_size if stream.use_denoising_batch else 1,
            dim=0,
        )

    def __call__(self, image: str | Image.Image | torch.Tensor | None = None, prompt: str | None = None):
        if self.mode == "img2img":
            return self.img2img(image, prompt)
        return self.txt2img(prompt)

    def txt2img(self, prompt: str | None = None):
        if prompt is not None:
            self.stream.update_prompt(prompt)
        tensor = (
            self.stream.txt2img_sd_turbo(self.batch_size)
            if self.sd_turbo
            else self.stream.txt2img(self.frame_buffer_size)
        )
        return self.postprocess_image(tensor)

    def img2img(self, image: str | Image.Image | torch.Tensor, prompt: str | None = None):
        if prompt is not None:
            self.stream.update_prompt(prompt)
        image = self._prepare_stream_input(image)
        tensor = self.stream(image)
        return self.postprocess_image(tensor)

    def _prepare_stream_input(
        self, image: str | Image.Image | torch.Tensor
    ) -> Image.Image | torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            return image.convert("RGB").resize((self.width, self.height))
        if isinstance(image, torch.Tensor) and image.min() < 0:
            # StreamDiffusion preprocess expects [0, 1] tensors, not legacy [-1, 1].
            return (image / 2 + 0.5).clamp(0, 1)
        return image

    def preprocess_image(self, image: str | Image.Image) -> torch.Tensor:
        return self.stream.image_processor.preprocess(
            self._prepare_stream_input(image),
            self.height,
            self.width,
        ).to(device=self.device, dtype=self.dtype)

    def postprocess_image(self, image_tensor: torch.Tensor):
        tensor = image_tensor.detach().cpu().float()
        if not torch.isfinite(tensor).all():
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        images = postprocess_image(tensor, output_type=self.output_type)
        return images if self.frame_buffer_size > 1 else images[0]

    def _load_model(
        self,
        model_id_or_path: str,
        t_index_list: list[int],
        lora_dict: dict[str, float] | None,
        lcm_lora_id: str | None,
        vae_id: str | None,
        acceleration: Literal["none", "xformers", "tensorrt"],
        attention_backend: str,
        warmup: int,
        do_add_noise: bool,
        use_lcm_lora: bool,
        use_tiny_vae: bool,
        cfg_type: Literal["none", "full", "self", "initialize"],
        seed: int,
        engine_dir: str | Path,
    ) -> StreamDiffusion:
        pipe = self._load_pipe(model_id_or_path)
        stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
        )
        patch_stream_for_sdxl(stream)

        if not self.sd_turbo and use_lcm_lora:
            if lcm_lora_id:
                stream.load_lcm_lora(pretrained_model_name_or_path_or_dict=lcm_lora_id)
            else:
                stream.load_lcm_lora()
            stream.fuse_lora()

        if lora_dict:
            for lora_name, lora_scale in lora_dict.items():
                stream.load_lora(lora_name)
                stream.fuse_lora(lora_scale=lora_scale)

        if use_tiny_vae:
            tiny_vae_id = vae_id or _default_tiny_vae_id(model_id_or_path)
            stream.vae = AutoencoderTiny.from_pretrained(tiny_vae_id).to(
                device=pipe.device,
                dtype=pipe.dtype,
            )
        else:
            _configure_full_vae(stream, model_id_or_path)

        acceleration_active = "none"
        if acceleration == "tensorrt":
            try:
                _accelerate_with_tensorrt(stream, self, model_id_or_path, engine_dir)
                acceleration_active = "tensorrt"
            except Exception:  # noqa: BLE001 - fall back for unsupported GPUs
                print("TensorRT acceleration failed; continuing without it.")
                traceback.print_exc()
                acceleration_active = "none"
        elif acceleration == "xformers":
            try:
                acceleration_active = apply_attention(
                    pipe,
                    "xformers",
                    kind="unet",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"xformers failed ({type(exc).__name__}: {exc}); falling back to sdpa.")
                acceleration_active = apply_attention(pipe, "sdpa", kind="unet")
        else:
            backend = attention_backend
            try:
                acceleration_active = apply_attention(
                    pipe,
                    backend,
                    kind="unet",
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[streamdiffusion] attention backend {backend!r} failed "
                    f"({type(exc).__name__}: {exc}); falling back to sdpa."
                )
                acceleration_active = apply_attention(pipe, "sdpa", kind="unet")

        setattr(stream, "_sdtd_acceleration_requested", acceleration)
        setattr(stream, "_sdtd_attention_backend", normalize_attention_backend(attention_backend))
        setattr(stream, "_sdtd_acceleration_active", acceleration_active)

        if seed < 0:
            seed = int(np.random.randint(0, 1_000_000))

        stream.prepare(
            "",
            "",
            num_inference_steps=50,
            guidance_scale=1.1 if stream.cfg_type in ["full", "self", "initialize"] else 1.0,
            generator=torch.manual_seed(seed),
            seed=seed,
        )
        return stream

    def _load_pipe(self, model_id_or_path: str):
        pipe_cls = StableDiffusionXLPipeline if _is_sdxl(model_id_or_path) else StableDiffusionPipeline
        try:
            return pipe_cls.from_pretrained(model_id_or_path).to(device=self.device, dtype=self.dtype)
        except (ValueError, OSError):
            return pipe_cls.from_single_file(model_id_or_path).to(device=self.device, dtype=self.dtype)


def _accelerate_with_tensorrt(
    stream: StreamDiffusion,
    wrapper: StreamDiffusionWrapper,
    model_id_or_path: str,
    engine_dir: str | Path,
) -> None:
    try:
        from streamdiffusion_td_bridge.cuda_compat import ensure_cuda_cudart_importable

        ensure_cuda_cudart_importable()
        from streamdiffusion_td_bridge.vendor.tensorrt_export_patch import apply_tensorrt_patches

        apply_tensorrt_patches()
        from streamdiffusion_td_bridge.vendor.tensorrt_build_patch import unet_cross_attention_dim
        from polygraphy import cuda
        from streamdiffusion.acceleration.tensorrt import (
            TorchVAEEncoder,
            compile_unet,
            compile_vae_decoder,
            compile_vae_encoder,
        )
        from streamdiffusion.acceleration.tensorrt.engine import (
            AutoencoderKLEngine,
            UNet2DConditionModelEngine,
        )
        from streamdiffusion.acceleration.tensorrt.models import UNet, VAE, VAEEncoder
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "TensorRT acceleration requested but StreamDiffusion TensorRT dependencies "
            "are unavailable. Run `./scripts/install_tensorrt_deps.sh`. "
            f"Root cause: {type(exc).__name__}: {exc}"
        ) from exc

    engine_dir = Path(engine_dir)
    trt_w, trt_h = _trt_engine_size(wrapper)
    prefix = _engine_prefix(
        model_id_or_path,
        wrapper.mode,
        wrapper.batch_size,
        wrapper.frame_buffer_size,
        trt_w,
        trt_h,
    )
    trt_build_opts = {
        "opt_image_height": trt_h,
        "opt_image_width": trt_w,
    }
    unet_path = engine_dir / prefix / "unet.engine"
    vae_encoder_path = engine_dir / prefix / "vae_encoder.engine"
    vae_decoder_path = engine_dir / prefix / "vae_decoder.engine"

    from streamdiffusion_td_bridge.vendor.tensorrt_build_patch import remove_invalid_trt_engine

    for path in (unet_path, vae_encoder_path, vae_decoder_path):
        remove_invalid_trt_engine(path)

    if not unet_path.exists():
        unet_path.parent.mkdir(parents=True, exist_ok=True)
        unet_model = UNet(
            fp16=True,
            device=stream.device,
            max_batch_size=stream.trt_unet_batch_size,
            min_batch_size=stream.trt_unet_batch_size,
            embedding_dim=unet_cross_attention_dim(stream.unet),
            unet_dim=stream.unet.config.in_channels,
        )
        compile_unet(
            stream.unet,
            unet_model,
            str(unet_path) + ".onnx",
            str(unet_path) + ".opt.onnx",
            str(unet_path),
            opt_batch_size=stream.trt_unet_batch_size,
            engine_build_options=trt_build_opts,
        )

    vae_batch = wrapper.batch_size if wrapper.mode == "txt2img" else stream.frame_bff_size
    if not vae_decoder_path.exists():
        vae_decoder_path.parent.mkdir(parents=True, exist_ok=True)
        stream.vae.forward = stream.vae.decode
        compile_vae_decoder(
            stream.vae,
            VAE(device=stream.device, max_batch_size=vae_batch, min_batch_size=vae_batch),
            str(vae_decoder_path) + ".onnx",
            str(vae_decoder_path) + ".opt.onnx",
            str(vae_decoder_path),
            opt_batch_size=vae_batch,
            engine_build_options=trt_build_opts,
        )
        delattr(stream.vae, "forward")

    if not vae_encoder_path.exists():
        vae_encoder_path.parent.mkdir(parents=True, exist_ok=True)
        vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
        compile_vae_encoder(
            vae_encoder,
            VAEEncoder(device=stream.device, max_batch_size=vae_batch, min_batch_size=vae_batch),
            str(vae_encoder_path) + ".onnx",
            str(vae_encoder_path) + ".opt.onnx",
            str(vae_encoder_path),
            opt_batch_size=vae_batch,
            engine_build_options=trt_build_opts,
        )

    cuda_stream = cuda.Stream()
    vae_config = stream.vae.config
    vae_dtype = stream.vae.dtype
    stream.unet = UNet2DConditionModelEngine(str(unet_path), cuda_stream, use_cuda_graph=False)
    stream.vae = AutoencoderKLEngine(
        str(vae_encoder_path),
        str(vae_decoder_path),
        cuda_stream,
        stream.pipe.vae_scale_factor,
        use_cuda_graph=False,
    )
    setattr(stream.vae, "config", vae_config)
    setattr(stream.vae, "dtype", vae_dtype)
    gc.collect()
    torch.cuda.empty_cache()


def _trt_engine_size(wrapper: StreamDiffusionWrapper) -> tuple[int, int]:
    from streamdiffusion_td_bridge.control import normalize_resolution

    return normalize_resolution(wrapper.width, wrapper.height)


def _engine_prefix(
    model_id_or_path: str,
    mode: str,
    batch_size: int,
    frame_buffer_size: int,
    width: int,
    height: int,
) -> str:
    model = Path(model_id_or_path).stem if Path(model_id_or_path).exists() else model_id_or_path
    safe_model = model.replace("/", "--").replace(":", "_")
    return (
        f"{safe_model}--mode-{mode}--batch-{batch_size}--frame-buffer-{frame_buffer_size}"
        f"--{width}x{height}"
    )


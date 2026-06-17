from __future__ import annotations

import gc
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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

        if acceleration == "xformers":
            try:
                stream.pipe.enable_xformers_memory_efficient_attention()
            except Exception as exc:  # noqa: BLE001 - fall back for unsupported GPUs
                print(f"xformers acceleration failed ({type(exc).__name__}: {exc}); continuing without it.")
        elif acceleration == "tensorrt":
            try:
                _accelerate_with_tensorrt(stream, self, model_id_or_path, engine_dir)
            except Exception as exc:  # noqa: BLE001 - fall back for unsupported GPUs
                print(f"TensorRT acceleration failed ({type(exc).__name__}: {exc}); continuing without it.")

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
            "are unavailable. Run `python -m streamdiffusion.tools.install-tensorrt`."
        ) from exc

    engine_dir = Path(engine_dir)
    prefix = _engine_prefix(
        model_id_or_path,
        wrapper.mode,
        wrapper.batch_size,
        wrapper.frame_buffer_size,
    )
    unet_path = engine_dir / prefix / "unet.engine"
    vae_encoder_path = engine_dir / prefix / "vae_encoder.engine"
    vae_decoder_path = engine_dir / prefix / "vae_decoder.engine"

    if not unet_path.exists():
        unet_path.parent.mkdir(parents=True, exist_ok=True)
        unet_model = UNet(
            fp16=True,
            device=stream.device,
            max_batch_size=stream.trt_unet_batch_size,
            min_batch_size=stream.trt_unet_batch_size,
            embedding_dim=stream.text_encoder.config.hidden_size,
            unet_dim=stream.unet.config.in_channels,
        )
        compile_unet(
            stream.unet,
            unet_model,
            str(unet_path) + ".onnx",
            str(unet_path) + ".opt.onnx",
            str(unet_path),
            opt_batch_size=stream.trt_unet_batch_size,
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


def _engine_prefix(
    model_id_or_path: str,
    mode: str,
    batch_size: int,
    frame_buffer_size: int,
) -> str:
    model = Path(model_id_or_path).stem if Path(model_id_or_path).exists() else model_id_or_path
    safe_model = model.replace("/", "--").replace(":", "_")
    return f"{safe_model}--mode-{mode}--batch-{batch_size}--frame-buffer-{frame_buffer_size}"


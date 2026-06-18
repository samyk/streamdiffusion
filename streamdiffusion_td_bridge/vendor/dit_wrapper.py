from __future__ import annotations

import logging
import warnings
from typing import Literal

import torch
from PIL import Image

from streamdiffusion_td_bridge.attention import apply_attention, normalize_attention_backend
from streamdiffusion_td_bridge.blackwell import is_blackwell, preferred_compute_dtype, tune_cuda_for_inference
from streamdiffusion_td_bridge.control import normalize_resolution
from streamdiffusion_td_bridge.modelopt_support import describe_modelopt_status, maybe_load_modelopt_checkpoint

PipelineKind = Literal["streamdiffusion", "flux2_klein", "dit"]

SD35_MEDIUM = "stabilityai/stable-diffusion-3.5-medium"
SD35_LARGE = "stabilityai/stable-diffusion-3.5-large"
SD3_MEDIUM = "stabilityai/stable-diffusion-3-medium"


def is_dit_preset(*, pipeline: PipelineKind | str, name: str = "") -> bool:
    return pipeline == "dit" or name.startswith("sd35_") or name.startswith("sd3_")


def is_transformer_preset(*, pipeline: PipelineKind | str, name: str = "") -> bool:
    from streamdiffusion_td_bridge.config import is_flux_preset

    return is_flux_preset(pipeline=pipeline, name=name) or is_dit_preset(pipeline=pipeline, name=name)


class DitWrapper:
    """DiT / SD3.5 img2img path via diffusers with Blackwell attention + compile."""

    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: list[int],
        *,
        width: int = 512,
        height: int = 512,
        frame_buffer_size: int = 1,
        guidance_scale: float = 4.5,
        seed: int = 2,
        flux_transformer_engine: bool = True,
        attention_backend: str = "auto",
        modelopt_enabled: bool = False,
        modelopt_checkpoint: str | None = None,
        warmup: int = 2,
        **_unused,
    ) -> None:
        tune_cuda_for_inference()
        self.model_id_or_path = model_id_or_path
        self.width, self.height = normalize_resolution(width, height, align=16)
        self.frame_buffer_size = max(1, int(frame_buffer_size))
        self.guidance_scale = float(guidance_scale)
        self.prompt = ""
        self.negative_prompt = ""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_inference_steps = max(1, len(t_index_list) if t_index_list else 4)
        self._attention_backend = normalize_attention_backend(attention_backend)
        self._use_transformer_engine = bool(flux_transformer_engine)
        self._runtime_mode = "float16"
        self._attention_active = "none"

        dtype = preferred_compute_dtype()
        pipe_cls = self._resolve_pipeline_class(model_id_or_path)
        try:
            self.pipe = pipe_cls.from_pretrained(model_id_or_path, torch_dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "GatedRepoError" in type(exc).__name__ or "authorized list" in message:
                raise RuntimeError(
                    f"Hugging Face access denied for {model_id_or_path}. "
                    f"Accept the license at https://huggingface.co/{model_id_or_path}"
                ) from exc
            raise

        if modelopt_enabled and hasattr(self.pipe, "transformer"):
            self.pipe.transformer = maybe_load_modelopt_checkpoint(
                self.pipe.transformer,
                modelopt_checkpoint,
            )

        self.pipe.to(self.device)
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)
        logging.getLogger("diffusers").setLevel(logging.ERROR)

        self._attention_active = apply_attention(
            self.pipe,
            self._attention_backend,
            kind="transformer",
        )

        use_engine = self._use_transformer_engine and is_blackwell()
        if use_engine and hasattr(self.pipe, "transformer") and self.pipe.transformer is not None:
            try:
                self.pipe.transformer = torch.compile(
                    self.pipe.transformer,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                self._runtime_mode = "blackwell-dit-compile-bf16"
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[dit] torch.compile failed ({type(exc).__name__}: {exc}); "
                    "continuing with bfloat16 eager mode."
                )
                self._runtime_mode = "blackwell-bfloat16"
        elif dtype == torch.bfloat16:
            self._runtime_mode = "bfloat16"
        else:
            self._runtime_mode = "float16"

        modelopt_status = describe_modelopt_status(
            enabled=modelopt_enabled,
            checkpoint=modelopt_checkpoint,
        )
        self._generator = torch.Generator(device=self.device).manual_seed(max(0, int(seed)))
        warmup_iters = max(int(warmup), self.frame_buffer_size)
        dummy = Image.new("RGB", (self.width, self.height), color=(0, 0, 0))
        self.prepare("", negative_prompt="")
        for _ in range(warmup_iters):
            _ = self.img2img(dummy)
        print(
            f"[dit] loaded {model_id_or_path} "
            f"({self._runtime_mode}, attention={self._attention_active}, "
            f"modelopt={modelopt_status}, steps={self.num_inference_steps})"
        )

    @staticmethod
    def _resolve_pipeline_class(model_id_or_path: str):
        from diffusers import StableDiffusion3Img2ImgPipeline

        return StableDiffusion3Img2ImgPipeline

    @property
    def stream(self):
        return self

    def set_t_index_list(self, t_index_list: list[int]) -> None:
        if not t_index_list:
            raise ValueError("t_index_list must not be empty")
        new_steps = max(1, len(t_index_list))
        if new_steps != self.num_inference_steps:
            print(f"[dit] num_inference_steps {self.num_inference_steps} -> {new_steps}")
            self.num_inference_steps = new_steps

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 4.5,
        delta: float = 1.0,
        seed: int = 2,
    ) -> None:
        del num_inference_steps, delta
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.guidance_scale = float(guidance_scale)
        self._generator = torch.Generator(device=self.device).manual_seed(max(0, int(seed)))

    def __call__(self, image: Image.Image | None = None, prompt: str | None = None):
        if image is None:
            raise ValueError("DiT bridge path requires an input image")
        return self.img2img(image, prompt)

    def img2img(self, image: Image.Image | str, prompt: str | None = None) -> Image.Image:
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB").resize((self.width, self.height))
        active_prompt = self.prompt if prompt is None else prompt
        strength = 0.65
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result = self.pipe(
                prompt=active_prompt,
                negative_prompt=self.negative_prompt or None,
                image=image,
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                strength=strength,
                generator=self._generator,
                output_type="pil",
            )
        return result.images[0]

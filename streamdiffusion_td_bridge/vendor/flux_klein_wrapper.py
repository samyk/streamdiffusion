from __future__ import annotations

import logging
import warnings
from typing import Literal

import torch
from PIL import Image

PipelineKind = Literal["streamdiffusion", "flux2_klein"]

FLUX2_KLEIN_4B = "black-forest-labs/FLUX.2-klein-4B"
FLUX2_KLEIN_9B = "black-forest-labs/FLUX.2-klein-9B"
# Klein is step-distilled; CFG / guidance is ignored by diffusers.
KLEIN_GUIDANCE_SCALE = 1.0


def is_flux_preset(*, pipeline: PipelineKind | str, name: str = "") -> bool:
    return pipeline == "flux2_klein" or name.startswith("flux2_klein")


def _blackwell_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 12


class FluxKleinWrapper:
    """Optional FLUX.2 Klein img2img path using diffusers Flux2KleinPipeline."""

    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: list[int],
        *,
        width: int = 512,
        height: int = 512,
        frame_buffer_size: int = 1,
        guidance_scale: float = 1.0,
        seed: int = 2,
        flux_transformer_engine: bool = True,
        warmup: int = 2,
        **_unused,
    ) -> None:
        from diffusers import Flux2KleinPipeline

        self.model_id_or_path = model_id_or_path
        self.width = width
        self.height = height
        self.frame_buffer_size = max(1, int(frame_buffer_size))
        self.guidance_scale = KLEIN_GUIDANCE_SCALE
        self.prompt = ""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_inference_steps = max(1, len(t_index_list) if t_index_list else 4)
        self._use_transformer_engine = bool(flux_transformer_engine)
        self._runtime_mode = "float16"

        use_engine = self._use_transformer_engine and _blackwell_gpu()
        dtype = torch.bfloat16 if use_engine else torch.float16

        try:
            self.pipe = Flux2KleinPipeline.from_pretrained(model_id_or_path, torch_dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "GatedRepoError" in type(exc).__name__ or "authorized list" in message:
                fallback = (
                    "flux2_klein_fast or flux2_klein_quality (4B)"
                    if "9B" in model_id_or_path
                    else "another preset"
                )
                raise RuntimeError(
                    f"Hugging Face access denied for {model_id_or_path}. "
                    f"Request access at https://huggingface.co/{model_id_or_path} "
                    f"or run with SDTD_PRESET={fallback}."
                ) from exc
            raise
        self.pipe.to(self.device)
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)
        logging.getLogger("diffusers").setLevel(logging.ERROR)

        if use_engine:
            try:
                self.pipe.transformer = torch.compile(
                    self.pipe.transformer,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                self._runtime_mode = "blackwell-transformer-engine-bf16"
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[flux2_klein] torch.compile failed ({type(exc).__name__}: {exc}); "
                    "continuing with bfloat16 eager mode."
                )
                self._runtime_mode = "blackwell-bfloat16"
        elif dtype == torch.bfloat16:
            self._runtime_mode = "bfloat16"
        else:
            self._runtime_mode = "float16"

        self._generator = torch.Generator(device=self.device).manual_seed(max(0, int(seed)))
        warmup_iters = max(int(warmup), self.frame_buffer_size)
        dummy = Image.new("RGB", (self.width, self.height), color=(0, 0, 0))
        self.prepare("")
        for _ in range(warmup_iters):
            _ = self.img2img(dummy)
        print(
            f"[flux2_klein] loaded {model_id_or_path} "
            f"({self._runtime_mode}, steps={self.num_inference_steps}, "
            f"frame_buffer={self.frame_buffer_size})"
        )

    @property
    def stream(self):
        """Compatibility shim for StreamDiffusion-only control paths."""
        return self

    def set_t_index_list(self, t_index_list: list[int]) -> None:
        if not t_index_list:
            raise ValueError("t_index_list must not be empty")
        new_steps = max(1, len(t_index_list))
        if new_steps != self.num_inference_steps:
            print(
                f"[flux2_klein] num_inference_steps "
                f"{self.num_inference_steps} -> {new_steps}"
            )
            self.num_inference_steps = new_steps

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        delta: float = 1.0,
        seed: int = 2,
    ) -> None:
        del negative_prompt, num_inference_steps, delta, guidance_scale
        self.prompt = prompt
        self.guidance_scale = KLEIN_GUIDANCE_SCALE
        self._generator = torch.Generator(device=self.device).manual_seed(max(0, int(seed)))

    def __call__(self, image: Image.Image | None = None, prompt: str | None = None):
        if image is None:
            raise ValueError("FLUX.2 Klein bridge path requires an input image")
        return self.img2img(image, prompt)

    def img2img(self, image: Image.Image | str, prompt: str | None = None) -> Image.Image:
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB").resize((self.width, self.height))
        active_prompt = self.prompt if prompt is None else prompt
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*[Gg]uidance scale.*ignored.*",
            )
            result = self.pipe(
                image=image,
                prompt=active_prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=KLEIN_GUIDANCE_SCALE,
                generator=self._generator,
                output_type="pil",
            )
        return result.images[0]

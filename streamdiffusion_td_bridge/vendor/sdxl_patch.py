"""Monkey-patch StreamDiffusion for SDXL pipelines."""

from __future__ import annotations

from typing import Optional, Union

import torch


def _is_sdxl_pipe(pipe) -> bool:
    return hasattr(pipe, "_get_add_time_ids") and hasattr(pipe, "text_encoder_2")


def _time_ids_for_stream(stream) -> torch.Tensor:
    pipe = stream.pipe
    projection_dim = int(getattr(pipe.text_encoder_2.config, "projection_dim", 1280))
    add_time_ids = pipe._get_add_time_ids(
        (stream.height, stream.width),
        (0, 0),
        (stream.height, stream.width),
        dtype=stream.dtype,
        text_encoder_projection_dim=projection_dim,
    )
    return add_time_ids.to(stream.device)


def _repeat_added(stream, pooled: torch.Tensor) -> None:
    batch_size = stream.prompt_embeds.shape[0]
    stream.add_text_embeds = pooled.repeat(batch_size, 1)
    stream.add_time_ids = _time_ids_for_stream(stream).repeat(batch_size, 1)


def encode_prompt_entry(stream, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    encoder_output = stream.pipe.encode_prompt(
        prompt=text,
        device=stream.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    pooled = encoder_output[2]
    if pooled is None:
        raise RuntimeError("SDXL encode_prompt did not return pooled_prompt_embeds")
    return encoder_output[0], pooled


def patch_stream_for_sdxl(stream) -> bool:
    if getattr(stream, "_sdxl_patched", False):
        return True
    if not _is_sdxl_pipe(stream.pipe):
        return False

    stream._sdxl_patched = True
    stream.add_text_embeds = None
    stream.add_time_ids = None

    original_prepare = stream.prepare

    @torch.no_grad()
    def prepare_sdxl(
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = torch.manual_seed(2),
        seed: int = 2,
    ) -> None:
        original_prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
            generator=generator,
            seed=seed,
        )
        encoder_output = stream.pipe.encode_prompt(
            prompt=prompt,
            device=stream.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=stream.guidance_scale > 1.0,
            negative_prompt=negative_prompt,
        )
        pooled = encoder_output[2]
        if pooled is None:
            raise RuntimeError("SDXL prepare did not return pooled_prompt_embeds")
        _repeat_added(stream, pooled.to(device=stream.device, dtype=stream.dtype))

    @torch.no_grad()
    def update_prompt_sdxl(prompt: str) -> None:
        encoder_output = stream.pipe.encode_prompt(
            prompt=prompt,
            device=stream.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        stream.prompt_embeds = encoder_output[0].repeat(stream.batch_size, 1, 1)
        pooled = encoder_output[2]
        if pooled is None:
            raise RuntimeError("SDXL update_prompt did not return pooled_prompt_embeds")
        _repeat_added(stream, pooled.to(device=stream.device, dtype=stream.dtype))

    @torch.no_grad()
    def unet_step_sdxl(
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, list[int]],
        idx: Optional[int] = None,
    ):
        if stream.guidance_scale > 1.0 and (stream.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list_in = torch.concat([t_list[0:1], t_list], dim=0)
        elif stream.guidance_scale > 1.0 and (stream.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list_in = torch.concat([t_list, t_list], dim=0)
        else:
            x_t_latent_plus_uc = x_t_latent
            t_list_in = t_list

        batch = x_t_latent_plus_uc.shape[0]
        added_cond_kwargs = {
            "text_embeds": stream.add_text_embeds[:batch],
            "time_ids": stream.add_time_ids[:batch],
        }

        model_pred = stream.unet(
            x_t_latent_plus_uc,
            t_list_in,
            encoder_hidden_states=stream.prompt_embeds[:batch],
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        if stream.guidance_scale > 1.0 and (stream.cfg_type == "initialize"):
            noise_pred_text = model_pred[1:]
            stream.stock_noise = torch.concat(
                [model_pred[0:1], stream.stock_noise[1:]], dim=0
            )
        elif stream.guidance_scale > 1.0 and (stream.cfg_type == "full"):
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred

        if stream.guidance_scale > 1.0 and (
            stream.cfg_type == "self" or stream.cfg_type == "initialize"
        ):
            noise_pred_uncond = stream.stock_noise * stream.delta
        elif stream.guidance_scale > 1.0 and (stream.cfg_type == "full"):
            pass
        else:
            noise_pred_uncond = None

        if stream.guidance_scale > 1.0 and stream.cfg_type != "none":
            model_pred = noise_pred_uncond + stream.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            model_pred = noise_pred_text

        if stream.use_denoising_batch:
            denoised_batch = stream.scheduler_step_batch(model_pred, x_t_latent, idx)
            if stream.cfg_type == "self" or stream.cfg_type == "initialize":
                scaled_noise = stream.beta_prod_t_sqrt * stream.stock_noise
                delta_x = stream.scheduler_step_batch(model_pred, scaled_noise, idx)
                alpha_next = torch.concat(
                    [
                        stream.alpha_prod_t_sqrt[1:],
                        torch.ones_like(stream.alpha_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = alpha_next * delta_x
                beta_next = torch.concat(
                    [
                        stream.beta_prod_t_sqrt[1:],
                        torch.ones_like(stream.beta_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = delta_x / beta_next
                init_noise = torch.concat(
                    [stream.init_noise[1:], stream.init_noise[0:1]], dim=0
                )
                stream.stock_noise = init_noise + delta_x
        else:
            denoised_batch = stream.scheduler_step_batch(model_pred, x_t_latent, idx)

        return denoised_batch, model_pred

    stream.prepare = prepare_sdxl
    stream.update_prompt = update_prompt_sdxl
    stream.unet_step = unet_step_sdxl
    return True

from __future__ import annotations

import gc
from typing import Any

import torch


def patch_tensorrt_onnx_export() -> None:
    """StreamDiffusion TRT export uses dynamic_axes; PyTorch 2.13+ defaults to dynamo export."""
    from streamdiffusion_td_bridge.diffusers_compat import ensure_diffusers_streamdiffusion_compat

    ensure_diffusers_streamdiffusion_compat()
    import streamdiffusion.acceleration.tensorrt.utilities as utilities

    if getattr(utilities, "_sdtd_export_patched", False):
        return

    def export_onnx(
        model: Any,
        onnx_path: str,
        model_data: Any,
        opt_image_height: int,
        opt_image_width: int,
        opt_batch_size: int,
        onnx_opset: int,
    ) -> None:
        with torch.inference_mode(), torch.autocast("cuda"):
            inputs = model_data.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
            torch.onnx.export(
                model,
                inputs,
                onnx_path,
                export_params=True,
                opset_version=onnx_opset,
                do_constant_folding=True,
                input_names=model_data.get_input_names(),
                output_names=model_data.get_output_names(),
                dynamic_axes=model_data.get_dynamic_axes(),
                dynamo=False,
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    utilities.export_onnx = export_onnx
    utilities._sdtd_export_patched = True

    # builder.py does `from .utilities import export_onnx` at import time.
    try:
        import streamdiffusion.acceleration.tensorrt.builder as builder

        builder.export_onnx = export_onnx
    except ImportError:
        pass


def apply_tensorrt_patches() -> None:
    patch_tensorrt_onnx_export()
    from streamdiffusion_td_bridge.vendor.tensorrt_build_patch import patch_tensorrt_engine_build

    patch_tensorrt_engine_build()

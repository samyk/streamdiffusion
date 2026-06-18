from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

_MAX_TRT_WORKSPACE_BYTES = 8 * 1024**3
_MIN_ENGINE_BYTES = 100_000


def _tensorrt_major_version() -> int:
    import tensorrt as trt

    return int(trt.__version__.split(".")[0])


def remove_invalid_trt_engine(engine_path: str | Path, *, min_bytes: int = _MIN_ENGINE_BYTES) -> None:
    path = Path(engine_path)
    if path.exists() and path.stat().st_size < min_bytes:
        print(f"Removing invalid TensorRT engine ({path.stat().st_size} bytes): {path}")
        path.unlink(missing_ok=True)


def patch_tensorrt_engine_build() -> None:
    """TRT 11+ dropped BuilderFlag.FP16; precision comes from ONNX types (STRONGLY_TYPED)."""
    import streamdiffusion.acceleration.tensorrt.utilities as utilities

    if getattr(utilities, "_sdtd_build_patched", False):
        return

    orig_build = utilities.Engine.build
    orig_build_engine = utilities.build_engine

    def build(
        self: Any,
        onnx_path: str,
        fp16: bool,
        input_profile=None,
        enable_refit: bool = False,
        enable_all_tactics: bool = False,
        timing_cache=None,
        workspace_size: int = 0,
    ) -> None:
        remove_invalid_trt_engine(self.engine_path)
        if _tensorrt_major_version() >= 11 and fp16:
            print(
                "TensorRT 11+ ignores BuilderFlag.FP16; building with STRONGLY_TYPED ONNX precision."
            )
            fp16 = False
        if workspace_size > _MAX_TRT_WORKSPACE_BYTES:
            workspace_size = _MAX_TRT_WORKSPACE_BYTES
        orig_build(
            self,
            onnx_path,
            fp16,
            input_profile=input_profile,
            enable_refit=enable_refit,
            enable_all_tactics=enable_all_tactics,
            timing_cache=timing_cache,
            workspace_size=workspace_size,
        )
        remove_invalid_trt_engine(self.engine_path)

    def build_engine(
        engine_path: str,
        onnx_opt_path: str,
        model_data: Any,
        opt_image_height: int,
        opt_image_width: int,
        opt_batch_size: int,
        build_static_batch: bool = False,
        build_dynamic_shape: bool = False,
        build_all_tactics: bool = False,
        build_enable_refit: bool = False,
    ):
        remove_invalid_trt_engine(engine_path)
        return orig_build_engine(
            engine_path,
            onnx_opt_path,
            model_data,
            opt_image_height,
            opt_image_width,
            opt_batch_size,
            build_static_batch=build_static_batch,
            build_dynamic_shape=build_dynamic_shape,
            build_all_tactics=build_all_tactics,
            build_enable_refit=build_enable_refit,
        )

    utilities.Engine.build = build
    utilities.build_engine = build_engine
    _patch_engine_allocate_buffers(utilities)
    utilities._sdtd_build_patched = True


def _patch_engine_allocate_buffers(utilities: Any) -> None:
    if getattr(utilities.Engine, "_sdtd_allocate_patched", False):
        return

    orig_allocate = utilities.Engine.allocate_buffers
    numpy_to_torch_dtype_dict = utilities.numpy_to_torch_dtype_dict

    def allocate_buffers(self: Any, shape_dict=None, device="cuda"):
        import tensorrt as trt

        if hasattr(self.engine, "num_io_tensors"):
            for idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(idx)
                if shape_dict and name in shape_dict:
                    shape = shape_dict[name]
                else:
                    shape = tuple(self.engine.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.context.set_input_shape(name, shape)
                    shape = tuple(self.context.get_tensor_shape(name))
                tensor = torch.empty(shape, dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
                self.tensors[name] = tensor
            return

        return orig_allocate(self, shape_dict=shape_dict, device=device)

    utilities.Engine.allocate_buffers = allocate_buffers
    utilities.Engine._sdtd_allocate_patched = True


def unet_cross_attention_dim(unet: Any) -> int:
    cross = unet.config.cross_attention_dim
    if isinstance(cross, (list, tuple)):
        return int(max(cross))
    return int(cross)

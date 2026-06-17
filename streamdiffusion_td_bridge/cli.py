from __future__ import annotations

import argparse
import asyncio

from .bridge import BridgeApp
from .config import PRESETS, BridgeConfig
from .defaults import HAL_BRIDGE_LAUNCH_DEFAULTS as D


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StreamDiffusion TouchDesigner bridge")
    parser.add_argument("--width", type=int, default=D["width"])
    parser.add_argument("--height", type=int, default=D["height"])
    parser.add_argument("--input-name", default="td_streamdiffusion_in")
    parser.add_argument("--output-name", default="streamdiffusion_out")
    parser.add_argument("--control-host", default="0.0.0.0")
    parser.add_argument("--control-port", type=int, default=8765)
    parser.add_argument("--daydream-host", default="0.0.0.0")
    parser.add_argument("--daydream-port", type=int, default=8780)
    parser.add_argument("--stream-id", default="remote-1")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=D["preset"])
    parser.add_argument("--prompt", default=D["prompt"])
    parser.add_argument("--negative-prompt", default=D["negative_prompt"])
    parser.add_argument("--guidance-scale", type=float, default=D["guidance_scale"])
    parser.add_argument("--delta", type=float, default=D["delta"])
    parser.add_argument("--seed", type=int, default=D["seed"])
    parser.add_argument("--engine-dir", default="engines")
    parser.add_argument(
        "--acceleration",
        choices=["none", "xformers", "tensorrt"],
        default=D["acceleration"],
        help="Override preset acceleration. Use xformers if TensorRT install/engine build fails.",
    )
    parser.add_argument("--video-backend", choices=["ndi", "mock"], default="ndi")
    parser.add_argument(
        "--passthrough-test",
        action="store_true",
        help="Force passthrough preset for NDI/network validation without loading StreamDiffusion.",
    )
    parser.add_argument(
        "--upscale",
        action=argparse.BooleanOptionalAction,
        default=D["upscale_enabled"],
        help="Upscale inference output on GPU before NDI send (Real-ESRGAN or bicubic fallback).",
    )
    parser.add_argument(
        "--upscale-factor",
        type=int,
        default=D["upscale_factor"],
        choices=[1, 2, 4],
        help="Upscale multiplier for NDI output (default: 2).",
    )
    parser.add_argument(
        "--upscale-model",
        default=None,
        help="Path to Real-ESRGAN weights (.pth). Defaults to engines/models/RealESRGAN_x{factor}plus.pth",
    )
    parser.add_argument(
        "--upscale-method",
        choices=["bicubic", "realesrgan", "maxine-vsr"],
        default=D["upscale_method"],
        help="maxine-vsr (NVIDIA, fast), realesrgan (sharp/slow), bicubic (fastest).",
    )
    parser.add_argument(
        "--upscale-half",
        action=argparse.BooleanOptionalAction,
        default=D["upscale_half"],
        help="Use fp16 Real-ESRGAN when supported (default: on).",
    )
    parser.add_argument(
        "--upscale-maxine-quality",
        default=D["upscale_maxine_quality"],
        choices=[
            "low",
            "medium",
            "high",
            "ultra",
            "highbitrate_low",
            "highbitrate_medium",
            "highbitrate_high",
            "highbitrate_ultra",
        ],
        help="Maxine VSR quality preset (default: medium).",
    )
    parser.add_argument(
        "--frame-buffer-size",
        type=int,
        default=None,
        help="StreamDiffusion frame_buffer_size / stream batch depth (default: preset value).",
    )
    parser.add_argument(
        "--flux-transformer-engine",
        action=argparse.BooleanOptionalAction,
        default=D["flux_transformer_engine"],
        help="FLUX.2 Klein: compile transformer with bfloat16 on Blackwell (default: on).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BridgeConfig(
        width=args.width,
        height=args.height,
        input_name=args.input_name,
        output_name=args.output_name,
        control_host=args.control_host,
        control_port=args.control_port,
        daydream_host=args.daydream_host,
        daydream_port=args.daydream_port,
        stream_id=args.stream_id,
        preset="passthrough" if args.passthrough_test else args.preset,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        delta=args.delta,
        seed=args.seed,
        engine_dir=args.engine_dir,
        acceleration=args.acceleration,
        video_backend=args.video_backend,
        upscale_enabled=args.upscale,
        upscale_factor=args.upscale_factor,
        upscale_method=args.upscale_method,
        upscale_half=args.upscale_half,
        upscale_maxine_quality=args.upscale_maxine_quality,
        upscale_model=args.upscale_model,
        flux_transformer_engine=args.flux_transformer_engine,
    )
    if args.frame_buffer_size is not None:
        config.frame_buffer_size = max(1, int(args.frame_buffer_size))
    asyncio.run(BridgeApp(config).run())


if __name__ == "__main__":
    main()


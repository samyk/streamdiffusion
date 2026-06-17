from __future__ import annotations

import argparse
import asyncio
import json

from .bridge import BridgeApp
from .config import BridgeConfig


async def _run_probe(args: argparse.Namespace) -> None:
    app = BridgeApp(
        BridgeConfig(
            width=args.width,
            height=args.height,
            input_name=args.input_name,
            output_name=args.output_name,
            control_host="127.0.0.1",
            control_port=args.control_port,
            preset=args.preset,
            video_backend=args.video_backend,
            prompt=args.prompt,
        )
    )
    task = asyncio.create_task(app.run())
    try:
        await asyncio.sleep(args.seconds)
        print(json.dumps(app.state.snapshot(), indent=2))
    finally:
        app.stop()
        await task


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short bridge latency/FPS probe")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--input-name", default="td_streamdiffusion_in")
    parser.add_argument("--output-name", default="streamdiffusion_out")
    parser.add_argument("--control-port", type=int, default=9876)
    parser.add_argument("--video-backend", choices=["mock", "ndi"], default="mock")
    parser.add_argument("--preset", default="passthrough")
    parser.add_argument("--prompt", default="")
    args = parser.parse_args()
    asyncio.run(_run_probe(args))


if __name__ == "__main__":
    main()


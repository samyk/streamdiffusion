from __future__ import annotations

import argparse


CALLBACK_SCRIPT = r'''
# Paste this into a Text DAT named `sdtd_ws_callbacks` and assign it as the
# WebSocket DAT callbacks DAT. Create a WebSocket DAT named `sdtd_websocket`.
#
# Example send from a Button/Slider Execute DAT:
#   parent().op("sdtd_ws_callbacks").module.send_prompt("liquid chrome orchid")
#   parent().op("sdtd_ws_callbacks").module.send_value("set_strength", 0.55)

import json


def _ws():
    return op("sdtd_websocket")


def send(command):
    _ws().sendText(json.dumps(command))


def send_prompt(prompt):
    send({"type": "set_prompt", "prompt": prompt})


def send_negative_prompt(prompt):
    send({"type": "set_negative_prompt", "negative_prompt": prompt})


def send_value(command_type, value):
    send({"type": command_type, "value": float(value)})


def load_preset(preset):
    send({"type": "load_model", "preset": preset})


def load_model(model, preset="custom", mode="img2img", acceleration="tensorrt"):
    send({
        "type": "load_model",
        "name": preset,
        "model": model,
        "mode": mode,
        "acceleration": acceleration,
    })


def set_filter(threshold=0.98, max_skip_frame=10):
    send({
        "type": "set_filter",
        "threshold": float(threshold),
        "max_skip_frame": int(max_skip_frame),
    })


def onReceiveText(dat, rowIndex, message, bytes=None, peer=None):
    try:
        data = json.loads(message)
    except Exception:
        return

    status = op("sdtd_status")
    if status is not None and data.get("type") == "status":
        status.clear()
        status.appendRow(["key", "value"])
        for key, value in data.items():
            status.appendRow([key, json.dumps(value) if isinstance(value, (dict, list)) else str(value)])
    return
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Print TouchDesigner WebSocket callback DAT code")
    parser.parse_args()
    print(CALLBACK_SCRIPT.strip())


if __name__ == "__main__":
    main()


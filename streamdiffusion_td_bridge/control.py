from __future__ import annotations

from typing import Any


def clamp_t_index(value: float) -> int:
    return max(0, min(49, int(round(value))))


def normalize_resolution(width: int, height: int) -> tuple[int, int]:
    """Snap to multiples of 8 and keep within a practical StreamDiffusion range."""
    width = max(256, min(1536, int(width)))
    height = max(256, min(1536, int(height)))
    width = max(8, (width // 8) * 8)
    height = max(8, (height // 8) * 8)
    return width, height


def denoise_to_t_index(value: float, *, normalized: bool = False) -> int:
    # Only interpret 0-1 as strength when explicitly requested (set_strength).
    # StreamDiffusionTD denoise is always an integer step in 1-49.
    if normalized:
        return clamp_t_index((1.0 - value) * 49)
    return clamp_t_index(value)


def parse_t_index_list(command: dict[str, Any]) -> list[int]:
    if "t_index_list" in command:
        return [clamp_t_index(float(v)) for v in command["t_index_list"]]
    if "steps" in command:
        return [clamp_t_index(float(v)) for v in command["steps"]]
    if "value" in command:
        return [
            denoise_to_t_index(
                float(command["value"]),
                normalized=command.get("scale") == "normalized",
            )
        ]
    raise ValueError("set_denoise requires value, steps, or t_index_list")


def parse_prompt_entries(command: dict[str, Any]) -> list[dict[str, Any]]:
    entries = command.get("prompts") or command.get("promptdict")
    if not entries:
        text = str(command.get("prompt", "")).strip()
        return [{"text": text, "weight": 1.0}] if text else []

    parsed: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, str):
            parsed.append({"text": entry, "weight": 1.0})
            continue
        text = str(entry.get("text", entry.get("prompt", ""))).strip()
        if not text:
            continue
        parsed.append({"text": text, "weight": float(entry.get("weight", 1.0))})
    return parsed

"""Reload hal_control_defs before importing (TD caches sys.modules)."""

from __future__ import annotations

import importlib
import sys

REPO = "/Users/samy/c/touch/samysd/touchdesigner"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import hal_control_defs

importlib.reload(hal_control_defs)

HAL_CONTROL_PAGE = hal_control_defs.HAL_CONTROL_PAGE
HAL_CONTROL_PARSCOPE = hal_control_defs.HAL_CONTROL_PARSCOPE
HAL_SYNC_PARSCOPE = hal_control_defs.HAL_SYNC_PARSCOPE
TD_HAL_DEFAULTS = hal_control_defs.TD_HAL_DEFAULTS
UPSCALE_FACTOR_LABELS = hal_control_defs.UPSCALE_FACTOR_LABELS
UPSCALE_FACTOR_NAMES = hal_control_defs.UPSCALE_FACTOR_NAMES
UPSCALE_MAXINE_QUALITY_LABELS = hal_control_defs.UPSCALE_MAXINE_QUALITY_LABELS
UPSCALE_MAXINE_QUALITY_NAMES = hal_control_defs.UPSCALE_MAXINE_QUALITY_NAMES
UPSCALE_METHOD_LABELS = hal_control_defs.UPSCALE_METHOD_LABELS
UPSCALE_METHOD_NAMES = hal_control_defs.UPSCALE_METHOD_NAMES
ATTENTION_BACKEND_LABELS = hal_control_defs.ATTENTION_BACKEND_LABELS
ATTENTION_BACKEND_NAMES = hal_control_defs.ATTENTION_BACKEND_NAMES
PRESET_MENU_LABELS = hal_control_defs.PRESET_MENU_LABELS
PRESET_MENU_NAMES = hal_control_defs.PRESET_MENU_NAMES
apply_td_hal_defaults = hal_control_defs.apply_td_hal_defaults

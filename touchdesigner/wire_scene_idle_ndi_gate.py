"""Wire + validate TD-native scene idle NDI gate."""

try:
    INSTANCE
except NameError:
    INSTANCE = "a"

import sys

REPO = "/Users/samy/c/touch/samysd/touchdesigner"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from build_scene_idle_ndi import validate_scene_idle, wire_scene_idle_ndi
from instances import get_instance

profile = get_instance(INSTANCE)
vidin = op(profile.vidin)
if vidin is None:
    raise RuntimeError(f"Missing {profile.vidin}")

wire_scene_idle_ndi(vidin, hal_control=profile.hal_control)
validate_scene_idle(vidin, hal_control=profile.hal_control)

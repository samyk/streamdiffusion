"""
TD-native scene idle NDI gate inside vidin (TOP/CHOP + par expressions only).

Uses cacheTOP (previous frame) + executeDAT frame-end preload so scene_ref
advances even when ndiout is inactive. Monitors upstream of ndiout, not ndiout.
"""

from __future__ import annotations

import sys

REPO = "/Users/samy/c/touch/samysd/touchdesigner"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from td_build_utils import (
    connect,
    cook_validate,
    destroy,
    ensure,
    has_par,
    place_node,
    pulse_par,
    require_chop,
    require_expr,
    require_no_errors,
    set_const,
    set_expr,
    set_first_match,
    td_op,
)

SCENE_IDLE_NODES = (
    "in1",
    "scene_monitor",
    "scene_thumb",
    "scene_blur",
    "scene_ref",
    "scene_diff",
    "scene_mono",
    "scene_topchop",
    "scene_metric",
    "scene_metric_lag",
    "scene_idle_tick",
    "scene_seed",
    "scene_seed_edge",
)


def _metric_expr() -> str:
    return "op('scene_metric_lag')[0]"


def _monitor_source(vidin, ndiout, in1):
    """Tap upstream of ndiout so idle detection cooks even when ndiout is inactive."""
    if in1 is not None and in1.inputs:
        return in1
    if ndiout is not None and ndiout.inputs and ndiout.inputs[0] is not None:
        return ndiout.inputs[0]
    return in1


def _enable_timeslice(chop) -> None:
    if chop is None:
        return
    set_const(chop, "timeslice", True)


def _wire_seed_edge(vidin, scene_seed) -> str:
    destroy(vidin, "scene_seed_edge")

    edge = ensure(vidin, "scene_seed_edge", "filterCHOP")
    edge.inputConnectors[0].connect(scene_seed)
    for filter_val in ("edge", "edg", "Edge"):
        if set_first_match(edge, ("filtertype", "type", "filter"), val=filter_val):
            set_first_match(edge, ("channel", "channame"), val="chan1")
            set_first_match(edge, ("scope",), val="*")
            place_node(edge)
            return "abs(op('scene_seed_edge')[0]) > 0.5"
    destroy(vidin, "scene_seed_edge")

    edge = ensure(vidin, "scene_seed_edge", "changeCHOP")
    edge.inputConnectors[0].connect(scene_seed)
    place_node(edge)
    return "abs(op('scene_seed_edge')[0]) > 0.5"


def _wire_ref_tick(vidin) -> None:
    """Pulse cache preload at frame end so scene_ref holds the previous blurred frame."""
    tick = ensure(vidin, "scene_idle_tick", "executeDAT")
    tick.par.active = True
    tick.par.start = False
    if has_par(tick, "framestart"):
        tick.par.framestart = False
    if has_par(tick, "frameend"):
        tick.par.frameend = True
    tick.text = """\
def onFrameEnd(frame):
    ref = op('scene_ref')
    if ref is not None and hasattr(ref.par, 'preload'):
        ref.par.preload.pulse()
    return
"""
    place_node(tick)


def wire_scene_idle_ndi(vidin, *, hal_control: str) -> None:
    hc = f"op('{hal_control}')"
    ndiout = vidin.op("ndiout1")
    if ndiout is None:
        raise RuntimeError(f"Missing ndiout1 in {vidin.path}")

    for legacy in (
        "scene_idle_exec",
        "scene_force",
        "scene_pars",
        "scene_src",
        "scene_prev",
        "scene_seed_lag",
        "scene_seed_delta",
        "scene_prompt_edge",
        "scene_seed_edge",
        "scene_blur",
        "scene_info",
        "scene_monitor",
        "scene_ref",
        "scene_topchop",
        "scene_metric",
        "scene_metric_lag",
        "scene_idle_tick",
    ):
        destroy(vidin, legacy)

    in1 = ensure(vidin, "in1", "inTOP")
    place_node(in1)

    monitor = _monitor_source(vidin, ndiout, in1)
    scene_monitor = ensure(vidin, "scene_monitor", "nullTOP")
    connect(scene_monitor, monitor)
    place_node(scene_monitor)

    scene_thumb = ensure(vidin, "scene_thumb", "fitTOP")
    connect(scene_thumb, scene_monitor)
    set_const(scene_thumb, "outputresolution", "custom")
    set_const(scene_thumb, "resolutionw", 64)
    set_const(scene_thumb, "resolutionh", 64)
    set_const(scene_thumb, "fit", "fill")
    place_node(scene_thumb)

    scene_blur = ensure(vidin, "scene_blur", "blurTOP")
    connect(scene_blur, scene_thumb)
    set_first_match(scene_blur, ("size", "sizex", "filtersize"), val=3)
    set_first_match(scene_blur, ("sizeunit", "units"), val="pixels")
    place_node(scene_blur)

    scene_ref = ensure(vidin, "scene_ref", "cacheTOP")
    connect(scene_ref, scene_blur)
    pulse_par(scene_ref, "preload")
    place_node(scene_ref)

    scene_diff = ensure(vidin, "scene_diff", "subtractTOP")
    scene_diff.inputConnectors[0].connect(scene_blur)
    scene_diff.inputConnectors[1].connect(scene_ref)
    place_node(scene_diff)

    scene_mono = ensure(vidin, "scene_mono", "monochromeTOP")
    connect(scene_mono, scene_diff)
    place_node(scene_mono)

    scene_topchop = ensure(vidin, "scene_topchop", "toptoCHOP")
    scene_topchop.par.top = scene_mono
    _enable_timeslice(scene_topchop)
    place_node(scene_topchop)

    scene_metric = ensure(vidin, "scene_metric", "analyzeCHOP")
    scene_metric.inputConnectors[0].connect(scene_topchop)
    if not set_first_match(scene_metric, ("function", "compute", "analyze"), val="average"):
        raise RuntimeError("analyzeCHOP scene_metric: no function parameter found")
    _enable_timeslice(scene_metric)
    place_node(scene_metric)

    scene_metric_lag = ensure(vidin, "scene_metric_lag", "lagCHOP")
    scene_metric_lag.inputConnectors[0].connect(scene_metric)
    set_first_match(scene_metric_lag, ("lag1", "lag"), val=2)
    set_first_match(scene_metric_lag, ("lagunit", "lagunits"), val="frames")
    _enable_timeslice(scene_metric_lag)
    place_node(scene_metric_lag)

    _wire_ref_tick(vidin)

    scene_seed = ensure(vidin, "scene_seed", "constantCHOP")
    if not set_first_match(
        scene_seed,
        ("value0", "const0value", "constant0", "const0value1"),
        expr=f"{hc}.par.Seed",
    ):
        raise RuntimeError("constantCHOP scene_seed: no value parameter found")
    place_node(scene_seed)

    seed_edge_expr = _wire_seed_edge(vidin, scene_seed)
    metric = _metric_expr()

    active_expr = (
        f"({hc}.par.Sceneidle == 'off' or {hc}.par.Sceneidlendi == 0) or "
        f"({metric} >= {hc}.par.Scenechangethreshold) or "
        f"({hc}.par.Sceneidle == 'promptwake' and ({seed_edge_expr}))"
    )
    if not set_expr(ndiout, "active", active_expr):
        raise RuntimeError("ndiout1.par.active cannot accept expressions")


def validate_scene_idle(vidin, *, hal_control: str) -> None:
    for name in SCENE_IDLE_NODES:
        if vidin.op(name) is None:
            raise RuntimeError(f"scene idle: missing {vidin.path}/{name}")

    hc = td_op(hal_control)
    if hc is None:
        raise RuntimeError(f"scene idle: missing {hal_control}")
    for par_name in ("Sceneidle", "Scenechangethreshold", "Sceneidlendi", "Seed"):
        if not hasattr(hc.par, par_name):
            raise RuntimeError(f"scene idle: {hal_control} missing par {par_name}")

    cook_validate(vidin, "scene idle vidin")
    for name in (
        "scene_monitor",
        "scene_blur",
        "scene_ref",
        "scene_thumb",
        "scene_diff",
        "scene_mono",
        "scene_topchop",
        "scene_metric",
        "scene_metric_lag",
        "scene_idle_tick",
        "scene_seed_edge",
    ):
        require_no_errors(vidin.op(name), label=name)
    require_chop(vidin.op("scene_metric"), label="scene_metric")
    require_chop(vidin.op("scene_metric_lag"), label="scene_metric_lag")
    require_chop(vidin.op("scene_seed_edge"), label="scene_seed_edge")
    require_expr(vidin.op("ndiout1"), "active", label="ndiout1")

    metric = float(vidin.op("scene_metric_lag")[0].eval())
    threshold = float(hc.par.Scenechangethreshold.eval())
    ndi_on = bool(vidin.op("ndiout1").par.active.eval())
    print(
        f"[scene_idle] validated {vidin.path} (control {hal_control}) "
        f"metric={metric:.4f} threshold={threshold:.4f} ndi_active={ndi_on}"
    )

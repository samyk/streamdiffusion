"""Runtime layout updates for vidout/combine (supports multiple instances)."""

TEXT_LIFT_PX = 36
TEXT_FONT_BASE = 28.0
TEXT_LINE_HEIGHT = 1.35
TEXT_CHAR_WIDTH = 0.55
FPS_FONT_BASE = 16.0
FPS_MARGIN_PX = 12

_ndi_fps_state: dict[str, dict] = {}


def _hal_control_for(vidout_path):
    suffix = "_b" if vidout_path.endswith("_b") else ""
    return op(f"/project1/hal_control{suffix}")


def _prompt_text(vidout_path):
    ctrl = _hal_control_for(vidout_path)
    if ctrl and hasattr(ctrl.par, "Prompt"):
        value = ctrl.par.Prompt.eval().strip()
        if value:
            return value
    return "prompt"


def _canvas_size(combine):
    """Canvas = comp1 background (in1 / hal)."""
    bg = combine.op("in1")
    out = combine.op("out1")
    if bg is not None and bg.width > 0 and bg.height > 0:
        return float(bg.width), float(bg.height)
    if out is not None and out.width > 0 and out.height > 0:
        return float(out.width), float(out.height)
    return 768.0, 768.0


def _text_height(prompt, out_w, font_size):
    avg_char_px = max(1.0, font_size * TEXT_CHAR_WIDTH)
    chars_per_line = max(8, int(out_w / avg_char_px))
    lines = max(1, (len(prompt) + chars_per_line - 1) // chars_per_line)
    line_px = font_size * TEXT_LINE_HEIGHT
    return max(48, int(lines * line_px + font_size * 0.25))


def _display_pars(vidout_path):
    ctrl = _hal_control_for(vidout_path)
    if ctrl is not None and hasattr(ctrl.par, "Pipscale"):
        return ctrl.par
    vidout = op(vidout_path)
    if vidout is not None:
        return vidout.par
    return None


def _ndi_receive_fps(ndi_top) -> float:
    if ndi_top is None:
        return 0.0
    key = ndi_top.path
    state = _ndi_fps_state.setdefault(
        key, {"last_count": None, "last_t": None, "fps": 0.0}
    )
    count = int(getattr(ndi_top, "totalCooks", 0))
    now = absTime.seconds
    if state["last_count"] is not None and count > state["last_count"]:
        dt = now - (state["last_t"] or now)
        if dt > 0.0005:
            instant = (count - state["last_count"]) / dt
            if state["fps"] > 0:
                state["fps"] = state["fps"] * 0.85 + instant * 0.15
            else:
                state["fps"] = instant
    state["last_count"] = count
    state["last_t"] = now
    return float(state["fps"])


def _configure_text_top(text_top, *, text, out_w, out_h, font_size, alignx, aligny):
    text_top.par.text = text
    text_top.par.alignx = alignx
    text_top.par.aligny = aligny
    text_top.par.outputresolution = "custom"
    text_top.par.wordwrap = alignx == "center"
    text_top.par.fontautosize = "off"
    text_top.par.fontsizex = font_size
    text_top.par.fontsizey = font_size
    text_top.par.fontsizexunit = "points"
    text_top.par.fontsizeyunit = "points"
    text_top.par.linespacing = max(2.0, font_size * 0.12)
    text_top.par.linespacingunit = "points"
    text_top.par.keepfontratio = True
    text_top.par.fontcolorr = 1
    text_top.par.fontcolorg = 1
    text_top.par.fontcolorb = 1
    text_top.par.fontalpha = 1
    text_top.par.resolutionw = int(out_w)
    text_top.par.resolutionh = int(out_h)


def update_layout(vidout_path="/project1/vidout"):
    vidout = op(vidout_path)
    combine = op(f"{vidout_path}/combine") if vidout else None
    pars = _display_pars(vidout_path)
    if vidout is None or combine is None or pars is None:
        return

    in1 = combine.op("in1")
    in2 = combine.op("in2")
    comp1 = combine.op("comp1")
    comp2 = combine.op("comp2")
    pip_resize = combine.op("pip_resize")
    if in1 is None or in2 is None or comp1 is None or pip_resize is None:
        return

    out_w, out_h = _canvas_size(combine)
    pip_w = max(float(in2.width), 1.0)
    pip_h = max(float(in2.height), 1.0)
    pip_scale = float(pars.Pipscale)
    scaled_w = max(1.0, pip_w * pip_scale)
    scaled_h = max(1.0, pip_h * pip_scale)

    pip_resize.par.outputresolution = "custom"
    pip_resize.par.fit = "fitbest"
    pip_resize.par.resolutionw = int(scaled_w)
    pip_resize.par.resolutionh = int(scaled_h)

    comp1.par.prefit = "nativeres"
    comp1.par.justifyh = "right"
    comp1.par.justifyv = "bottom"
    comp1.par.sx = 1
    comp1.par.sy = 1
    comp1.par.tx = 0
    comp1.par.ty = 0

    text_scale = float(pars.Textscale)
    text_lift = float(getattr(pars, "Textlift", TEXT_LIFT_PX))
    font_size = max(12.0, TEXT_FONT_BASE * text_scale)
    prompt = _prompt_text(vidout_path)
    text_h = _text_height(prompt, out_w, font_size)

    for name in ("text2", "text3", "text4"):
        text_top = combine.op(name)
        if text_top is None:
            continue
        _configure_text_top(
            text_top,
            text=prompt,
            out_w=out_w,
            out_h=text_h,
            font_size=font_size,
            alignx="center",
            aligny="bottom",
        )

    if comp2 is not None:
        comp2.par.sx = 1
        comp2.par.sy = 1
        comp2.par.justifyh = "center"
        comp2.par.justifyv = "bottom"
        comp2.par.tx = 0
        comp2.par.ty = text_lift

    fps = _ndi_receive_fps(op(f"{vidout_path}/ndiin2"))
    text_fps = combine.op("text_fps")
    comp_hud = combine.op("comp_hud")
    if text_fps is not None:
        fps_label = f"NDI {fps:.1f} fps" if fps > 0 else "NDI -- fps"
        fps_font = max(10.0, FPS_FONT_BASE)
        _configure_text_top(
            text_fps,
            text=fps_label,
            out_w=max(180, int(out_w * 0.25)),
            out_h=36,
            font_size=fps_font,
            alignx="left",
            aligny="bottom",
        )
    if comp_hud is not None:
        comp_hud.par.justifyh = "left"
        comp_hud.par.justifyv = "bottom"
        comp_hud.par.tx = FPS_MARGIN_PX
        comp_hud.par.ty = FPS_MARGIN_PX

import json
import urllib.error
import urllib.request

CONTROL_PATH = "/project1/hal_control"
REMOTE_HOST = "192.168.0.90"
REMOTE_PORT = 8780
STREAM_ID = "remote-1"

_last_payload = None


def _control():
    return op(CONTROL_PATH)


def _base_url():
    ctrl = _control()
    host = ctrl.par.Remotehost.eval() if ctrl and hasattr(ctrl.par, "Remotehost") else REMOTE_HOST
    port = int(ctrl.par.Remoteport) if ctrl and hasattr(ctrl.par, "Remoteport") else REMOTE_PORT
    return f"http://{host}:{port}"


def _stream_id():
    ctrl = _control()
    if ctrl and hasattr(ctrl.par, "Streamid"):
        value = ctrl.par.Streamid.eval().strip()
        if value:
            return value
    return STREAM_ID


def _steps(ctrl):
    steps = [int(ctrl.par.Denoise)]
    for name in ("Step2", "Step3", "Step4"):
        value = int(getattr(ctrl.par, name))
        if value > 0:
            steps.append(max(1, min(49, value)))
    return steps or [35]


def _prompts(ctrl):
    prompts = [{"text": ctrl.par.Prompt.eval().strip(), "weight": 1.0}]
    text2 = ctrl.par.Prompt2.eval().strip()
    weight2 = float(ctrl.par.Prompt2weight)
    if text2 and weight2 > 0:
        prompts.append({"text": text2, "weight": weight2})
    return [entry for entry in prompts if entry["text"]]


def _loras(ctrl):
    loras = []
    for index in (1, 2, 3):
        path_par = getattr(ctrl.par, f"Lora{index}path", None)
        scale_par = getattr(ctrl.par, f"Lora{index}scale", None)
        if path_par is None:
            continue
        path = path_par.eval().strip()
        if path:
            loras.append({"path": path, "scale": float(scale_par) if scale_par else 1.0})
    return loras


def build_params():
    ctrl = _control()
    if ctrl is None:
        return {}

    model_id = ctrl.par.Modelid.eval().strip() if hasattr(ctrl.par, "Modelid") else ""
    preset = ctrl.par.Preset.eval() if hasattr(ctrl.par, "Preset") else ""
    quality_mode = ctrl.par.Qualitymode.eval() if hasattr(ctrl.par, "Qualitymode") else "fast"
    if quality_mode == "quality" and preset.endswith("_fast"):
        preset = preset.replace("_fast", "_quality")

    params = {
        "preset": preset,
        "quality_mode": quality_mode,
        "prompts": _prompts(ctrl),
        "negative_prompt": ctrl.par.Negativeprompt.eval()
        if hasattr(ctrl.par, "Negativeprompt")
        else "",
        "t_index_list": _steps(ctrl),
        "guidance_scale": float(ctrl.par.Guidance),
        "delta": float(ctrl.par.Delta),
        "seed": int(ctrl.par.Seed),
        "acceleration": ctrl.par.Acceleration.eval(),
        "mode": ctrl.par.Sdmode.eval(),
        "sdmode": ctrl.par.Sdmode.eval(),
        "width": int(ctrl.par.Width),
        "height": int(ctrl.par.Height),
        "use_tiny_vae": bool(int(ctrl.par.Usetinyvae))
        if hasattr(ctrl.par, "Usetinyvae")
        else True,
        "enable_similar_image_filter": float(ctrl.par.Filterthreshold) > 0.0,
        "similar_image_filter_threshold": float(ctrl.par.Filterthreshold),
        "similar_image_filter_max_skip_frame": int(ctrl.par.Filterskip),
        "paused": bool(int(ctrl.par.Pausestream)) if hasattr(ctrl.par, "Pausestream") else False,
        "prompt_interpolation_method": ctrl.par.Promptinterp.eval()
        if hasattr(ctrl.par, "Promptinterp")
        else "average",
    }

    if model_id:
        params["model_id"] = model_id
    if params["prompts"]:
        params["prompt"] = params["prompts"][0]["text"]

    loras = _loras(ctrl)
    if loras:
        params["loras"] = loras

    vae_id = ctrl.par.Vaeid.eval().strip() if hasattr(ctrl.par, "Vaeid") else ""
    if vae_id:
        params["vae_id"] = vae_id

    ip_path = ctrl.par.Ipimagepath.eval().strip() if hasattr(ctrl.par, "Ipimagepath") else ""
    if ip_path:
        params["ipadapter_image"] = ip_path
        params["ipadapter_scale"] = float(ctrl.par.Ipscale) if hasattr(ctrl.par, "Ipscale") else 0.5

    cn_model = ctrl.par.Controlnetmodel.eval().strip() if hasattr(ctrl.par, "Controlnetmodel") else ""
    if cn_model:
        params["controlnet_model"] = cn_model
        params["controlnet_scale"] = (
            float(ctrl.par.Controlnetscale) if hasattr(ctrl.par, "Controlnetscale") else 0.5
        )

    return params


def push_params(force=False):
    global _last_payload
    params = build_params()
    payload = {"pipeline": "streamdiffusion", "params": params}
    encoded = json.dumps(payload, sort_keys=True)
    if not force and encoded == _last_payload:
        return
    _last_payload = encoded

    url = f"{_base_url()}/v1/streams/{_stream_id()}"
    request = urllib.request.Request(
        url,
        data=encoded.encode("utf-8"),
        method="PATCH",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3.0) as response:
            response.read()
        print(
            f"[hal_remote_sync] {CONTROL_PATH} -> {url} "
            f"{params.get('preset')} {params.get('width')}x{params.get('height')} "
            f"t_index={params.get('t_index_list')} prompt={params.get('prompt', '')!r}"
        )
    except urllib.error.URLError as exc:
        print(f"[hal_remote_sync] PATCH failed: {exc}")


def _update_combine_layout():
    ctrl = _control()
    if ctrl is None:
        return
    suffix = "_b" if ctrl.path.endswith("_b") else ""
    layout = op(f"/project1/vidout{suffix}/combine_layout")
    if layout is not None:
        layout.module.update_layout(f"/project1/vidout{suffix}")


def onValueChange(par, prev):
    if par.name in ("Pipscale", "Textscale", "Textlift"):
        _update_combine_layout()
        return

    tokens = (
        "prompt",
        "negative",
        "denoise",
        "step",
        "guidance",
        "delta",
        "seed",
        "preset",
        "quality",
        "modelid",
        "acceleration",
        "sdmode",
        "width",
        "height",
        "filter",
        "pause",
        "lora",
        "tinyvae",
        "vaeid",
        "ipimage",
        "ipscale",
        "controlnet",
        "promptinterp",
    )
    if any(token in par.name.lower() for token in tokens):
        push_params()
    return


def onPulse(par):
    if par.name in ("Pushall", "Startstream"):
        push_params(force=True)
    return


def onStart():
    push_params(force=True)
    return

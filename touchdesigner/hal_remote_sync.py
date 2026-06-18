import json
import urllib.error
import urllib.request

CONTROL_PATH = "/project1/hal_control"
REMOTE_HOST = "192.168.0.90"
REMOTE_PORT = 8780
STREAM_ID = "remote-1"

_last_payload = None
_last_prompt = None
_last_push_at = 0.0
_push_debounce_s = 0.35


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


def _klein_steps(denoise, step2=0, step3=0, step4=0):
    # Klein: Denoise = step count (1-6). Step2-4 add extra steps when > 0.
    count = max(1, min(6, int(denoise)))
    for value in (step2, step3, step4):
        if int(value) > 0:
            count += 1
    count = min(6, count)
    return list(range(1, count + 1))


def _turbo_steps(denoise, step2=0, step3=0, step4=0):
    steps = [max(1, min(49, int(denoise)))]
    for value in (step2, step3, step4):
        if int(value) > 0:
            steps.append(max(1, min(49, int(value))))
    return steps


def _steps(ctrl):
    preset = ctrl.par.Preset.eval() if hasattr(ctrl.par, "Preset") else ""
    denoise = int(ctrl.par.Denoise)
    step2 = int(ctrl.par.Step2)
    step3 = int(ctrl.par.Step3)
    step4 = int(ctrl.par.Step4)
    flux_presets = {"flux2_klein_fast", "flux2_klein_quality", "flux2_klein_9b"}
    preset_steps = {
        "sdxl_turbo_fast": [35],
        "sdxl_turbo_quality": [32, 45],
        "sd_turbo_fast": [35],
        "sd_turbo_quality": [32, 45],
        "lcm_lora_style": [0, 16, 32, 45],
        "flux2_klein_fast": [1, 2, 3, 4],
        "flux2_klein_quality": [1, 2, 3, 4, 5, 6],
        "flux2_klein_9b": [1, 2, 3, 4],
    }
    if preset in flux_presets:
        return _klein_steps(denoise, step2, step3, step4)
    steps = _turbo_steps(denoise, step2, step3, step4)
    if not steps:
        return preset_steps.get(preset, [35])
    if preset != "lcm_lora_style" and min(steps) < 15:
        return [max(15, int(v)) for v in steps]
    return steps


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

    params = {
        "preset": preset,
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
        "frame_buffer_size": int(ctrl.par.Framebatch)
        if hasattr(ctrl.par, "Framebatch")
        else 1,
        "flux_transformer_engine": bool(int(ctrl.par.Fluxtransformerengine))
        if hasattr(ctrl.par, "Fluxtransformerengine")
        else True,
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

    if hasattr(ctrl.par, "Upscaleenabled"):
        params["upscale_enabled"] = bool(int(ctrl.par.Upscaleenabled))
        params["upscale_factor"] = int(ctrl.par.Upscalefactor.eval())
        params["upscale_method"] = ctrl.par.Upscalemethod.eval()
        params["upscale_half"] = bool(int(ctrl.par.Upscalehalf))
        params["upscale_maxine_quality"] = ctrl.par.Upscalemaxinequality.eval()
        upscale_model = ctrl.par.Upscalemodel.eval().strip()
        if upscale_model:
            params["upscale_model"] = upscale_model

    ip_model = ctrl.par.Ipmodel.eval().strip() if hasattr(ctrl.par, "Ipmodel") else ""
    if ip_model and ip_path:
        params["ipadapter_model"] = ip_model

    return params


def push_params(force=False):
    global _last_payload, _last_push_at
    import time

    now = time.monotonic()
    if not force and now - _last_push_at < _push_debounce_s:
        return

    params = build_params()
    payload = {"pipeline": "streamdiffusion", "params": params}
    encoded = json.dumps(payload, sort_keys=True)
    if not force and encoded == _last_payload:
        return
    _last_payload = encoded
    _last_push_at = now

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


def onValueChange(par, prev):
    if par.name == "Prompt":
        global _last_prompt
        ctrl = _control()
        if ctrl is None:
            return
        text = ctrl.par.Prompt.eval().strip()
        if text == _last_prompt:
            return
        _last_prompt = text

    tokens = (
        "prompt",
        "negative",
        "denoise",
        "step",
        "guidance",
        "delta",
        "seed",
        "preset",
        "modelid",
        "acceleration",
        "sdmode",
        "width",
        "height",
        "framebatch",
        "fluxtransformer",
        "filter",
        "pause",
        "lora",
        "tinyvae",
        "vaeid",
        "ipimage",
        "ipscale",
        "ipmodel",
        "controlnet",
        "promptinterp",
        "upscale",
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

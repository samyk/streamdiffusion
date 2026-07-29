const output = document.getElementById("output");
const camera = document.getElementById("camera");
const capture = document.getElementById("capture");
const statusEl = document.getElementById("status");
const livePrompt = document.getElementById("live-prompt");
const livePromptInline = document.getElementById("live-prompt-inline");
const customForm = document.getElementById("custom-form");
const customPrompt = document.getElementById("custom-prompt");
const clearPromptBtn = document.getElementById("clear-prompt");
const startBtn = document.getElementById("start");
const flipCameraBtn = document.getElementById("flip-camera");
const fullscreenBtn = document.getElementById("fullscreen");
const prevBtn = document.getElementById("prev-prompt");
const nextBtn = document.getElementById("next-prompt");
const toggleUiBtn = document.getElementById("toggle-ui");
const overlay = document.getElementById("overlay");
const denoise = document.getElementById("denoise");
const denoiseValue = document.getElementById("denoise-value");
const denoiseReset = document.getElementById("denoise-reset");
const hideControlsBtn = document.getElementById("hide-controls");

const state = {
  streamId: "remote-1",
  publicUrl: null,
  inferWidth: 960,
  inferHeight: 536,
  prompts: [],
  promptIndex: 0,
  currentPrompt: "",
  customPromptLocked: false,
  ws: null,
  running: false,
  captureTimer: null,
  promptTimer: null,
  denoiseTimer: null,
  jpegQuality: 0.58,
  targetFps: 8,
  minFps: 4,
  maxFps: 14,
  goodFrames: 0,
  encodingFrame: false,
  maxBufferedBytes: 512 * 1024,
  denoiseLow: 15,
  denoiseHigh: 30,
  denoiseAuto: true,
  denoisePeriodMs: 15000,
  lastDenoiseSent: null,
  pendingDenoise: null,
  denoiseSendTimer: null,
  facingMode: "environment",
  isMobile: /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent),
  mirrorInput: !/Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent),
  previewMinWidth: 72,
  previewMaxWidth: 360,
};

function setStatus(text) {
  statusEl.textContent = text;
}

function setOverlayHidden(hidden) {
  overlay.classList.toggle("hidden", hidden);
  document.body.classList.toggle("controls-hidden", hidden);
  toggleUiBtn.textContent = hidden ? "Configure" : "Hide";
  toggleUiBtn.setAttribute("aria-label", hidden ? "Configure controls" : "Hide controls");
}

function setLivePrompt(text) {
  livePrompt.textContent = text;
  livePromptInline.textContent = text;
}

function wsUrl() {
  const base = state.publicUrl ? new URL(state.publicUrl) : location;
  const proto = base.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${base.host}/ws`;
}

async function loadConfig() {
  const [configRes, promptsRes] = await Promise.all([
    fetch("/api/config"),
    fetch("/api/prompts"),
  ]);
  const config = await configRes.json();
  const promptsPayload = await promptsRes.json();
  state.streamId = config.stream_id;
  state.publicUrl = config.public_url || null;
  state.inferWidth = config.infer_width;
  state.inferHeight = config.infer_height;
  state.currentPrompt = (config.prompt || "").trim();
  state.prompts = Array.isArray(promptsPayload.prompts) ? promptsPayload.prompts : [];
  if (state.prompts.length === 0 && config.prompt) {
    state.prompts = [config.prompt];
  }
  if (state.prompts.length === 0) {
    state.prompts = ["abstract art"];
  }
  const liveIndex = state.prompts.indexOf(state.currentPrompt);
  if (state.currentPrompt && liveIndex < 0) {
    state.customPromptLocked = true;
    clearPromptBtn.hidden = false;
    customPrompt.value = state.currentPrompt;
    setLivePrompt(state.currentPrompt);
  } else {
    state.customPromptLocked = false;
    clearPromptBtn.hidden = true;
    customPrompt.value = "";
    state.promptIndex = liveIndex >= 0 ? liveIndex : state.promptIndex;
    setLivePrompt(state.prompts[state.promptIndex] || "");
  }
}

async function patchParams(params) {
  const res = await fetch(`/v1/streams/${encodeURIComponent(state.streamId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ params }),
  });
  if (!res.ok) throw new Error(`PATCH failed: ${res.status}`);
}

async function applyPrompt(prompt, { custom = false } = {}) {
  const trimmed = prompt.trim();
  if (!trimmed) return;
  state.currentPrompt = trimmed;
  setLivePrompt(trimmed);
  await patchParams({ prompt: trimmed, custom_prompt: custom });
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "set_prompt", prompt: trimmed }));
  }
}

function rotatePrompt(delta = 1) {
  if (!state.running) return;
  if (state.customPromptLocked) return;
  if (!state.prompts.length) return;
  state.promptIndex = (state.promptIndex + delta + state.prompts.length) % state.prompts.length;
  applyPrompt(state.prompts[state.promptIndex]).catch(console.error);
}

function startPromptAutorotate() {
  clearInterval(state.promptTimer);
  state.promptTimer = window.setInterval(() => rotatePrompt(1), 10000);
}

function stopLiveLoops() {
  clearTimeout(state.captureTimer);
  clearTimeout(state.denoiseSendTimer);
  clearInterval(state.promptTimer);
  clearInterval(state.denoiseTimer);
  state.denoiseSendTimer = null;
  state.pendingDenoise = null;
  state.encodingFrame = false;
}

function normalizeDenoise(value) {
  return Math.max(0, Math.min(60, Math.round(Number(value) || 22)));
}

async function applyDenoise(value, { force = false } = {}) {
  const step = normalizeDenoise(value);
  const backendStep = Math.max(1, Math.min(49, step));
  denoise.value = String(step);
  denoiseValue.value = String(step);
  if (!force && step === state.lastDenoiseSent) return;
  if (force) {
    state.lastDenoiseSent = step;
    await patchParams({ t_index_list: [backendStep] });
    return;
  }
  state.pendingDenoise = step;
  if (state.denoiseSendTimer) return;
  state.denoiseSendTimer = window.setTimeout(() => flushDenoise().catch(console.error), 150);
}

async function flushDenoise() {
  const step = state.pendingDenoise;
  state.pendingDenoise = null;
  state.denoiseSendTimer = null;
  if (step == null || step === state.lastDenoiseSent) return;
  state.lastDenoiseSent = step;
  const backendStep = Math.max(1, Math.min(49, step));
  await patchParams({ t_index_list: [backendStep] });
  if (state.pendingDenoise != null) {
    state.denoiseSendTimer = window.setTimeout(() => flushDenoise().catch(console.error), 150);
  }
}

function startDenoiseAutorange() {
  clearInterval(state.denoiseTimer);
  const low = normalizeDenoise(state.denoiseLow);
  const high = normalizeDenoise(state.denoiseHigh);
  let step = normalizeDenoise(denoise.value);
  let direction = step >= high ? -1 : 1;

  state.denoiseTimer = window.setInterval(() => {
    if (!state.running) return;
    if (!state.denoiseAuto) return;
    applyDenoise(step).catch(console.error);
    if (step >= high) direction = -1;
    if (step <= low) direction = 1;
    step += direction;
  }, Math.max(250, Math.round(state.denoisePeriodMs / Math.max(1, high - low))));
}

async function enterFullscreen() {
  const el = document.documentElement;
  if (document.fullscreenElement || !el.requestFullscreen) return;
  try {
    await el.requestFullscreen({ navigationUI: "hide" });
  } catch (err) {
    console.debug("fullscreen unavailable", err);
  }
}

function connectSocket() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl());
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      state.ws = ws;
      resolve(ws);
    };
    ws.onerror = () => reject(new Error(`WebSocket failed: ${wsUrl()}`));
    ws.onclose = () => {
      if (state.running) setStatus("Disconnected - tap Start to reconnect");
      state.ws = null;
      state.running = false;
      stopLiveLoops();
      startBtn.classList.remove("running");
    };
    ws.onmessage = (event) => {
      if (!(event.data instanceof ArrayBuffer)) return;
      const blob = new Blob([event.data], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);
      if (output.dataset.objectUrl) URL.revokeObjectURL(output.dataset.objectUrl);
      output.dataset.objectUrl = url;
      output.src = url;
      output.classList.add("ready");
    };
  });
}

function stopCamera() {
  if (!camera.srcObject) return;
  for (const track of camera.srcObject.getTracks()) {
    track.stop();
  }
  camera.srcObject = null;
}

async function startCamera() {
  stopCamera();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: { ideal: state.facingMode },
      width: { ideal: 1920 },
      height: { ideal: 1080 },
    },
  });
  camera.srcObject = stream;
  camera.classList.toggle("mirrored", state.mirrorInput);
  await camera.play();
}

async function flipCamera() {
  state.facingMode = state.facingMode === "environment" ? "user" : "environment";
  flipCameraBtn.textContent = state.facingMode === "environment" ? "Flip" : "Back";
  if (!state.running) return;
  try {
    await startCamera();
  } catch (err) {
    console.error(err);
    setStatus("Could not switch camera");
  }
}

function startCaptureLoop() {
  const ctx = capture.getContext("2d", { alpha: false });
  clearTimeout(state.captureTimer);

  const noteBackpressure = () => {
    state.goodFrames = 0;
    state.targetFps = Math.max(state.minFps, state.targetFps - 1);
  };

  const noteSentFrame = () => {
    if (!state.ws || state.ws.bufferedAmount > state.maxBufferedBytes / 4) return;
    state.goodFrames += 1;
    if (state.goodFrames >= state.targetFps * 4) {
      state.goodFrames = 0;
      state.targetFps = Math.min(state.maxFps, state.targetFps + 1);
    }
  };

  const schedule = () => {
    const intervalMs = Math.max(33, Math.round(1000 / state.targetFps));
    state.captureTimer = window.setTimeout(tick, intervalMs);
  };

  const tick = () => {
    if (!state.running || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    if (state.encodingFrame || state.ws.bufferedAmount > state.maxBufferedBytes) {
      noteBackpressure();
      schedule();
      return;
    }
    const vw = camera.videoWidth;
    const vh = camera.videoHeight;
    if (!vw || !vh) {
      schedule();
      return;
    }

    capture.width = state.inferWidth;
    capture.height = state.inferHeight;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, state.inferWidth, state.inferHeight);

    const targetAspect = state.inferWidth / state.inferHeight;
    const sourceAspect = vw / vh;
    let dw = state.inferWidth;
    let dh = state.inferHeight;
    let dx = 0;
    let dy = 0;
    if (sourceAspect > targetAspect) {
      dh = Math.round(state.inferWidth / sourceAspect);
      dy = Math.round((state.inferHeight - dh) / 2);
    } else {
      dw = Math.round(state.inferHeight * sourceAspect);
      dx = Math.round((state.inferWidth - dw) / 2);
    }

    if (state.mirrorInput) {
      ctx.save();
      ctx.translate(state.inferWidth, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(camera, 0, 0, vw, vh, dx, dy, dw, dh);
      ctx.restore();
    } else {
      ctx.drawImage(camera, 0, 0, vw, vh, dx, dy, dw, dh);
    }
    state.encodingFrame = true;
    capture.toBlob(
      (blob) => {
        if (!blob || !state.ws || state.ws.readyState !== WebSocket.OPEN) {
          state.encodingFrame = false;
          return;
        }
        blob
          .arrayBuffer()
          .then((buf) => {
            if (state.ws && state.ws.readyState === WebSocket.OPEN) {
              state.ws.send(buf);
              noteSentFrame();
            }
          })
          .finally(() => {
            state.encodingFrame = false;
          });
      },
      "image/jpeg",
      state.jpegQuality,
    );
    schedule();
  };

  schedule();
}

async function start() {
  if (state.running) return;
  startBtn.disabled = true;
  try {
    if (state.isMobile) {
      await enterFullscreen();
    }
    await loadConfig();
    await startCamera();
    await connectSocket();
    state.running = true;
    startBtn.classList.add("running");
    startCaptureLoop();
    startPromptAutorotate();
    startDenoiseAutorange();
    setStatus("");
    await applyPrompt(state.currentPrompt || state.prompts[state.promptIndex]);
    await applyDenoise(denoise.value, { force: true });
    window.setTimeout(() => setOverlayHidden(true), 2500);
  } catch (err) {
    console.error(err);
    setStatus(err.message || "Failed to start (camera needs HTTPS on remote phones)");
    startBtn.classList.remove("running");
    state.running = false;
  } finally {
    startBtn.disabled = false;
  }
}

prevBtn.addEventListener("click", () => rotatePrompt(-1));
nextBtn.addEventListener("click", () => rotatePrompt(1));

customForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = customPrompt.value.trim();
  if (!prompt) return;
  state.customPromptLocked = true;
  clearPromptBtn.hidden = false;
  applyPrompt(prompt, { custom: true }).catch(console.error);
});

clearPromptBtn.addEventListener("click", () => {
  state.customPromptLocked = false;
  clearPromptBtn.hidden = true;
  customPrompt.value = "";
  patchParams({ clear_custom_prompt: true }).catch(console.error);
  applyPrompt(state.prompts[state.promptIndex] || "").catch(console.error);
  startPromptAutorotate();
});

denoise.addEventListener("input", () => {
  state.denoiseAuto = false;
  denoiseReset.hidden = false;
  clearInterval(state.denoiseTimer);
  applyDenoise(denoise.value).catch(console.error);
});

denoiseReset.addEventListener("click", () => {
  state.denoiseAuto = true;
  denoiseReset.hidden = true;
  startDenoiseAutorange();
});

camera.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  camera.setPointerCapture(event.pointerId);
});

camera.addEventListener("pointermove", (event) => {
  if (!camera.hasPointerCapture(event.pointerId)) return;
  const margin = 10;
  const maxWidth = Math.min(state.previewMaxWidth, window.innerWidth * 0.42);
  const nextWidth = Math.max(
    state.previewMinWidth,
    Math.min(maxWidth, window.innerWidth - event.clientX - margin),
  );
  camera.style.width = `${nextWidth}px`;
});

camera.addEventListener("pointerup", (event) => {
  if (camera.hasPointerCapture(event.pointerId)) {
    camera.releasePointerCapture(event.pointerId);
  }
});

startBtn.addEventListener("click", start);
flipCameraBtn.addEventListener("click", flipCamera);
fullscreenBtn.addEventListener("click", enterFullscreen);
hideControlsBtn.addEventListener("click", () => setOverlayHidden(true));
toggleUiBtn.addEventListener("click", () => {
  setOverlayHidden(!overlay.classList.contains("hidden"));
});

loadConfig().catch(() => {
  setStatus("Could not load server config");
});

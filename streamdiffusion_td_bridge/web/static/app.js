const output = document.getElementById("output");
const streamerPreview = document.getElementById("streamer-preview");
const camera = document.getElementById("camera");
const capture = document.getElementById("capture");
const statusEl = document.getElementById("status");
const livePrompt = document.getElementById("live-prompt");
const livePromptInline = document.getElementById("live-prompt-inline");
const customForm = document.getElementById("custom-form");
const customPrompt = document.getElementById("custom-prompt");
const promptHistoryList = document.getElementById("prompt-history");
const clearPromptBtn = document.getElementById("clear-prompt");
const startBtn = document.getElementById("start");
const flipCameraBtn = document.getElementById("flip-camera");
const fullscreenBtn = document.getElementById("fullscreen");
const prevBtn = document.getElementById("prev-prompt");
const nextBtn = document.getElementById("next-prompt");
const toggleUiBtn = document.getElementById("toggle-ui");
const takeoverStreamBtn = document.getElementById("takeover-stream");
const viewerCountEl = document.getElementById("viewer-count");
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
  promptHistory: [],
  promptIndex: 0,
  currentPrompt: "",
  customPromptLocked: false,
  ws: null,
  wsRole: "viewer",
  running: false,
  producerActive: false,
  viewerCount: 0,
  captureTimer: null,
  promptTimer: null,
  denoiseTimer: null,
  jpegQuality: 0.58,
  targetFps: 12,
  minFps: 6,
  maxFps: 24,
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
  toggleUiBtn.textContent = hidden ? "Params" : "Hide";
  toggleUiBtn.setAttribute("aria-label", hidden ? "Show params" : "Hide params");
  updateTakeoverUi();
}

function setLivePrompt(text) {
  livePrompt.textContent = text;
  livePromptInline.textContent = text;
}

function updateTakeoverUi() {
  const canTakeOver = !state.running && state.producerActive;
  document.body.classList.toggle("producing", state.running);
  document.body.classList.toggle("producer-active", canTakeOver);
  startBtn.textContent = canTakeOver ? "Take over stream" : "Start";
  takeoverStreamBtn.hidden = !canTakeOver || !document.body.classList.contains("controls-hidden");
  if (!state.producerActive) {
    streamerPreview.removeAttribute("src");
    streamerPreview.classList.remove("ready");
  }
}

function updateViewerCount() {
  const count = Math.max(0, Number(state.viewerCount) || 0);
  viewerCountEl.hidden = !state.running;
  viewerCountEl.textContent = `${count} ${count === 1 ? "viewer" : "viewers"}`;
  viewerCountEl.classList.toggle("active", count > 0);
}

function dedupePrompts(prompts) {
  const seen = new Set();
  const deduped = [];
  for (const prompt of prompts) {
    const text = String(prompt || "").trim();
    const key = text.toLowerCase();
    if (!text || seen.has(key)) continue;
    seen.add(key);
    deduped.push(text);
  }
  return deduped;
}

function setPromptHistory(prompts) {
  state.promptHistory = dedupePrompts(prompts).slice(0, 100);
  promptHistoryList.replaceChildren(
    (() => {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "Recent";
      return option;
    })(),
    ...state.promptHistory.map((prompt) => {
      const option = document.createElement("option");
      option.value = prompt;
      option.textContent = prompt;
      return option;
    }),
  );
  promptHistoryList.value = "";
}

function rememberPrompt(prompt) {
  setPromptHistory([prompt, ...state.promptHistory, ...state.prompts]);
}

function wsUrl(role = "viewer") {
  const base = state.publicUrl ? new URL(state.publicUrl) : location;
  const proto = base.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${base.host}/ws?role=${encodeURIComponent(role)}`;
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
  state.producerActive = Boolean(config.producer_active);
  state.viewerCount = Number(config.viewer_count) || 0;
  state.prompts = Array.isArray(promptsPayload.prompts) ? promptsPayload.prompts : [];
  setPromptHistory([
    ...(Array.isArray(config.prompt_history) ? config.prompt_history : []),
    ...(Array.isArray(promptsPayload.history) ? promptsPayload.history : []),
    ...state.prompts,
  ]);
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
  updateTakeoverUi();
  updateViewerCount();
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
  rememberPrompt(trimmed);
  await patchParams({ prompt: trimmed, custom_prompt: custom });
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
    await patchParams({ t_index_list: [backendStep], denoise_auto: state.denoiseAuto });
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
  await patchParams({ t_index_list: [backendStep], denoise_auto: state.denoiseAuto });
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

function connectSocket(role = "viewer") {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl(role));
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      state.ws = ws;
      state.wsRole = role;
      resolve(ws);
    };
    ws.onerror = () => reject(new Error(`WebSocket failed: ${wsUrl(role)}`));
    ws.onclose = () => {
      if (role !== "producer") {
        if (state.ws === ws) state.ws = null;
        return;
      }
      if (state.running) setStatus("Disconnected - tap Start to reconnect");
      state.ws = null;
      state.running = false;
      stopLiveLoops();
      stopCamera();
      startBtn.classList.remove("running");
      connectSocket("viewer")
        .then(() => loadConfig())
        .then(() => startCamera({ previewOnly: true }).catch(console.debug))
        .then(() => setStatus(""))
        .catch(() => setStatus("Disconnected - tap Start to reconnect"));
    };
    ws.onmessage = (event) => {
      if (typeof event.data === "string") {
        handleSocketEvent(event.data);
        return;
      }
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

function handleSocketEvent(raw) {
  let payload;
  try {
    payload = JSON.parse(raw);
  } catch {
    return;
  }
  if (payload.type === "producer_status") {
    state.producerActive = Boolean(payload.active);
    updateTakeoverUi();
  } else if (payload.type === "viewer_count") {
    state.viewerCount = Number(payload.count) || 0;
    updateViewerCount();
  } else if (payload.type === "params_update") {
    applyRemoteParams(payload);
  } else if (payload.type === "input_preview" && typeof payload.jpeg === "string") {
    if (state.running) return;
    streamerPreview.src = payload.jpeg;
    streamerPreview.classList.add("ready");
  }
}

function applyRemoteParams(payload) {
  const params = payload.params && typeof payload.params === "object" ? payload.params : {};
  if (typeof params.prompt === "string" && Array.isArray(payload.prompt_history)) {
    setPromptHistory([...payload.prompt_history, ...state.prompts]);
  }
  if (params.clear_custom_prompt) {
    state.customPromptLocked = false;
    clearPromptBtn.hidden = true;
  }
  if (typeof params.prompt === "string" && params.prompt.trim()) {
    const prompt = params.prompt.trim();
    state.currentPrompt = prompt;
    setLivePrompt(prompt);
    rememberPrompt(prompt);
    const promptIndex = state.prompts.indexOf(prompt);
    if (promptIndex >= 0) state.promptIndex = promptIndex;
    state.customPromptLocked = Boolean(params.custom_prompt) || promptIndex < 0;
    clearPromptBtn.hidden = !state.customPromptLocked;
    if (state.customPromptLocked) {
      customPrompt.value = prompt;
    }
  }
  if (typeof params.denoise_auto === "boolean") {
    const wasDenoiseAuto = state.denoiseAuto;
    state.denoiseAuto = params.denoise_auto;
    denoiseReset.hidden = state.denoiseAuto;
    if (!state.denoiseAuto) {
      clearInterval(state.denoiseTimer);
    } else if (!wasDenoiseAuto && state.running) {
      startDenoiseAutorange();
    }
  }
  if (Array.isArray(params.t_index_list) && params.t_index_list.length > 0) {
    if (state.running && params.denoise_auto !== false) return;
    const step = normalizeDenoise(params.t_index_list[0]);
    state.lastDenoiseSent = step;
    denoise.value = String(step);
    denoiseValue.value = String(step);
  }
}

function stopCamera() {
  if (!camera.srcObject) return;
  for (const track of camera.srcObject.getTracks()) {
    track.stop();
  }
  camera.srcObject = null;
}

async function startCamera({ previewOnly = false } = {}) {
  stopCamera();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: { ideal: state.facingMode },
      width: { ideal: previewOnly ? 480 : 1920 },
      height: { ideal: previewOnly ? 360 : 1080 },
    },
  });
  camera.srcObject = stream;
  camera.classList.toggle("mirrored", state.mirrorInput);
  await camera.play();
}

async function flipCamera() {
  state.facingMode = state.facingMode === "environment" ? "user" : "environment";
  flipCameraBtn.textContent = state.facingMode === "environment" ? "Flip" : "Back";
  try {
    await startCamera({ previewOnly: !state.running });
  } catch (err) {
    console.error(err);
    setStatus("Could not switch camera");
  }
}

function startCaptureLoop() {
  const ctx = capture.getContext("2d", { alpha: false });
  clearTimeout(state.captureTimer);

  const drawCamera = (dx, dy, dw, dh) => {
    if (state.mirrorInput) {
      ctx.save();
      ctx.translate(state.inferWidth, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(
        camera,
        0,
        0,
        camera.videoWidth,
        camera.videoHeight,
        dx,
        dy,
        dw,
        dh,
      );
      ctx.restore();
    } else {
      ctx.drawImage(
        camera,
        0,
        0,
        camera.videoWidth,
        camera.videoHeight,
        dx,
        dy,
        dw,
        dh,
      );
    }
  };

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

    const targetAspect = state.inferWidth / state.inferHeight;
    const sourceAspect = vw / vh;
    let coverW = state.inferWidth;
    let coverH = state.inferHeight;
    let coverX = 0;
    let coverY = 0;
    if (sourceAspect > targetAspect) {
      coverW = Math.round(state.inferHeight * sourceAspect);
      coverX = Math.round((state.inferWidth - coverW) / 2);
    } else {
      coverH = Math.round(state.inferWidth / sourceAspect);
      coverY = Math.round((state.inferHeight - coverH) / 2);
    }

    ctx.globalAlpha = 0.65;
    drawCamera(coverX, coverY, coverW, coverH);
    ctx.globalAlpha = 1;

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

    drawCamera(dx, dy, dw, dh);
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
    await startCamera({ previewOnly: false });
    if (state.ws) {
      state.ws.close();
      state.ws = null;
    }
    await connectSocket("producer");
    state.running = true;
    state.producerActive = true;
    startBtn.classList.add("running");
    updateTakeoverUi();
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
    updateTakeoverUi();
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

promptHistoryList.addEventListener("change", () => {
  const prompt = promptHistoryList.value.trim();
  promptHistoryList.value = "";
  if (!prompt) return;
  customPrompt.value = prompt;
  state.customPromptLocked = true;
  clearPromptBtn.hidden = false;
  applyPrompt(prompt, { custom: true }).catch(console.error);
});

clearPromptBtn.addEventListener("click", () => {
  state.customPromptLocked = false;
  clearPromptBtn.hidden = true;
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
  patchParams({ denoise_auto: true }).catch(console.error);
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
takeoverStreamBtn.addEventListener("click", start);
flipCameraBtn.addEventListener("click", flipCamera);
fullscreenBtn.addEventListener("click", enterFullscreen);
hideControlsBtn.addEventListener("click", () => setOverlayHidden(true));
toggleUiBtn.addEventListener("click", () => {
  setOverlayHidden(!overlay.classList.contains("hidden"));
});

loadConfig()
  .then(() => connectSocket("viewer"))
  .then(() => startCamera({ previewOnly: true }).catch(console.debug))
  .catch(() => {
    setStatus("Could not load server config");
  });

/* ============================================================
   Skin Condition Classifier - frontend logic
   Vanilla JS, no framework. Manages three screens (capture, results,
   info), webcam capture, file upload, and the /api/predict call.
   ============================================================ */

// ---------- Per-condition educational content ----------
// Baked into the JS so the info screen renders instantly with no extra API call.
// Sourced from public AAD / Mayo / NHS skin-health pages. Educational, not medical advice.
const CONDITION_INFO = {
  akiec: {
    title: "Actinic keratosis",
    tagline: "Pre-cancerous. Caused by long-term sun damage.",
    about: "Rough, scaly patches caused by years of sun exposure. Considered pre-cancerous: a small fraction progress to squamous cell carcinoma if left untreated. Common on the face, scalp, ears, hands, and forearms.",
    signs: [
      "Dry, scaly patches that often feel like sandpaper.",
      "Pink, red, or skin-coloured.",
      "Usually a few millimetres across; flat or slightly raised.",
      "Most often on sun-exposed skin in older adults.",
    ],
    doctor: "Any persistent rough, scaly patch on sun-exposed skin should be evaluated. Early treatment is straightforward.",
    care: [
      "Broad-spectrum SPF 30+ daily, even on cloudy days.",
      "Sun-protective clothing, hat, sunglasses.",
      "Avoid peak-sun hours (roughly 10am to 4pm).",
      "Self-check your skin monthly.",
    ],
  },
  bcc: {
    title: "Basal cell carcinoma",
    tagline: "Most common skin cancer. Slow-growing, almost always curable when caught early.",
    about: "The most common form of skin cancer. It grows slowly and rarely spreads, but it can damage surrounding skin and tissue if ignored. Driven by cumulative UV exposure.",
    signs: [
      "Pearly or waxy bump, often pink or skin-coloured.",
      "Visible small blood vessels on the surface.",
      "A sore that bleeds, scabs, and doesn't heal.",
      "A flat, scaly reddish patch (on the chest or back).",
    ],
    doctor: "Any sore that doesn't heal within a few weeks, or a new pearly bump, should be checked promptly. Caught early, treatment is usually a quick outpatient procedure.",
    care: [
      "Daily broad-spectrum SPF 30+ for life.",
      "Avoid tanning beds entirely.",
      "Annual full-body dermatologist check, especially after one BCC.",
      "Self-check monthly: any new bump or non-healing sore deserves a look.",
    ],
  },
  bkl: {
    title: "Benign keratosis",
    tagline: "Harmless growths that appear with age. Not cancerous.",
    about: "Common, harmless growths that appear with age. Sometimes called age spots or barnacles. They can look alarming but are not cancerous.",
    signs: [
      "Waxy, stuck-on-looking patches.",
      "Brown, black, or tan, sometimes mixed colours.",
      "Can be flat or raised; surface often slightly rough.",
      "Common on chest, back, shoulders, scalp.",
    ],
    doctor: "If a spot changes rapidly, bleeds without obvious cause, or you can't tell it apart from a melanoma, have it checked. Otherwise no action needed.",
    care: [
      "No medical treatment required.",
      "If a spot is irritated by clothing or jewellery, a dermatologist can remove it.",
      "Sun protection slows the appearance of new ones.",
    ],
  },
  df: {
    title: "Dermatofibroma",
    tagline: "Benign firm nodule, often on the legs. Frequently follows a minor skin injury.",
    about: "A small firm nodule under the skin, often where there was a minor injury such as an insect bite, ingrown hair, or shaving nick. Benign and very common, especially on the legs of adults.",
    signs: [
      "Small, firm, brown / pink / red bump.",
      "Dimples inward when squeezed from the sides (the dimple sign).",
      "Usually 5 to 10 mm across.",
      "Most often on the legs.",
    ],
    doctor: "If it changes size or colour, becomes painful, or you are unsure whether it is a dermatofibroma at all, have it checked.",
    care: [
      "No treatment needed unless it bothers you.",
      "Don't pick at it - that can cause irritation or scarring.",
      "Removal is possible but leaves a small scar; usually only done if symptomatic.",
    ],
  },
  mel: {
    title: "Melanoma",
    tagline: "Serious skin cancer. Highly treatable when caught early. Don't wait.",
    about: "A serious form of skin cancer that starts in the pigment-producing cells (melanocytes). Can spread quickly if not caught early. Early detection dramatically improves outcomes - this is the one that matters most.",
    signs: [
      "A new or changing mole is the single most important sign.",
      "ABCDE rule: Asymmetric, irregular Border, multiple Colours, Diameter over 6mm, Evolving.",
      "A mole that bleeds, itches, or scabs.",
      "A spot that looks different from the rest of your moles (the ugly duckling sign).",
    ],
    doctor: "As soon as possible - days, not months. If you suspect melanoma, ask for an urgent dermatology appointment. Early treatment can be curative; delay is the real danger.",
    care: [
      "Monthly skin self-exam - photograph moles to track changes.",
      "Annual full-body dermatologist check if you have risk factors.",
      "Strict sun protection: SPF 30+, hat, sun-protective clothing, no tanning beds.",
      "If you've had one melanoma, follow-up screening is essential.",
    ],
  },
  nv: {
    title: "Melanocytic nevus",
    tagline: "Common mole. Usually harmless. Watch for change over time.",
    about: "A common cluster of pigment cells. Most adults have between ten and forty. Usually harmless, but moles can occasionally evolve into melanoma - so monitoring matters more than the mole itself.",
    signs: [
      "Round, even-coloured, smaller than 6mm.",
      "Brown, tan, or skin-coloured.",
      "Stable over time - change is the warning sign.",
    ],
    doctor: "If a mole changes shape, colour, or size; bleeds; itches; or starts looking different from your other moles. The ABCDE rule for melanoma applies here too.",
    care: [
      "Self-check monthly - photographs help spot slow changes.",
      "Use sun protection: SPF, hat, shade.",
      "If you have many moles or atypical ones, see a dermatologist yearly.",
      "Don't try to remove moles yourself.",
    ],
  },
  vasc: {
    title: "Vascular lesion",
    tagline: "Marks involving blood vessels. Almost always benign.",
    about: "A broad category for skin marks involving blood vessels - cherry angiomas, spider veins, hemangiomas. Almost always benign and a normal part of ageing skin.",
    signs: [
      "Bright red, purple, or blue.",
      "Usually small (a few millimetres), raised, smooth.",
      "May briefly turn pale when pressed.",
    ],
    doctor: "If it grows rapidly, bleeds excessively when bumped, or is in a problematic spot, have it checked. Most don't need any treatment.",
    care: [
      "No medical treatment usually needed.",
      "Sun protection helps prevent new ones.",
      "Cosmetic removal with lasers or electrocautery is an option.",
    ],
  },
};

// ---------- DOM refs ----------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const screens = {
  capture: $('[data-screen="capture"]'),
  results: $('[data-screen="results"]'),
  info: $('[data-screen="info"]'),
  history: $('[data-screen="history"]'),
};

const cameraVideo = $("#camera-video");
const previewImage = $("#preview-image");
const placeholder = $("#capture-placeholder");
const captureStage = $("#capture-stage");
const captureCanvas = $("#capture-canvas");
const fileInput = $("#file-input");

const btnCamera = $("#btn-camera");
const btnUpload = $("#btn-upload");
const btnClassify = $("#btn-classify");
const btnSwitchCamera = $("#btn-switch-camera");
const btnInfo = $("#btn-info");
const btnBack = $("#btn-back");
const btnRetry1 = $("#btn-retry-1");
const btnRetry2 = $("#btn-retry-2");

const preflightAlert = $("#preflight-alert");
const preflightReason = $("#preflight-reason");
const btnDismissPreflight = $("#btn-dismiss-preflight");

const chatCard = $("#chat-card");
const chatMessages = $("#chat-messages");
const chatForm = $("#chat-form");
const chatInput = $("#chat-input");
const chatSend = $("#chat-send");

// ---------- State ----------
let cameraStream = null; // MediaStream when webcam is active
let capturedBlob = null; // The image to send to /api/predict
let lastResult = null; // Last /api/predict response
let conversation = []; // Chat history: [{role: "user"|"assistant", content: str}, ...]
let currentFacing = "environment"; // "environment" (rear) or "user" (front / selfie)

// ---------- Loading overlay ----------
function loading(on) {
  let el = $(".loading");
  if (!el) {
    el = document.createElement("div");
    el.className = "loading";
    el.innerHTML = '<div class="spinner"></div>';
    document.body.appendChild(el);
  }
  el.classList.toggle("active", !!on);
}

// ---------- Screen routing ----------
function showScreen(name) {
  Object.values(screens).forEach(s => s.classList.remove("active"));
  screens[name].classList.add("active");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ---------- Camera handling ----------
async function startCamera() {
  // Stop any existing stream first (matters when switching between front/rear).
  stopCamera();
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: currentFacing }, width: 1280, height: 960 },
      audio: false,
    });
    cameraVideo.srcObject = cameraStream;
    cameraVideo.classList.add("visible");
    previewImage.classList.remove("visible");
    placeholder.style.display = "none";
    btnCamera.textContent = "Take photo";
    btnClassify.disabled = true; // disabled until they take a frame
    btnSwitchCamera.hidden = false; // show the swap-camera control
    updateStageInteractive();
  } catch (err) {
    alert("Couldn't access the camera. " + (err.message || "") +
          "\n\nYou can still use \"Upload photo\" to pick an image.");
  }
}

async function switchCamera() {
  // Flip environment <-> user and restart the stream.
  currentFacing = currentFacing === "environment" ? "user" : "environment";
  if (cameraStream) {
    await startCamera();
  }
}

function takeSnapshot() {
  captureCanvas.width = cameraVideo.videoWidth;
  captureCanvas.height = cameraVideo.videoHeight;
  captureCanvas.getContext("2d").drawImage(cameraVideo, 0, 0);
  captureCanvas.toBlob((blob) => {
    if (!blob) return;
    capturedBlob = blob;
    const url = URL.createObjectURL(blob);
    previewImage.src = url;
    previewImage.classList.add("visible");
    cameraVideo.classList.remove("visible");
    stopCamera();
    btnCamera.textContent = "Use camera";
    btnSwitchCamera.hidden = true;
    btnClassify.disabled = false;
    updateStageInteractive();
  }, "image/jpeg", 0.92);
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  btnSwitchCamera.hidden = true;
}

// The capture stage acts as a big tap target when it's in placeholder state.
// Tapping starts the camera. When something is showing (video / photo), the
// stage is no longer a button - the on-screen controls take over.
function updateStageInteractive() {
  const isPlaceholder = !cameraStream && !capturedBlob;
  captureStage.classList.toggle("is-tappable", isPlaceholder);
  captureStage.setAttribute("aria-disabled", String(!isPlaceholder));
}

function handleStageTap(e) {
  // Don't hijack clicks on the swap-camera button
  if (e.target.closest(".camera-switch-btn")) return;
  if (cameraStream || capturedBlob) return;
  startCamera();
}

captureStage.addEventListener("click", handleStageTap);
captureStage.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    handleStageTap(e);
  }
});

btnCamera.addEventListener("click", () => {
  if (cameraStream) takeSnapshot();
  else startCamera();
});

btnSwitchCamera.addEventListener("click", switchCamera);

// Initialize the stage's interactive state on first load
updateStageInteractive();

// ---------- File upload ----------
btnUpload.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  capturedBlob = file;
  stopCamera();
  const url = URL.createObjectURL(file);
  previewImage.src = url;
  previewImage.classList.add("visible");
  cameraVideo.classList.remove("visible");
  placeholder.style.display = "none";
  btnCamera.textContent = "Use camera";
  btnClassify.disabled = false;
});

// ---------- Classify (preflight + predict pipeline) ----------
btnClassify.addEventListener("click", async () => {
  if (!capturedBlob) return;
  hidePreflightAlert();
  loading(true);
  try {
    // Step 1: photo quality preflight via Claude vision.
    // Fails open: if Claude is unavailable, this just returns ok=true.
    const preflight = await postImage("/api/preflight", capturedBlob);
    if (!preflight.ok) {
      showPreflightAlert(preflight.reason || "This photo isn't usable for classification.");
      return;
    }

    // Step 2: run the actual classifier.
    lastResult = await postImage("/api/predict", capturedBlob);
    renderResults(lastResult);
    resetChat();
    pushHistoryEntry(capturedBlob, lastResult).catch(() => {});
    showScreen("results");
  } catch (err) {
    alert("Classification failed: " + err.message);
  } finally {
    loading(false);
  }
});

async function postImage(url, blob) {
  const formData = new FormData();
  formData.append("file", blob, "photo.jpg");
  const res = await fetch(url, { method: "POST", body: formData });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Server returned ${res.status}`);
  }
  return res.json();
}

function showPreflightAlert(reason) {
  preflightReason.textContent = reason;
  preflightAlert.hidden = false;
}
function hidePreflightAlert() {
  preflightAlert.hidden = true;
}
btnDismissPreflight.addEventListener("click", hidePreflightAlert);

// ---------- Render results ----------
function renderResults(data) {
  const thumb = $("#results-thumb");
  thumb.src = URL.createObjectURL(capturedBlob);

  if (data.low_confidence) {
    renderCouldntIdentify(data);
  } else {
    renderConfidentResult(data);
  }
}

// Confident path - show top-3 bars + warning if malignant + grad-cam heatmap if available + chat
function renderConfidentResult(data) {
  const top1 = data.top_3[0];

  // Show the chat panel only on confident results - on low-confidence we don't have
  // a real classification to discuss.
  chatCard.hidden = false;

  // Show heatmap (Grad-CAM) next to the photo if the backend produced one
  const heatmapFrame = $("#heatmap-frame");
  const heatmapImg = $("#results-heatmap");
  if (data.heatmap) {
    heatmapImg.src = data.heatmap;
    heatmapFrame.hidden = false;
  } else {
    heatmapFrame.hidden = true;
  }

  // Show the More Info / Try Another row in normal layout (not low-conf)
  $("#btn-info").style.display = "";
  $("#btn-info").disabled = false;
  $("#btn-info").textContent = "More information";
  $("#btn-retry-1").classList.remove("btn-primary");
  $("#btn-retry-1").classList.add("btn-secondary");

  $("#top-guess").innerHTML = `
    <div class="label">Top guess</div>
    <h2 class="name">${escapeHtml(top1.name)}</h2>
    <p class="meta">
      <span class="tag ${top1.malignancy.replace(/\s/g, "-")}">${escapeHtml(top1.malignancy)}</span>
      ${(top1.probability * 100).toFixed(0)}% confidence
    </p>
  `;

  const list = $("#results-list");
  list.innerHTML = data.top_3.map(g => `
    <div class="guess-row">
      <div class="info">
        <div class="name-row">
          <span class="name">${escapeHtml(g.name)}</span>
          <span class="code">${escapeHtml(g.code)} &middot; ${escapeHtml(g.malignancy)}</span>
        </div>
        <div class="bar-track"><div class="bar-fill" style="width: 0%"></div></div>
      </div>
      <div class="pct">${(g.probability * 100).toFixed(0)}%</div>
    </div>
  `).join("");

  requestAnimationFrame(() => {
    $$(".bar-fill").forEach((bar, i) => {
      bar.style.width = (data.top_3[i].probability * 100).toFixed(1) + "%";
    });
  });

  const warning = $("#results-warning");
  const concerning = data.top_3.some(g =>
    g.malignancy === "cancerous" || g.malignancy === "pre-cancerous"
  );
  if (concerning) {
    warning.innerHTML = `
      <div>
        <strong>Heads up.</strong>
        The top guesses include a potentially malignant or pre-malignant category.
        False alarms are common, but this is a real reason to have it checked by a dermatologist.
      </div>
    `;
    warning.hidden = false;
  } else {
    warning.hidden = true;
  }
}

// Low-confidence path - no fake diagnosis, show tips for a better photo
function renderCouldntIdentify(data) {
  // Hide the heatmap and chat - both would be meaningless on a low-confidence result
  $("#heatmap-frame").hidden = true;
  chatCard.hidden = true;

  // Hide info button, promote "Try another photo" to primary
  $("#btn-info").style.display = "none";
  $("#btn-retry-1").classList.remove("btn-secondary");
  $("#btn-retry-1").classList.add("btn-primary");
  $("#btn-retry-1").textContent = "Take another photo";

  $("#top-guess").innerHTML = `
    <div class="label">No confident match</div>
    <h2 class="name">Couldn't identify this photo</h2>
    <p class="meta">
      The model isn't confident this matches any of the seven conditions it was trained on.
      That's most often a photo issue, not a skin issue.
    </p>
  `;

  $("#results-list").innerHTML = `
    <div class="tips">
      <h3>For a better classification, try this</h3>
      <ul>
        <li><strong>Use bright, even light.</strong> Natural daylight near a window works best. Avoid overhead shadows.</li>
        <li><strong>Get closer.</strong> The skin spot should fill most of the frame - not be a tiny dot.</li>
        <li><strong>Hold steady and focus.</strong> Blurry photos are the most common cause of low confidence.</li>
        <li><strong>No glare or wet skin.</strong> Reflective spots throw the model off.</li>
        <li><strong>Make sure it's a skin lesion.</strong> The model only knows seven dermatology categories - it can't classify acne, eczema, rosacea, or anything that isn't skin.</li>
      </ul>
    </div>
  `;

  $("#results-warning").hidden = true;
}

// ---------- Info screen ----------
btnInfo.addEventListener("click", () => {
  if (!lastResult || lastResult.low_confidence) return;
  const code = lastResult.top_3[0].code;
  renderInfo(code);
  showScreen("info");
});

function renderInfo(code) {
  const info = CONDITION_INFO[code];
  if (!info) return;
  $("#info-article").innerHTML = `
    <h1>${escapeHtml(info.title)}</h1>
    <p class="tagline">${escapeHtml(info.tagline)}</p>

    <h2>About</h2>
    <p>${escapeHtml(info.about)}</p>

    <h2>Common signs</h2>
    <ul>${info.signs.map(s => `<li>${escapeHtml(s)}</li>`).join("")}</ul>

    <h2>When to see a doctor</h2>
    <p>${escapeHtml(info.doctor)}</p>

    <h2>General care</h2>
    <ul>${info.care.map(c => `<li>${escapeHtml(c)}</li>`).join("")}</ul>

    <p class="disclaimer">
      General educational information, not medical advice. If you're concerned about
      something on your skin, see a dermatologist.
    </p>
  `;
}

// ---------- Navigation ----------
btnBack.addEventListener("click", () => showScreen("results"));

function resetCapture() {
  stopCamera();
  capturedBlob = null;
  lastResult = null;
  previewImage.classList.remove("visible");
  cameraVideo.classList.remove("visible");
  placeholder.style.display = "";
  btnCamera.textContent = "Use camera";
  btnClassify.disabled = true;
  fileInput.value = "";

  // Reset results-screen button state in case the previous result was low-confidence
  btnInfo.style.display = "";
  btnInfo.disabled = false;
  btnRetry1.classList.remove("btn-primary");
  btnRetry1.classList.add("btn-secondary");
  btnRetry1.textContent = "Try another photo";

  showScreen("capture");
}
btnRetry1.addEventListener("click", resetCapture);
btnRetry2.addEventListener("click", resetCapture);

// ---------- Chat (follow-up Q&A via /api/chat) ----------
function resetChat() {
  conversation = [];
  chatMessages.innerHTML = "";
  chatInput.value = "";
  chatSend.disabled = false;
}

function appendChatMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `chat-message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  chatMessages.appendChild(wrap);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return wrap;
}

function appendTypingIndicator() {
  const wrap = document.createElement("div");
  wrap.className = "chat-message assistant typing";
  wrap.innerHTML = `
    <div class="typing-indicator">
      <span></span><span></span><span></span>
    </div>
  `;
  chatMessages.appendChild(wrap);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return wrap;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text || !lastResult) return;

  appendChatMessage("user", text);
  conversation.push({ role: "user", content: text });
  chatInput.value = "";
  chatSend.disabled = true;
  const typing = appendTypingIndicator();

  try {
    const top1 = lastResult.top_3[0];
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: conversation,
        context: {
          name: top1.name,
          code: top1.code,
          malignancy: top1.malignancy,
          probability: top1.probability,
        },
      }),
    });
    const data = await res.json();
    typing.remove();
    if (data.ok && data.reply) {
      conversation.push({ role: "assistant", content: data.reply });
      appendChatMessage("assistant", data.reply);
    } else {
      appendChatMessage("system", data.error || "Couldn't reach the chat service.");
    }
  } catch (err) {
    typing.remove();
    appendChatMessage("system", "Network error. Try again in a moment.");
  } finally {
    chatSend.disabled = false;
    chatInput.focus();
  }
});

// ---------- Tiny html-escape helper ----------
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ---------- Scan history (localStorage, device-only) ----------
const HISTORY_KEY = "konsei-history";
const HISTORY_MAX = 10;

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (_) { return []; }
}
function saveHistory(entries) {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, HISTORY_MAX))); } catch (_) {}
}

async function pushHistoryEntry(blob, result) {
  // Shrink to a tiny thumbnail so localStorage doesn't fill up
  const thumb = await blobToThumbnailDataUrl(blob, 160);
  const top1 = result.top_3[0];
  const entry = {
    ts: Date.now(),
    thumb,
    top1: { name: top1.name, code: top1.code, probability: top1.probability, malignancy: top1.malignancy },
    low_confidence: !!result.low_confidence,
  };
  const next = [entry, ...loadHistory()].slice(0, HISTORY_MAX);
  saveHistory(next);
}

function blobToThumbnailDataUrl(blob, size) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(size / img.width, size / img.height, 1);
      const w = Math.round(img.width * scale);
      const h = Math.round(img.height * scale);
      const c = document.createElement("canvas");
      c.width = w; c.height = h;
      c.getContext("2d").drawImage(img, 0, 0, w, h);
      resolve(c.toDataURL("image/jpeg", 0.72));
    };
    img.onerror = () => resolve("");
    img.src = URL.createObjectURL(blob);
  });
}

function renderHistory() {
  const list = $("#history-list");
  const empty = $("#history-empty");
  const entries = loadHistory();
  list.innerHTML = "";
  empty.hidden = entries.length > 0;
  for (const e of entries) {
    const li = document.createElement("li");
    li.className = "history-row";
    const date = new Date(e.ts).toLocaleString(undefined, {
      month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
    });
    const headline = e.low_confidence
      ? `<span class="history-headline muted">Couldn't identify</span>`
      : `<span class="history-headline">${escapeHtml(e.top1.name)}</span>
         <span class="history-pct">${Math.round(e.top1.probability * 100)}%</span>`;
    const tag = e.low_confidence
      ? ""
      : `<span class="history-tag ${e.top1.malignancy.replace(/\s/g, "-")}">${escapeHtml(e.top1.malignancy)}</span>`;
    li.innerHTML = `
      <img class="history-thumb" src="${e.thumb}" alt="scan thumbnail"/>
      <div class="history-meta">
        <div class="history-top-row">${headline}</div>
        <div class="history-bottom-row">${tag}<span class="history-date">${escapeHtml(date)}</span></div>
      </div>
    `;
    list.appendChild(li);
  }
}

$("#btn-history").addEventListener("click", () => { renderHistory(); showScreen("history"); });
$("#btn-history-back").addEventListener("click", () => showScreen("capture"));
$("#btn-history-clear").addEventListener("click", () => {
  if (!confirm("Clear all saved scans from this device? This can't be undone.")) return;
  saveHistory([]);
  renderHistory();
});

// ---------- Theme toggle (light / dark override) ----------
(function initTheme() {
  const btn = document.getElementById("btn-theme");
  const root = document.documentElement;
  const KEY = "konsei-theme";
  const saved = localStorage.getItem(KEY);
  if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);

  function current() {
    const override = root.getAttribute("data-theme");
    if (override) return override;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function apply(mode) {
    root.setAttribute("data-theme", mode);
    try { localStorage.setItem(KEY, mode); } catch (_) {}
  }
  if (btn) {
    btn.addEventListener("click", () => apply(current() === "dark" ? "light" : "dark"));
  }
})();

// ---------- Register the service worker (PWA install + offline cache) ----------
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch((err) => {
      console.warn("Service worker registration failed:", err);
    });
  });
}

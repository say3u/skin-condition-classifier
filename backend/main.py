"""
FastAPI service for the skin condition classifier.

Two responsibilities:
  1. Expose POST /api/predict - accepts an image, returns top-3 + low_confidence flag.
  2. Serve the static frontend at / (index.html + style.css + app.js).

Run locally:
    uvicorn backend.main:app --reload --port 7860

In production (Docker / HF Spaces): the Dockerfile runs uvicorn on 0.0.0.0:7860.
"""

import io
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend import claude as claude_api
from backend.gradcam import compute_gradcam_overlay
from backend.model import load_model, predict_top3

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
MODEL_PATH = ROOT / "skin_classifier_efficientnet_b0.pt"

app = FastAPI(
    title="KONSEI Skin Lab API",
    description="Educational only — not a diagnostic tool.",
    version="1.0.0",
)

# ---------- Rate limiting ----------
# Per-IP request caps to prevent someone scripting our endpoints and
# draining the Claude budget or hammering the model.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        {"detail": "Too many requests. Slow down and try again in a minute."},
        status_code=429,
    )


# ---------- Origin allowlist ----------
# Only accept API traffic that looks like it came from our own web app.
# Not bulletproof (headers can be spoofed) but blocks the naive curl / script abuse.
ALLOWED_ORIGIN_HOSTS = {
    "localhost",
    "127.0.0.1",
    "sayeu-skin-condition-classifier.hf.space",       # HF Space
    "huggingface.co",                                  # HF page iframe origin
    "konsei.app",                                      # custom domain
    "www.konsei.app",
}
# Allow overriding via env so this can be extended without a code change.
extra = os.environ.get("EXTRA_ORIGIN_HOSTS", "")
if extra:
    for host in [h.strip() for h in extra.split(",") if h.strip()]:
        ALLOWED_ORIGIN_HOSTS.add(host)


def _origin_host(request: Request) -> str | None:
    """Best-effort extraction of the request's origin host, from Origin then Referer."""
    origin = request.headers.get("origin") or request.headers.get("referer") or ""
    if not origin:
        return None
    try:
        return urlparse(origin).hostname
    except Exception:
        return None


def enforce_origin(request: Request):
    """Reject API calls that don't come from an allowlisted host."""
    host = _origin_host(request)
    if host is None or host not in ALLOWED_ORIGIN_HOSTS:
        raise HTTPException(
            status_code=403,
            detail="This API is only available from the KONSEI Skin Lab web app.",
        )


# Load the model once at startup, not per request.
bundle = load_model(MODEL_PATH)


@app.post("/api/predict")
@limiter.limit("15/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    """Accept an image, return the top-3 classification plus a Grad-CAM heatmap overlay."""
    enforce_origin(request)
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Expected an image, got {file.content_type}.")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty upload.")
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(413, "Image too large. Keep it under 12 MB.")

    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw))   # re-open after verify
    except (UnidentifiedImageError, Exception):
        raise HTTPException(400, "Could not read the image.")

    result = predict_top3(bundle, image)

    # Only compute the heatmap when the model is actually confident enough that the
    # explanation is meaningful. On low-confidence "couldn't identify" results, the
    # heatmap would be noise.
    if not result["low_confidence"]:
        try:
            top1_idx = bundle["classes"].index(result["top_3"][0]["code"])
            overlay, _ = compute_gradcam_overlay(bundle, image, target_class_idx=top1_idx)
            result["heatmap"] = overlay
        except Exception as exc:
            # Grad-CAM failure shouldn't break prediction. Log and continue.
            print(f"[gradcam] failed: {exc}")
            result["heatmap"] = None
    else:
        result["heatmap"] = None

    return result


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "classes": bundle["classes"],
        "claude_available": claude_api.is_available(),
    }


# ---------- Photo quality preflight (Claude vision) ----------

def _read_uploaded_image(file: UploadFile, raw: bytes) -> Image.Image:
    """Shared validation for both /api/predict and /api/preflight."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Expected an image, got {file.content_type}.")
    if not raw:
        raise HTTPException(400, "Empty upload.")
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(413, "Image too large. Keep it under 12 MB.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        return Image.open(io.BytesIO(raw))
    except (UnidentifiedImageError, Exception):
        raise HTTPException(400, "Could not read the image.")


@app.post("/api/preflight")
@limiter.limit("15/minute")
async def preflight(request: Request, file: UploadFile = File(...)):
    """
    Quick photo quality + subject check via Claude vision.

    Returns {ok, category, reason, used_claude}. If Claude is unavailable (no API key),
    used_claude=false and ok=true - the classifier's own low-confidence fallback handles it.
    """
    enforce_origin(request)
    raw = await file.read()
    image = _read_uploaded_image(file, raw)
    return claude_api.preflight_check(image)


# ---------- Follow-up Q&A chat (Claude text) ----------

@app.post("/api/chat")
@limiter.limit("20/minute")
async def chat(request: Request, payload: dict[str, Any] = Body(...)):
    """
    Stateless chat endpoint. The frontend tracks history and posts the full conversation
    each turn.

    Body:
      {
        "messages": [{"role": "user"|"assistant", "content": str}, ...],
        "context":  {"name": str, "code": str, "malignancy": str, "probability": float} | null
      }
    """
    enforce_origin(request)
    messages = payload.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "Body must include a non-empty 'messages' array.")
    context = payload.get("context")
    return claude_api.chat_reply(messages, context)


# ----- Static frontend -----
# Serve index.html at "/" and the other assets at their bare filenames.
@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


# Mount /static for any other assets we add later (images, fonts, etc.)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# Serve top-level frontend files (style.css, app.js, etc.) at their bare paths.
@app.get("/{filename:path}")
async def static_root(filename: str):
    if filename.startswith("api/"):
        raise HTTPException(404)
    candidate = FRONTEND_DIR / filename
    if candidate.is_file():
        return FileResponse(candidate)
    raise HTTPException(404)

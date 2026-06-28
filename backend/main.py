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
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

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

# Load the model once at startup, not per request.
bundle = load_model(MODEL_PATH)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Accept an image, return the top-3 classification plus a Grad-CAM heatmap overlay."""
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
async def preflight(file: UploadFile = File(...)):
    """
    Quick photo quality + subject check via Claude vision.

    Returns {ok, category, reason, used_claude}. If Claude is unavailable (no API key),
    used_claude=false and ok=true - the classifier's own low-confidence fallback handles it.
    """
    raw = await file.read()
    image = _read_uploaded_image(file, raw)
    return claude_api.preflight_check(image)


# ---------- Follow-up Q&A chat (Claude text) ----------

@app.post("/api/chat")
async def chat(payload: dict[str, Any] = Body(...)):
    """
    Stateless chat endpoint. The frontend tracks history and posts the full conversation
    each turn.

    Body:
      {
        "messages": [{"role": "user"|"assistant", "content": str}, ...],
        "context":  {"name": str, "code": str, "malignancy": str, "probability": float} | null
      }
    """
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

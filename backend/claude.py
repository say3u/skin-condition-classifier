"""
Claude API integration: photo quality preflight + follow-up Q&A chat.

Design principle: BOTH features fail open. If the ANTHROPIC_API_KEY is missing,
the SDK isn't installed, or the call errors, the app keeps working. Preflight
just passes through to the classifier; chat tells the user the feature is
unavailable in this deployment. Claude is an enhancement, not a hard dependency.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any

from PIL import Image

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

# Haiku is the right tier for these tasks - quick photo judgment and short Q&A.
# Cheap enough to run on a demo budget.
CLAUDE_MODEL = "claude-haiku-4-5-20251001"


def _client():
    """Return an Anthropic client if API key is set, else None."""
    if not _HAS_ANTHROPIC:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    return anthropic.Anthropic()


def is_available() -> bool:
    """True if Claude features can run (SDK installed + key set)."""
    return _client() is not None


def _image_to_jpeg_base64(image: Image.Image, max_dim: int = 768) -> str:
    """Resize and JPEG-encode for Claude. Smaller = faster + cheaper, plenty for quality judgment."""
    img = image.convert("RGB")
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------- Preflight ----------

def preflight_check(image: Image.Image) -> dict[str, Any]:
    """
    Quick vision-based photo quality check.

    Decides whether the uploaded image is even worth running through the
    classifier. Catches things the model would silently mis-classify
    (blurry photos, non-skin photos, extreme close-ups of clothing, etc.).

    Returns:
        {
          "ok":          bool,    # safe to pass to the classifier?
          "category":    str | None, "good" | "not_skin" | "too_blurry" | "too_dark" | "too_far" | "obstructed"
          "reason":      str | None, single-sentence plain-English explanation
          "used_claude": bool,    # false if we skipped Claude entirely (no key, etc.)
        }
    """
    client = _client()
    if client is None:
        # Fail open - let the classifier's own low-confidence fallback handle it.
        return {"ok": True, "category": None, "reason": None, "used_claude": False}

    prompt = (
        "You are a strict quality gatekeeper for an educational skin-lesion classifier. "
        "The downstream model was trained on close-up photos of human skin lesions "
        "(moles, growths, patches) and will produce confidently wrong results on anything else.\n\n"
        "Look at this image and decide if it is a usable input.\n\n"
        "Return ONLY a JSON object, no preamble, no markdown, no commentary:\n"
        "{\n"
        '  "ok": <true|false>,\n'
        '  "category": "<good|not_skin|too_blurry|too_dark|too_far|obstructed>",\n'
        '  "reason": "<one short sentence in plain English explaining your decision>"\n'
        "}\n\n"
        "Rules:\n"
        "- If the image is not skin (e.g. a wall, a face from far away, a screenshot, clothing), ok=false.\n"
        "- If the image is too blurry/dark/zoomed-out to see skin detail, ok=false.\n"
        "- If the image is a clear close-up of skin (with or without a visible lesion), ok=true.\n"
        "- If unsure, lean toward ok=false.\n"
        "- DO NOT diagnose, name, or assess any condition. Only judge image quality and subject."
    )

    try:
        img_b64 = _image_to_jpeg_base64(image)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = response.content[0].text.strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {"ok": True, "category": None, "reason": None, "used_claude": False}
        data = json.loads(match.group(0))
        return {
            "ok": bool(data.get("ok", True)),
            "category": data.get("category"),
            "reason": data.get("reason"),
            "used_claude": True,
        }
    except Exception as exc:
        print(f"[claude:preflight] failed, falling through: {exc!r}")
        return {"ok": True, "category": None, "reason": None, "used_claude": False}


# ---------- Chat ----------

CHAT_SYSTEM_PROMPT = (
    "You are a careful, friendly assistant inside an educational skin-lesion classifier. "
    "The user just got a classification result from a small ML model and may have follow-up questions.\n\n"
    "Hard rules:\n"
    "- You are NOT a doctor. You do NOT diagnose. You do NOT prescribe.\n"
    "- You do NOT confirm or refute the classifier's guess. Treat it as one statistical input.\n"
    "- You DO explain what a condition is in general, plain-English terms.\n"
    "- You DO explain what a dermatologist appointment typically involves.\n"
    "- You DO explain the ABCDE rule for monitoring moles, photographing skin changes, and "
    "  how to find a dermatologist.\n"
    "- For anything urgent, scary, or specific to the user's case, redirect to seeing a real dermatologist.\n\n"
    "Style:\n"
    "- 2-5 sentences. Calm, plain tone. Not clinical jargon.\n"
    "- Never use the words \"definitely\", \"certainly\", \"diagnose\", or \"safe to assume\"."
)


def chat_reply(messages: list[dict], context: dict | None = None) -> dict[str, Any]:
    """
    Get a single chat reply from Claude.

    Args:
        messages: conversation history as [{"role": "user"|"assistant", "content": str}, ...].
                  The most recent message is the user's current question.
        context:  optional last-classification context the assistant should be aware of:
                  {"name": str, "code": str, "malignancy": str, "probability": float (0..1)}

    Returns:
        {"ok": bool, "reply": str | None, "error": str | None}
    """
    client = _client()
    if client is None:
        return {
            "ok": False,
            "reply": None,
            "error": "Chat is unavailable: this deployment doesn't have an Anthropic API key configured.",
        }

    # Sanitize history
    clean_history = [
        {"role": m["role"], "content": str(m["content"])}
        for m in (messages or [])
        if m.get("role") in ("user", "assistant") and m.get("content")
    ][-10:]   # cap to last 10 turns

    if not clean_history:
        return {"ok": False, "reply": None, "error": "Empty message."}

    system = CHAT_SYSTEM_PROMPT
    if context:
        prob = context.get("probability") or 0
        system += (
            f"\n\nContext for this conversation: the classifier's top guess was "
            f"{context.get('name')} ({context.get('code')}), labelled "
            f"\"{context.get('malignancy')}\", with {int(prob * 100)}% confidence. "
            f"It is one statistical guess from a small model trained on ~10k images. "
            f"It is not a diagnosis."
        )

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=400,
            system=system,
            messages=clean_history,
        )
        reply = response.content[0].text.strip()
        return {"ok": True, "reply": reply, "error": None}
    except Exception as exc:
        print(f"[claude:chat] failed: {exc!r}")
        return {
            "ok": False,
            "reply": None,
            "error": "Couldn't reach the chat service. Please try again in a moment.",
        }

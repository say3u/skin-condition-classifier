"""
Grad-CAM (Selvaraju et al. 2017) implementation for the trained EfficientNet-B0.

Given an input image and the model's predicted class, produces a heatmap of which
regions of the photo most influenced the prediction. Useful as a sanity check
("is the model looking at the lesion, or at random skin?") and as an explainability
hook in the UI.

Returns the overlay as a base64-encoded PNG, ready to drop into <img src="data:image/png;base64,..."/>.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import torch
from PIL import Image


def compute_gradcam_overlay(
    bundle: dict[str, Any],
    image: Image.Image,
    target_class_idx: int | None = None,
    alpha: float = 0.45,
) -> tuple[str, int]:
    """
    Compute a Grad-CAM heatmap and overlay it on the original image.

    Args:
        bundle:           the model bundle returned by load_model()
        image:            the input PIL image (will be converted to RGB)
        target_class_idx: class index to explain (None = use the model's top-1 prediction)
        alpha:            blend strength of the heatmap (0 = invisible, 1 = solid)

    Returns:
        (overlay_png_base64, target_class_idx)
    """
    model = bundle["model"]
    device = bundle["device"]
    preprocess = bundle["preprocess"]

    rgb = image.convert("RGB")
    x = preprocess(rgb).unsqueeze(0).to(device)
    x.requires_grad_(False)

    # Hook the last conv block of EfficientNet's backbone. That's where spatial
    # information is still preserved before the global average pool collapses it.
    target_layer = model.features[-1]

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_module, _inputs, output):
        activations.append(output)

    def bwd_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    try:
        # Forward + backward for the requested class
        model.zero_grad()
        logits = model(x)
        if target_class_idx is None:
            target_class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, target_class_idx]
        score.backward()
    finally:
        fh.remove()
        bh.remove()

    # Grad-CAM: weight the activations by the channel-mean of the gradients.
    act = activations[0].detach()                        # [1, C, H, W]
    grad = gradients[0].detach()                         # [1, C, H, W]
    weights = grad.mean(dim=(2, 3), keepdim=True)        # [1, C, 1, 1]
    cam = (weights * act).sum(dim=1).squeeze(0)          # [H, W]
    cam = torch.relu(cam)                                # only positive contributions
    cam_np = cam.cpu().numpy()
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()                   # normalize to [0, 1]

    overlay_img = _apply_heatmap(rgb, cam_np, alpha=alpha)

    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG", optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}", target_class_idx


def _apply_heatmap(base_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Blend a warm (red/orange/yellow) heatmap onto the base image where cam is hot.

    Hand-built colormap so we don't have to pull in matplotlib at inference time:
        cam=0   -> transparent
        cam=0.5 -> deep red
        cam=1.0 -> bright yellow
    """
    W, H = base_image.size
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    cam_resized = np.asarray(cam_img, dtype=np.float32) / 255.0

    # Warm gradient: black -> red -> orange -> yellow
    r = np.clip(cam_resized * 1.5, 0, 1)
    g = np.clip(cam_resized * 1.5 - 0.5, 0, 1)
    b = np.clip(cam_resized * 1.5 - 1.4, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)

    # Alpha by cam intensity, capped by overall `alpha`
    alpha_layer = (cam_resized * alpha)
    rgba = np.concatenate([rgb, alpha_layer[..., None]], axis=-1)
    heat = Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")

    base = base_image.convert("RGBA")
    composed = Image.alpha_composite(base, heat)
    return composed.convert("RGB")

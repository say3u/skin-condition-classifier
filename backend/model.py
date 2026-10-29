"""Model load + single-image prediction. Pure inference, no web framework."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

from backend.calibration import load_temperature

MALIGNANCY = {
    "akiec": "pre-cancerous",
    "bcc":   "cancerous",
    "bkl":   "benign",
    "df":    "benign",
    "mel":   "cancerous",
    "nv":    "benign",
    "vasc":  "benign",
}

# If top-1 is below this, return low_confidence=True instead of pretending.
LOW_CONFIDENCE_FLOOR = 0.40

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(model_path: Path) -> dict[str, Any]:
    """Load the checkpoint once at startup. Returns everything predict_top3 needs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model] loading {model_path} on {device}")

    ckpt = torch.load(model_path, map_location=device)
    classes = ckpt["classes"]
    full_names = ckpt["full_names"]
    img_size = ckpt.get("img_size", 224)

    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    temperature = load_temperature()
    print(f"[model] ready - {len(classes)} classes, temperature={temperature}")
    return {
        "model": model,
        "device": device,
        "classes": classes,
        "full_names": full_names,
        "preprocess": preprocess,
        "temperature": temperature,
    }


def predict_top3(bundle: dict[str, Any], image: Image.Image) -> dict[str, Any]:
    """Run inference and shape the response the frontend wants."""
    x = bundle["preprocess"](image.convert("RGB")).unsqueeze(0).to(bundle["device"])
    with torch.no_grad():
        logits = bundle["model"](x)
        # Temperature scaling: divide logits by T before softmax for calibrated probabilities.
        # T=1.0 is a no-op (identical to uncalibrated behavior).
        logits = logits / bundle.get("temperature", 1.0)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = probs.argsort()[::-1][:3]
    classes = bundle["classes"]
    full_names = bundle["full_names"]
    top1_prob = float(probs[top_idx[0]])

    top_3 = [
        {
            "code": classes[i],
            "name": full_names[classes[i]],
            "malignancy": MALIGNANCY[classes[i]],
            "probability": float(probs[i]),
        }
        for i in top_idx
    ]

    return {
        "top_3": top_3,
        "low_confidence": top1_prob < LOW_CONFIDENCE_FLOOR,
        "low_confidence_threshold": LOW_CONFIDENCE_FLOOR,
    }

"""
Desktop app for the skin condition classifier.

Drag in a photo, get the top-3 guesses. Loads the .pt I trained in the notebook.

NOT a diagnosis. The model is trained on dermatoscopic images (a lens pressed against
the skin), so phone photos will underperform - this is a learning project, not a
medical tool.

Run it:
    python app/desktop_app.py

That prints a local URL (usually http://127.0.0.1:7860). Open it in any browser.
"""

from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

# ----- paths -----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "skin_classifier_efficientnet_b0.pt"

# ----- load the model once at startup -----
# (don't reload on every prediction - that would take seconds per click)
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Couldn't find the model at {MODEL_PATH}.\n"
        "Train it in notebooks/01_train_ham10000_efficientnet.ipynb first, "
        "then download skin_classifier_efficientnet_b0.pt into the project root."
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"loading model from {MODEL_PATH} on {DEVICE}...")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
CLASSES = ckpt["classes"]
FULL_NAMES = ckpt["full_names"]
IMG_SIZE = ckpt.get("img_size", 224)

# rebuild the same architecture and load weights
model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE).eval()
print(f"model loaded - {len(CLASSES)} classes: {CLASSES}")

# same eval transform as the notebook (matters - mismatched preprocessing = garbage predictions)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# malignancy hint shown next to each prediction - keeps the user oriented
MALIGNANCY = {
    "akiec": "pre-cancerous",
    "bcc":   "cancerous",
    "bkl":   "benign",
    "df":    "benign",
    "mel":   "cancerous",
    "nv":    "benign",
    "vasc":  "benign",
}


def predict(image: Image.Image):
    """Returns a {label: probability} dict for the top 3 guesses + a markdown summary."""
    if image is None:
        return {}, "Upload or drag in a photo to get a guess."

    img = image.convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()

    # top-3 for Gradio's Label component
    top_idx = probs.argsort()[::-1][:3]
    label_probs = {
        f"{CLASSES[i]} - {FULL_NAMES[CLASSES[i]]} ({MALIGNANCY[CLASSES[i]]})": float(probs[i])
        for i in top_idx
    }

    # human-readable summary underneath
    top1 = CLASSES[top_idx[0]]
    summary_lines = [
        f"**Top guess: `{top1}` ({FULL_NAMES[top1]}) - {MALIGNANCY[top1]}.**",
        "",
        "This is a statistical guess from a model trained on ~10k dermatoscopic images. "
        "It is NOT a medical opinion. Please see a dermatologist if you're actually worried.",
    ]
    if top1 == "mel" or any(CLASSES[i] in ("mel", "bcc", "akiec") for i in top_idx):
        summary_lines.insert(
            1,
            "> Heads up: the top guesses include a potentially malignant or pre-malignant category. "
            "Don't panic - false alarms are common - but this is a real reason to get it checked.",
        )

    return label_probs, "\n".join(summary_lines)


# ----- UI -----
DESCRIPTION = """
# Skin condition classifier

Drop in a photo of a skin lesion. The model returns its top-3 guesses across 7 categories
from the HAM10000 dataset.

**This is not a diagnosis.** It's a personal learning project. The model was trained on
dermatoscopic images (a clinical lens pressed against the skin), so ordinary phone photos
will perform notably worse. If something on your skin worries you, see a dermatologist.
"""

with gr.Blocks(title="Skin condition classifier") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="Skin lesion photo", sources=["upload", "clipboard"])
            go = gr.Button("Classify", variant="primary")
        with gr.Column(scale=1):
            out_label = gr.Label(num_top_classes=3, label="Top 3 guesses")
            out_text = gr.Markdown()

    go.click(predict, inputs=inp, outputs=[out_label, out_text])
    # also fire on image change so you don't have to click
    inp.change(predict, inputs=inp, outputs=[out_label, out_text])

    gr.Markdown(
        "---\n"
        "Built by David Nguyen as a learning project. "
        "Code: github.com/dsayeu/skin-condition-classifier (placeholder until I push). "
        "See `docs/ETHICS.md` in the repo for the full limitations write-up."
    )

if __name__ == "__main__":
    # share=False keeps it local-only (no public URL). Set to True if you want to demo it remotely.
    demo.launch(server_name="127.0.0.1", share=False, inbrowser=True)

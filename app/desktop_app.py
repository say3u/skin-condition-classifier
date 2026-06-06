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


# If top-1 is below this, the model basically doesn't know what it's looking at.
# Better to admit it than fake a confident wrong answer (e.g. classifying a photo of
# a wall, or pilaris, or anything outside the 7 trained classes).
LOW_CONFIDENCE_FLOOR = 0.40


def predict(image: Image.Image):
    """Returns a {label: probability} dict for the top 3 guesses + a markdown summary."""
    if image is None:
        return {}, "*Upload or drag in a photo to get a guess.*"

    img = image.convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()

    top_idx = probs.argsort()[::-1][:3]
    top1_prob = float(probs[top_idx[0]])
    top1 = CLASSES[top_idx[0]]

    # ----- Low-confidence path: be honest instead of guessing -----
    if top1_prob < LOW_CONFIDENCE_FLOOR:
        # Still show the top-3 so curious users can see what the model was thinking,
        # but lead with a clear "I don't know" message.
        label_probs = {
            f"{CLASSES[i]} - {FULL_NAMES[CLASSES[i]]} ({MALIGNANCY[CLASSES[i]]})": float(probs[i])
            for i in top_idx
        }
        summary = (
            f"### 🤷 I'm not confident about this one\n\n"
            f"My top guess is only **{top1_prob:.0%}** sure — below the threshold I trust "
            f"({int(LOW_CONFIDENCE_FLOOR * 100)}%). That usually means one of:\n\n"
            f"- The photo isn't of a skin lesion at all (the model has no \"not skin\" class).\n"
            f"- It *is* a lesion, but a kind I wasn't trained on (I only know 7 categories — "
            f"acne, eczema, rosacea, keratosis pilaris and many others aren't in there).\n"
            f"- The lighting / angle / focus is throwing me off.\n\n"
            f"Treat the guesses below as low-trust. If this is a real concern, see a "
            f"dermatologist — they'll do a much better job than this model."
        )
        return label_probs, summary

    # ----- Normal-confidence path -----
    label_probs = {
        f"{CLASSES[i]} - {FULL_NAMES[CLASSES[i]]} ({MALIGNANCY[CLASSES[i]]})": float(probs[i])
        for i in top_idx
    }

    summary_lines = [
        f"### Top guess: **{FULL_NAMES[top1]}** ({top1}) - {MALIGNANCY[top1]}",
        "",
        "This is a statistical guess from a model trained on ~10k dermatoscopic images. "
        "It is NOT a medical opinion. Please see a dermatologist if you're actually worried.",
    ]
    if top1 == "mel" or any(CLASSES[i] in ("mel", "bcc", "akiec") for i in top_idx):
        summary_lines.insert(
            1,
            "> **Heads up:** the top guesses include a potentially malignant or pre-malignant "
            "category. Don't panic - false alarms are common - but this is a real reason to get "
            "it checked by a dermatologist.",
        )

    return label_probs, "\n".join(summary_lines)


# ----- UI -----
HEADER = """
# 🔬 Skin Condition Classifier
#### A personal learning project — not a diagnostic tool.
"""

HOW_TO_USE = """
### How to use

1. **Drop or upload a photo** of a skin lesion on the left.
2. The model returns its **top-3 guesses** across 7 dermatology categories on the right.
3. Each guess shows whether it's *benign*, *pre-cancerous*, or *cancerous* — and how confident the model is.

⚠️ **This is not a diagnosis.** It's trained on clinical dermatoscope images (a lens pressed
against the skin), so ordinary phone photos won't perform as well. If something on your skin
worries you, **see a dermatologist** — that's the only right answer here.
"""

FOOTER = """
---
Built by **David Nguyen** as a learning project. The model is fine-tuned EfficientNet-B0
on the [HAM10000](https://www.nature.com/articles/sdata2018161) dataset (~10k images, 7
classes).

Source code & limitations: [github.com/say3u/skin-condition-classifier](https://github.com/say3u/skin-condition-classifier)
"""

# Soft theme is friendlier than the default; primary_hue=teal reads as medical/clean.
theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
)

with gr.Blocks(title="Skin Condition Classifier", theme=theme) as demo:
    gr.Markdown(HEADER)

    with gr.Accordion("How to use this app", open=True):
        gr.Markdown(HOW_TO_USE)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("### 1. Your photo")
            inp = gr.Image(
                type="pil",
                label=None,
                sources=["upload", "clipboard"],
                height=400,
            )
            go = gr.Button("🔍 Classify", variant="primary", size="lg")
            clear = gr.Button("Clear", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### 2. Top-3 guesses")
            out_label = gr.Label(num_top_classes=3, label=None, show_label=False)
            out_text = gr.Markdown("*Upload or drag in a photo to get a guess.*")

    gr.Markdown(FOOTER)

    # Wiring
    go.click(predict, inputs=inp, outputs=[out_label, out_text])
    inp.change(predict, inputs=inp, outputs=[out_label, out_text])
    clear.click(
        lambda: (None, {}, "*Upload or drag in a photo to get a guess.*"),
        inputs=None,
        outputs=[inp, out_label, out_text],
    )

if __name__ == "__main__":
    # share=False keeps it local-only (no public URL). Set to True if you want to demo it remotely.
    demo.launch(server_name="127.0.0.1", share=False, inbrowser=True)

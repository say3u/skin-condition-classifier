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
EXAMPLES_DIR = PROJECT_ROOT / "app" / "examples"

# ----- load the model once at startup -----
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

model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE).eval()
print(f"model loaded - {len(CLASSES)} classes: {CLASSES}")

# Same eval transform as the notebook - mismatched preprocessing means garbage predictions.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

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
# Better to admit it than fake a confident wrong answer.
LOW_CONFIDENCE_FLOOR = 0.40

EMPTY_HINT = "Drop a photo on the left to get a guess."


def predict(image: Image.Image, history: list | None):
    """Classify + append to session history. Returns (label_probs, summary, history_state, gallery)."""
    history = history or []
    if image is None:
        return {}, EMPTY_HINT, history, _history_to_gallery(history)

    img = image.convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()

    top_idx = probs.argsort()[::-1][:3]
    top1_prob = float(probs[top_idx[0]])
    top1 = CLASSES[top_idx[0]]

    label_probs = {
        f"{FULL_NAMES[CLASSES[i]]} ({CLASSES[i]}, {MALIGNANCY[CLASSES[i]]})": float(probs[i])
        for i in top_idx
    }

    if top1_prob < LOW_CONFIDENCE_FLOOR:
        summary = (
            f"#### Low confidence\n\n"
            f"The top guess is only {top1_prob:.0%} sure, below the "
            f"{int(LOW_CONFIDENCE_FLOOR * 100)}% threshold this app trusts. That usually means:\n\n"
            f"- The photo isn't a skin lesion (there's no \"not skin\" class in the model).\n"
            f"- It is a lesion, but a kind the model wasn't trained on. Only seven categories "
            f"are in the training set; acne, eczema, rosacea, and many others are not.\n"
            f"- The lighting, focus, or angle is throwing the model off.\n\n"
            f"Treat the guesses below as low trust. If you're actually worried about something "
            f"on your skin, see a dermatologist."
        )
        history_caption = f"low confidence ({top1_prob:.0%})"
    else:
        lines = [
            f"#### Top guess: {FULL_NAMES[top1]}",
            f"_{top1} — {MALIGNANCY[top1]} — {top1_prob:.0%} confidence_",
            "",
            "A statistical guess from a model trained on roughly ten thousand "
            "dermatoscopic images. It is not a medical opinion. If you're actually "
            "concerned, see a dermatologist.",
        ]
        if top1 == "mel" or any(CLASSES[i] in ("mel", "bcc", "akiec") for i in top_idx):
            lines.insert(
                2,
                "> The top guesses include a potentially malignant or pre-malignant category. "
                "False alarms are common, but this is a real reason to get it checked.",
            )
        summary = "\n".join(lines)
        history_caption = f"{top1} — {top1_prob:.0%} — {MALIGNANCY[top1]}"

    history = [(img.copy(), history_caption)] + history
    history = history[:5]
    return label_probs, summary, history, _history_to_gallery(history)


def _history_to_gallery(history: list) -> list:
    return [(im, cap) for im, cap in history]


def clear_history():
    return [], []


# ----- UI -----
HEADER_MD = """
<div class="app-header">
  <h1>Skin Condition Classifier</h1>
  <p class="subtitle">A learning project — not a diagnostic tool.</p>
</div>
"""

INTRO_MD = """
Upload a photo of a skin lesion and the model returns its top three guesses across seven
categories from the HAM10000 dataset. Every guess is labelled benign, pre-cancerous, or
cancerous, with a confidence number.

This is not a diagnosis. The model was trained on clinical dermatoscope images, so ordinary
phone photos will perform noticeably worse. If something on your skin worries you, see a
dermatologist.
"""

FOOTER_MD = """
<div class="app-footer">
  Built by David Nguyen. Fine-tuned EfficientNet-B0 on HAM10000.
  <a href="https://github.com/say3u/skin-condition-classifier">Source &amp; limitations</a>.
</div>
"""

# Quiet, restrained theme. Slate neutrals + a single muted accent.
theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#f5f7fa", c100="#e4e9f0", c200="#cdd5e0", c300="#a8b3c4",
        c400="#7d8aa1", c500="#566479", c600="#3f4a5c", c700="#303948",
        c800="#222936", c900="#171c26", c950="#0e121a",
    ),
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_md,
    spacing_size=gr.themes.sizes.spacing_md,
).set(
    body_background_fill="#fafbfc",
    body_background_fill_dark="#0f1116",
    block_background_fill="white",
    block_background_fill_dark="#161922",
    block_border_width="1px",
    block_border_color="#e6e8ec",
    block_border_color_dark="#252834",
    block_shadow="0 1px 2px rgba(15, 17, 22, 0.04)",
    block_title_text_weight="600",
    block_label_text_weight="500",
    button_primary_background_fill="*primary_700",
    button_primary_background_fill_hover="*primary_800",
    button_primary_text_color="white",
)

CUSTOM_CSS = """
.gradio-container { max-width: 980px !important; margin: 0 auto !important; }
.app-header { padding: 24px 0 8px; border-bottom: 1px solid #e6e8ec; margin-bottom: 16px; }
.app-header h1 { margin: 0; font-size: 22px; font-weight: 600; letter-spacing: -0.01em; color: #1f2330; }
.app-header .subtitle { margin: 4px 0 0; color: #6b7280; font-size: 14px; }
.app-footer { padding: 16px 0 8px; border-top: 1px solid #e6e8ec; margin-top: 24px; color: #6b7280; font-size: 13px; }
.app-footer a { color: #3f4a5c; text-decoration: none; border-bottom: 1px solid #d4d8df; }
.app-footer a:hover { color: #1f2330; }
.section-label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: #6b7280; margin: 0 0 8px; font-weight: 600; }
footer { display: none !important; }
"""

with gr.Blocks(title="Skin Condition Classifier", theme=theme, css=CUSTOM_CSS) as demo:
    gr.HTML(HEADER_MD)
    gr.Markdown(INTRO_MD)

    history_state = gr.State([])

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.HTML('<p class="section-label">Photo</p>')
            inp = gr.Image(
                type="pil",
                show_label=False,
                sources=["upload", "clipboard"],
                height=380,
            )
            with gr.Row():
                go = gr.Button("Classify", variant="primary", scale=2)
                clear = gr.Button("Clear", variant="secondary", scale=1)

            example_files = sorted(
                [str(p) for p in EXAMPLES_DIR.glob("*.jpg")] +
                [str(p) for p in EXAMPLES_DIR.glob("*.png")]
            )
            if example_files:
                gr.Examples(
                    examples=[[p] for p in example_files],
                    inputs=[inp],
                    label="Examples",
                    examples_per_page=8,
                )

        with gr.Column(scale=1):
            gr.HTML('<p class="section-label">Top three guesses</p>')
            out_label = gr.Label(num_top_classes=3, show_label=False)
            out_text = gr.Markdown(EMPTY_HINT)

    with gr.Accordion("Recent scans (this session)", open=False):
        gr.Markdown(
            "The last five photos you classified. Cleared when the tab closes; "
            "nothing is saved to disk."
        )
        history_gallery = gr.Gallery(
            show_label=False,
            columns=5,
            height=160,
            object_fit="cover",
            preview=False,
        )
        clear_history_btn = gr.Button("Clear history", size="sm")

    gr.HTML(FOOTER_MD)

    # Wiring
    go.click(predict, inputs=[inp, history_state],
             outputs=[out_label, out_text, history_state, history_gallery])
    inp.change(predict, inputs=[inp, history_state],
               outputs=[out_label, out_text, history_state, history_gallery])
    clear.click(
        lambda: (None, {}, EMPTY_HINT),
        inputs=None,
        outputs=[inp, out_label, out_text],
    )
    clear_history_btn.click(clear_history, inputs=None, outputs=[history_state, history_gallery])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False, inbrowser=True)

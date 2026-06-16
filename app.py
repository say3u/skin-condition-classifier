"""
Desktop app for the skin condition classifier.

Three-screen flow:
  1. Capture - take a photo with your webcam, or upload one.
  2. Results - top-3 guesses with confidence.
  3. Info    - description, signs, when to see a doctor, general care.

Black-and-white UI, centered narrow column. Loads the .pt trained in the notebook.

Not a diagnosis. Educational / triage demo only.

Run locally:
    python app.py
Open http://127.0.0.1:7860 (the browser tab opens automatically).

Also the entry point used by Hugging Face Spaces (HF runs `python app.py` at the repo
root). One file, two deploy targets.
"""

from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

# ----- paths -----
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "skin_classifier_efficientnet_b0.pt"
EXAMPLES_DIR = PROJECT_ROOT / "app" / "examples"

# ----- load model -----
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Couldn't find the model at {MODEL_PATH}. "
        "Train it in notebooks/01_train_ham10000_efficientnet.ipynb first."
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
print(f"model loaded - {len(CLASSES)} classes")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

MALIGNANCY = {
    "akiec": "pre-cancerous", "bcc": "cancerous", "bkl": "benign",
    "df": "benign", "mel": "cancerous", "nv": "benign", "vasc": "benign",
}

LOW_CONFIDENCE_FLOOR = 0.40

# ---- Per-condition educational info (general info, not medical advice). ----
# Sources: AAD, Mayo Clinic, NHS public skin-health pages. Phrased as education.
CONDITION_INFO = {
    "akiec": {
        "title": "Actinic keratosis (and Bowen's disease)",
        "tagline": "Pre-cancerous. Caused by long-term sun damage.",
        "about": (
            "Rough, scaly patches caused by years of sun exposure. Considered pre-cancerous: "
            "a small fraction progress to squamous cell carcinoma if left untreated. Common on "
            "the face, scalp, ears, hands, and forearms."
        ),
        "signs": [
            "Dry, scaly patches that often feel like sandpaper.",
            "Pink, red, or skin-coloured.",
            "Usually a few millimetres across; flat or slightly raised.",
            "Most often on sun-exposed skin in older adults.",
        ],
        "doctor": (
            "Any persistent rough, scaly patch on sun-exposed skin should be evaluated. "
            "Early treatment is straightforward."
        ),
        "care": [
            "Broad-spectrum SPF 30+ daily, even on cloudy days.",
            "Sun-protective clothing, hat, sunglasses.",
            "Avoid peak-sun hours (roughly 10am to 4pm).",
            "Self-check your skin monthly.",
        ],
    },
    "bcc": {
        "title": "Basal cell carcinoma",
        "tagline": "Most common skin cancer. Slow-growing, almost always curable when caught early.",
        "about": (
            "The most common form of skin cancer. It grows slowly and rarely spreads, but it can "
            "damage surrounding skin and tissue if ignored. Driven by cumulative UV exposure. "
            "Most often appears on the face, ears, and neck."
        ),
        "signs": [
            "Pearly or waxy bump, often pink or skin-coloured.",
            "Visible small blood vessels on the surface.",
            "A sore that bleeds, scabs, and doesn't heal.",
            "A flat, scaly reddish patch (on the chest or back).",
        ],
        "doctor": (
            "Any sore that doesn't heal within a few weeks, or a new pearly bump, should be "
            "checked promptly. Caught early, treatment is usually a quick outpatient procedure."
        ),
        "care": [
            "Daily broad-spectrum SPF 30+ for life.",
            "Avoid tanning beds entirely.",
            "Annual full-body dermatologist check, especially after one BCC.",
            "Self-check monthly: any new bump or non-healing sore deserves a look.",
        ],
    },
    "bkl": {
        "title": "Benign keratosis",
        "tagline": "Harmless growths that appear with age. Not cancerous.",
        "about": (
            "Common, harmless growths that appear with age. Sometimes called age spots or "
            "barnacles. They can look alarming but are not cancerous."
        ),
        "signs": [
            "Waxy, stuck-on-looking patches.",
            "Brown, black, or tan, sometimes mixed colours.",
            "Can be flat or raised; surface often slightly rough.",
            "Common on chest, back, shoulders, scalp.",
        ],
        "doctor": (
            "If a spot changes rapidly, bleeds without obvious cause, or you can't tell it apart "
            "from a melanoma, have it checked. Otherwise no action needed."
        ),
        "care": [
            "No medical treatment required.",
            "If a spot is irritated by clothing or jewellery, a dermatologist can remove it.",
            "Sun protection slows the appearance of new ones.",
        ],
    },
    "df": {
        "title": "Dermatofibroma",
        "tagline": "Benign firm nodule, often on the legs. Frequently follows a minor skin injury.",
        "about": (
            "A small firm nodule under the skin, often where there was a minor injury such as an "
            "insect bite, ingrown hair, or shaving nick. Benign and very common, especially on "
            "the legs of adults."
        ),
        "signs": [
            "Small, firm, brown / pink / red bump.",
            "Dimples inward when squeezed from the sides (the dimple sign).",
            "Usually 5 to 10 mm across.",
            "Most often on the legs.",
        ],
        "doctor": (
            "If it changes size or colour, becomes painful, or you are unsure whether it is a "
            "dermatofibroma at all, have it checked."
        ),
        "care": [
            "No treatment needed unless it bothers you.",
            "Don't pick at it - that can cause irritation or scarring.",
            "Removal is possible but leaves a small scar; usually only done if symptomatic.",
        ],
    },
    "mel": {
        "title": "Melanoma",
        "tagline": "Serious skin cancer. Highly treatable when caught early. Don't wait.",
        "about": (
            "A serious form of skin cancer that starts in the pigment-producing cells "
            "(melanocytes). Can spread quickly if not caught early. Early detection dramatically "
            "improves outcomes - this is the one that matters most."
        ),
        "signs": [
            "A new or changing mole is the single most important sign.",
            "ABCDE rule: Asymmetric, irregular Border, multiple Colours, Diameter over 6mm, Evolving.",
            "A mole that bleeds, itches, or scabs.",
            "A spot that looks different from the rest of your moles (the ugly duckling sign).",
        ],
        "doctor": (
            "As soon as possible - days, not months. If you suspect melanoma, ask for an urgent "
            "dermatology appointment. Early treatment can be curative; delay is the real danger."
        ),
        "care": [
            "Monthly skin self-exam - photograph moles to track changes.",
            "Annual full-body dermatologist check if you have risk factors.",
            "Strict sun protection: SPF 30+, hat, sun-protective clothing, no tanning beds.",
            "If you've had one melanoma, follow-up screening is essential.",
        ],
    },
    "nv": {
        "title": "Melanocytic nevus (common mole)",
        "tagline": "Common, usually harmless. Watch for change over time.",
        "about": (
            "A common cluster of pigment cells. Most adults have between ten and forty. Usually "
            "harmless, but moles can occasionally evolve into melanoma - so monitoring matters "
            "more than the mole itself."
        ),
        "signs": [
            "Round, even-coloured, smaller than 6mm.",
            "Brown, tan, or skin-coloured.",
            "Stable over time - change is the warning sign.",
        ],
        "doctor": (
            "If a mole changes shape, colour, or size; bleeds; itches; or starts looking "
            "different from your other moles. The ABCDE rule for melanoma applies here too."
        ),
        "care": [
            "Self-check monthly - photographs help spot slow changes.",
            "Use sun protection: SPF, hat, shade.",
            "If you have many moles or atypical ones, see a dermatologist yearly.",
            "Don't try to remove moles yourself.",
        ],
    },
    "vasc": {
        "title": "Vascular lesion",
        "tagline": "Marks involving blood vessels. Almost always benign.",
        "about": (
            "A broad category for skin marks involving blood vessels - cherry angiomas, spider "
            "veins, hemangiomas. Almost always benign and a normal part of ageing skin."
        ),
        "signs": [
            "Bright red, purple, or blue.",
            "Usually small (a few millimetres), raised, smooth.",
            "May briefly turn pale when pressed.",
        ],
        "doctor": (
            "If it grows rapidly, bleeds excessively when bumped, or is in a problematic spot, "
            "have it checked. Most don't need any treatment."
        ),
        "care": [
            "No medical treatment usually needed.",
            "Sun protection helps prevent new ones.",
            "Cosmetic removal with lasers or electrocautery is an option.",
        ],
    },
}


def info_markdown_for(class_code: str) -> str:
    info = CONDITION_INFO[class_code]
    lines = [
        f"# {info['title']}",
        f"_{info['tagline']}_",
        "",
        "## About",
        info["about"],
        "",
        "## Common signs",
    ]
    lines += [f"- {s}" for s in info["signs"]]
    lines += ["", "## When to see a doctor", info["doctor"], "", "## General care"]
    lines += [f"- {c}" for c in info["care"]]
    lines += [
        "",
        "---",
        "_General educational information, not medical advice. If you're concerned about "
        "something on your skin, see a dermatologist._",
    ]
    return "\n".join(lines)


# ---- Screen transition handlers ----

def classify(image: Image.Image):
    """Capture screen -> results screen."""
    if image is None:
        # Stay on capture screen, no-op.
        return (
            {}, "", None, None,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    img = image.convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()

    top_idx = probs.argsort()[::-1][:3]
    top1_prob = float(probs[top_idx[0]])
    top1 = CLASSES[top_idx[0]]

    label_probs = {
        f"{FULL_NAMES[CLASSES[i]]} ({MALIGNANCY[CLASSES[i]]})": float(probs[i])
        for i in top_idx
    }

    if top1_prob < LOW_CONFIDENCE_FLOOR:
        summary = (
            f"#### Low confidence ({top1_prob:.0%})\n\n"
            f"The top guess sits below the {int(LOW_CONFIDENCE_FLOOR * 100)}% trust threshold. "
            f"This usually means the photo isn't a lesion the model recognises. Only seven "
            f"categories are in the training set; acne, eczema, rosacea, keratosis pilaris and "
            f"many others aren't among them. Treat the guesses below as low trust."
        )
        # Disable info button for low-confidence (no class to look up)
        info_btn_update = gr.update(interactive=False, value="More information unavailable")
    else:
        lines = [
            f"#### {FULL_NAMES[top1]}",
            f"_{top1_prob:.0%} confidence — {MALIGNANCY[top1]}_",
        ]
        if top1 in ("mel", "bcc", "akiec") or any(
            CLASSES[i] in ("mel", "bcc", "akiec") for i in top_idx
        ):
            lines += [
                "",
                "> The top guesses include a potentially malignant or pre-malignant category. "
                "False alarms are common, but this is a real reason to have it checked by a "
                "dermatologist.",
            ]
        summary = "\n".join(lines)
        info_btn_update = gr.update(interactive=True, value="More information")

    return (
        label_probs,
        summary,
        top1,
        img,
        gr.update(visible=False),  # capture
        gr.update(visible=True),   # results
        gr.update(visible=False),  # info
        info_btn_update,
    )


def go_to_info(top1: str | None):
    """Results screen -> info screen."""
    if top1 is None:
        # Shouldn't happen if the button was correctly disabled; bounce to capture.
        return (
            "",
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    return (
        info_markdown_for(top1),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
    )


def back_to_results():
    """Info screen -> results screen."""
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def try_another():
    """Any screen -> capture screen (clear everything)."""
    return (
        None,                       # input image
        {},                         # label
        "",                         # summary text
        None,                       # top1 state
        None,                       # results thumbnail
        "",                         # info markdown
        gr.update(visible=True),    # capture
        gr.update(visible=False),   # results
        gr.update(visible=False),   # info
        gr.update(interactive=True, value="More information"),  # reset info btn
    )


# ----- UI -----
HEADER_HTML = """
<div class="app-header">
  <h1>Skin Condition Classifier</h1>
  <p class="subtitle">A learning project, not a diagnostic tool.</p>
</div>
"""

FOOTER_HTML = """
<div class="app-footer">
  <a href="https://github.com/say3u/skin-condition-classifier">github.com/say3u/skin-condition-classifier</a>
</div>
"""

INTRO_MD = (
    "Take a photo with your camera or upload one. The model returns its top three guesses "
    "across seven dermatology categories. Not a diagnosis - see a dermatologist if you're "
    "worried."
)

# Pure black-and-white CSS.
CUSTOM_CSS = """
.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    padding: 32px 40px 48px !important;
    background: #ffffff !important;
}

/* Buttons shouldn't stretch the full width of the desktop rectangle - looks weird. */
.action-row { display: flex; gap: 12px; justify-content: center; margin-top: 16px; }
.action-row button { min-width: 200px; max-width: 320px; flex: 1; }
body { background: #ffffff !important; color: #000 !important; }

/* Header / footer */
.app-header { text-align: center; padding-bottom: 20px; border-bottom: 1px solid #000; margin-bottom: 28px; }
.app-header h1 { margin: 0; font-size: 24px; font-weight: 700; letter-spacing: -0.02em; color: #000; }
.app-header .subtitle { margin: 6px 0 0; color: #555; font-size: 13px; }
.app-footer { text-align: center; padding-top: 28px; border-top: 1px solid #000; margin-top: 40px; color: #555; font-size: 12px; }
.app-footer a { color: #000; text-decoration: underline; }

/* Block defaults: square corners, single black border, no shadow */
.block, .gr-block, .gr-form, .form { border-radius: 0 !important; box-shadow: none !important; background: #fff !important; }
.block { border-color: #000 !important; }

/* Labels above components - small uppercase */
.block-label, label > span:first-child {
    color: #000 !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
}

/* Buttons */
button { border-radius: 0 !important; font-weight: 600 !important; transition: background 0.15s, color 0.15s; }
.primary, button.primary { background: #000 !important; color: #fff !important; border: 1px solid #000 !important; }
.primary:hover, button.primary:hover { background: #fff !important; color: #000 !important; }
.secondary, button.secondary { background: #fff !important; color: #000 !important; border: 1px solid #000 !important; }
.secondary:hover, button.secondary:hover { background: #000 !important; color: #fff !important; }
button[disabled], button.primary[disabled] { background: #ddd !important; color: #888 !important; border-color: #ddd !important; cursor: not-allowed; }

/* Confidence bars: solid black */
.label-wrap .bar, .output-class .bar { background: #000 !important; }

/* Markdown typography */
.markdown h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.01em; margin: 0 0 4px; color: #000; }
.markdown h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.12em; margin: 24px 0 10px; color: #000; border-bottom: 1px solid #000; padding-bottom: 6px; }
.markdown h3, .markdown h4 { font-size: 14px; font-weight: 600; margin: 16px 0 6px; color: #000; }
.markdown p { line-height: 1.55; color: #1a1a1a; }
.markdown ul { padding-left: 18px; }
.markdown li { margin: 4px 0; line-height: 1.45; }
.markdown blockquote { border-left: 3px solid #000; padding: 10px 14px; margin: 14px 0; background: #f5f5f5; color: #000; }
.markdown em { color: #555; }
.markdown hr { border: 0; border-top: 1px solid #000; margin: 24px 0 12px; }

/* Hide gradio's own footer */
footer { display: none !important; }
"""

theme = gr.themes.Monochrome(
    radius_size=gr.themes.sizes.radius_none,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(title="Skin Condition Classifier") as demo:
    gr.HTML(HEADER_HTML)

    # State carried across screens
    top1_state = gr.State(None)

    # ============ Screen 1 — Capture ============
    with gr.Group(visible=True) as capture_screen:
        gr.Markdown(INTRO_MD)
        inp = gr.Image(
            type="pil",
            sources=["webcam", "upload", "clipboard"],
            height=420,
            show_label=False,
        )
        with gr.Row(elem_classes="action-row"):
            classify_btn = gr.Button("Classify", variant="primary", size="lg")

        example_files = sorted(
            [str(p) for p in EXAMPLES_DIR.glob("*.jpg")] +
            [str(p) for p in EXAMPLES_DIR.glob("*.png")]
        )
        if example_files:
            gr.Examples(
                examples=[[p] for p in example_files],
                inputs=[inp],
                label="Or try an example",
                examples_per_page=6,
            )

    # ============ Screen 2 — Results ============
    with gr.Group(visible=False) as results_screen:
        results_thumb = gr.Image(
            show_label=False,
            interactive=False,
            height=200,
            container=False,
        )
        out_label = gr.Label(num_top_classes=3, show_label=False)
        out_text = gr.Markdown()
        with gr.Row(elem_classes="action-row"):
            info_btn = gr.Button("More information", variant="primary")
            retry_btn = gr.Button("Try another photo", variant="secondary")

    # ============ Screen 3 — Info ============
    with gr.Group(visible=False) as info_screen:
        info_md = gr.Markdown()
        with gr.Row(elem_classes="action-row"):
            back_btn = gr.Button("Back to results", variant="primary")
            retry_btn2 = gr.Button("Try another photo", variant="secondary")

    gr.HTML(FOOTER_HTML)

    # ---- Wiring ----
    classify_btn.click(
        classify,
        inputs=[inp],
        outputs=[
            out_label, out_text, top1_state, results_thumb,
            capture_screen, results_screen, info_screen,
            info_btn,
        ],
    )
    info_btn.click(
        go_to_info,
        inputs=[top1_state],
        outputs=[info_md, capture_screen, results_screen, info_screen],
    )
    back_btn.click(
        back_to_results,
        outputs=[capture_screen, results_screen, info_screen],
    )
    for btn in (retry_btn, retry_btn2):
        btn.click(
            try_another,
            outputs=[
                inp, out_label, out_text, top1_state, results_thumb, info_md,
                capture_screen, results_screen, info_screen, info_btn,
            ],
        )

if __name__ == "__main__":
    # share=True gives a public *.gradio.live URL that lasts 72 hours.
    # The URL still tunnels through this machine, so this script has to keep running.
    demo.launch(
        server_name="127.0.0.1",
        share=True,
        inbrowser=True,
        theme=theme,
        css=CUSTOM_CSS,
    )

---
title: Skin Condition Classifier
emoji: 🩺
colorFrom: gray
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Educational skin-lesion classifier (HAM10000 + EfficientNet)
---

# Skin Condition Classifier

A web app that takes a photo of a skin lesion and returns its top guesses across seven
dermatology categories &mdash; built as a personal project to learn computer vision, model
deployment, and what it takes to build responsible health-adjacent ML tools.

**Live demo:** [huggingface.co/spaces/sayeu/skin-condition-classifier](https://huggingface.co/spaces/sayeu/skin-condition-classifier)

> Educational only. Not a medical device, not a diagnosis. If something on your skin worries
> you, see a dermatologist. See [`docs/ETHICS.md`](docs/ETHICS.md) for the full limitations write-up.

## What it does

Take a photo with your webcam or upload one. The model returns three possible matches with
confidence scores. If the photo is too unclear or doesn't match any of the seven trained
categories, the UI says &ldquo;couldn't identify&rdquo; and gives concrete photo-quality tips instead of
faking a guess. When a result includes a potentially malignant category, the UI surfaces a
clear warning and points the user at a real dermatologist.

## What I built

- **Trained an EfficientNet-B0** with transfer learning on the [HAM10000](https://www.nature.com/articles/sdata2018161)
  dermatoscopy dataset (~10k images, 7 lesion classes). Used class-weighted loss to handle
  the dataset's heavy imbalance (one class is ~67% of the data) and split by lesion (not by
  image) to prevent data leakage.
- **Reported metrics that matter for health data** &mdash; per-class precision/recall, macro-F1,
  confusion matrix &mdash; instead of leaning on raw accuracy on an imbalanced dataset.
- **Wrote a FastAPI backend** that exposes a `POST /api/predict` endpoint wrapping the model.
- **Hand-built the frontend** &mdash; vanilla HTML, CSS, and JavaScript, no framework. Webcam
  capture via `getUserMedia`, three-screen flow (capture &rarr; results &rarr; detail), Playfair
  Display + Inter typography, fully responsive.
- **Deployed via Docker** to Hugging Face Spaces &mdash; one container serving both the API
  and the static frontend.

## Results (HAM10000, EfficientNet-B0, 20 epochs on Colab T4)

| Metric | Value |
|---|---|
| Best validation macro-F1 | **0.730** |
| Melanoma (`mel`) recall | **67.9%** |
| Training images | ~10,000 |
| Classes | 7 |

A 67.9% melanoma recall means the model misses roughly 1 in 3 melanomas, which is **why the
whole UI is built around honest framing** &mdash; top-3 guesses instead of one, the &ldquo;couldn't
identify&rdquo; path, the malignancy warning callout, and the explicit disclaimer. The hard part
of health ML isn't training a model &mdash; it's deciding what to do when the model is wrong.

## Tech stack

PyTorch &middot; torchvision &middot; FastAPI &middot; Docker &middot; HTML / CSS / JavaScript &middot; Hugging Face Spaces

## Roadmap

- [x] HAM10000 classifier trained, deployed live
- [x] Real frontend with three-screen flow + honest low-confidence path
- [ ] Fitzpatrick17k second model + per-skin-type fairness analysis ([notebook started](notebooks/02_train_fitzpatrick17k_efficientnet.ipynb))
- [ ] Grad-CAM heatmaps on the results screen (so you can see where the model looked)
- [ ] Calibrated confidence via temperature scaling
- [ ] Photo-quality preflight (Claude vision API)

---

## What this project demonstrates

This repo is deliberately scoped to show the *full* ML workflow done thoughtfully —
the parts that matter to an employer, not just "train a model":

- **Transfer learning** — fine-tuning a pretrained **EfficientNet-B0** instead of training from scratch.
- **Honest data handling** — the [HAM10000](docs/DATA.md) dataset is heavily imbalanced
  (~67% one class). I address it with **class-weighted loss** and select the best model by
  **macro-F1**, not accuracy (accuracy is misleading on imbalanced data).
- **Leakage-aware splitting** — train/val/test are split **by lesion**, so multiple photos
  of the *same* lesion never straddle the split and inflate the score.
- **Metrics that matter for health data** — per-class **precision / recall / F1** and a
  **confusion matrix**, with explicit attention to **melanoma recall** (missing a cancer is
  far worse than a false alarm).
- **Awareness of real-world limits** — documented domain gap (dermatoscope vs. phone photos)
  and **skin-tone bias**, the failure modes that sink naive derm models.

## The 7 categories (HAM10000)

| Code | Condition | Notes |
|------|-----------|-------|
| `akiec` | Actinic keratoses / Bowen's | pre-cancerous |
| `bcc` | Basal cell carcinoma | cancerous |
| `bkl` | Benign keratosis | benign |
| `df` | Dermatofibroma | benign |
| `mel` | **Melanoma** | cancerous — the one we care most about catching |
| `nv` | Melanocytic nevus (mole) | benign — the majority class |
| `vasc` | Vascular lesion | benign |

## Quickstart (Google Colab — free GPU, recommended)

1. Open [`notebooks/01_train_ham10000_efficientnet.ipynb`](notebooks/01_train_ham10000_efficientnet.ipynb) in
   [Google Colab](https://colab.research.google.com).
2. Set the runtime to GPU: **Runtime → Change runtime type → T4 GPU**.
3. Run the cells top to bottom. The data-download cell explains how to grab HAM10000
   from Kaggle (see [`docs/DATA.md`](docs/DATA.md)).
4. In ~10–15 minutes you'll have a trained classifier, a confusion matrix, and a
   single-image "top-3 guesses" demo.

### Run locally instead

```bash
git clone <your-repo-url>
cd skin-condition-classifier
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
jupyter notebook notebooks/01_train_ham10000_efficientnet.ipynb
```

A CUDA GPU is strongly recommended; CPU training will be slow.

## Two models

There are two training notebooks in this repo because they cover different needs:

| Notebook | Dataset | Classes | What it's for |
|---|---|---|---|
| `01_train_ham10000_efficientnet.ipynb` | HAM10000 (~10k dermatoscopic images) | **7** lesion types | High-precision cancer-focused triage. Biopsy-confirmed labels. |
| `02_train_fitzpatrick17k_efficientnet.ipynb` | Fitzpatrick17k (~17k atlas images) | **~30** common conditions | Everyday-skin coverage (eczema, psoriasis, acne, rosacea, etc.) + **fairness analysis across skin tones**. |

The HAM10000 model has higher top-1 accuracy but only knows 7 things — it'll confidently
mislabel a photo of eczema as a mole. The Fitzpatrick17k model trades top-1 accuracy for
breadth, and ships with a per-Fitzpatrick-skin-type performance breakdown so you can see
whether the model is equally reliable across skin tones (a real, documented failure mode
in dermatology AI).

## Project structure

```
skin-condition-classifier/
├── notebooks/
│   ├── 01_train_ham10000_efficientnet.ipynb       # 7-class cancer triage model
│   └── 02_train_fitzpatrick17k_efficientnet.ipynb # ~30-class broad model + fairness
├── app/
│   └── desktop_app.py                             # local desktop UI (Gradio)
├── docs/
│   ├── DATA.md      # how to obtain HAM10000 + dataset caveats
│   └── ETHICS.md    # limitations, bias, and responsible-use notes
├── skin_classifier_efficientnet_b0.pt             # HAM10000 weights (NOT in git)
├── skin_classifier_fitz17k_efficientnet_b0.pt    # Fitzpatrick17k weights (NOT in git)
├── requirements.txt
├── LICENSE
└── README.md
```

## Running the desktop app

After training the notebook (so `skin_classifier_efficientnet_b0.pt` is in the project root):

```bash
pip install -r requirements.txt
python app.py
```

It opens a local Gradio window in your browser at `http://127.0.0.1:7860`. Drag in a photo,
see the top-3 guesses and an honest "this is not a diagnosis" banner. No data leaves your
machine.

**Want click-to-try example images** so you don't have to find your own photo first?

```bash
python app/get_examples.py
```

That copies 4 sample images out of your local HAM10000 download (if you have one) into
`app/examples/`, where the app picks them up automatically. Or drop your own `.jpg` files
in there directly.

## Results — baseline (EfficientNet-B0, 20 epochs on Colab T4)

| Metric | Value |
|---|---|
| Best validation macro-F1 | **0.730** |
| Melanoma (`mel`) recall | **67.9%** |
| Major confusion | benign moles called melanoma (false alarm); rare bkl ↔ nv mix |

**Reading the result honestly:**

- A 67.9% melanoma recall means the model misses roughly 1 in 3 real melanomas. **That is unsafe as a real screen** — it's why the project ships with the "not a diagnosis" framing baked into the UI.
- It *over*-flags melanoma on benign moles (false alarms), which is the safer direction to err but still annoying.
- Training longer past 20 epochs barely moved the needle — going from 5 → 20 epochs only nudged mel recall from 64.2% → 67.9%. The real bottleneck is **class imbalance** (melanoma is ~11% of the dataset), not training time. Improving from here needs different interventions (focal loss, stronger melanoma-only augmentation, ensembling), not more epochs.

See [`docs/ETHICS.md`](docs/ETHICS.md) for the full limitations write-up.

## Roadmap

The plan is **desktop first, then phone**:

- [ ] Train the baseline notebook end-to-end and record results above
- [ ] Add **Grad-CAM** so I can see what the model is actually looking at
- [ ] **Desktop app** — wrap the model in a simple UI: drag in a photo, see the top guesses.
      Starting with a local Gradio/Streamlit app, maybe a real PyQt window later.
- [ ] Deal with the **phone-photo domain gap** (train on smartphone shots, not just dermatoscope)
- [ ] Check fairness **across skin tones** (Fitzpatrick types)
- [ ] **Phone app** — reuse the same model behind an API, build the UI in React Native or Flutter

## License

[MIT](LICENSE) © 2026 David Nguyen

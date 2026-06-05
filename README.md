# 🔬 Skin Condition Classifier (Educational)

A personal project I'm building to learn computer vision properly: take a photo of a skin
lesion and get the **top guesses** across 7 dermatological categories. The model comes first
(this repo), then a **desktop app**, and eventually a **phone app**.

> ⚠️ **This is an educational / triage demo — NOT a medical device and NOT a diagnosis.**
> It produces a statistical best-guess for learning purposes only. It must never be used
> to make real health decisions. If you are worried about a skin lesion, see a licensed
> dermatologist. See [`docs/ETHICS.md`](docs/ETHICS.md) for the full list of limitations.

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

## Project structure

```
skin-condition-classifier/
├── notebooks/
│   └── 01_train_ham10000_efficientnet.ipynb   # ← the main, runnable notebook
├── docs/
│   ├── DATA.md      # how to obtain HAM10000 + dataset caveats
│   └── ETHICS.md    # limitations, bias, and responsible-use notes
├── requirements.txt
├── LICENSE
└── README.md
```

## Results

_Fill these in after your first training run — the notebook prints everything you need:_

- Test macro-F1: `__`
- Test accuracy: `__`
- **Melanoma (`mel`) recall: `__`**  ← the number to watch

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

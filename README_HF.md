---
title: Skin Condition Classifier
emoji: 🩺
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
license: mit
short_description: Educational skin-lesion classifier (HAM10000 + EfficientNet)
---

# Skin Condition Classifier

A learning project, not a diagnostic tool.

Take a photo of a skin lesion (or upload one) and the model returns its top three guesses
across seven dermatology categories from the HAM10000 dataset. Each guess is labelled
benign, pre-cancerous, or cancerous, with a confidence number. If you're worried about
something on your skin, see a dermatologist — that's the only right answer.

**Tech:** fine-tuned EfficientNet-B0 on HAM10000 (~10k dermatoscopic images, 7 classes).
Best validation macro-F1: 0.730. Built with Gradio.

**Source code & limitations:** [github.com/say3u/skin-condition-classifier](https://github.com/say3u/skin-condition-classifier)

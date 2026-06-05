# Ethics, Limitations & Responsible Use

This project is an **educational / triage demo**. Being explicit about what it *cannot* do is
part of doing ML responsibly — and is itself a skill worth demonstrating.

## ⚠️ This is not a medical device

- It does **not** diagnose anything. It returns a statistical best-guess for learning purposes.
- It has not been clinically validated, peer-reviewed, or cleared by any regulator
  (e.g., FDA, EU MDR).
- It must **never** be used to decide whether to seek, delay, or skip medical care.
- A real diagnostic product would require clinical validation, regulatory clearance, and a
  qualified clinician in the loop. This project deliberately stays on the educational side of
  that line.

**If you are worried about a skin lesion, see a licensed dermatologist.** Warning signs for
melanoma include a mole that is changing, asymmetric, has irregular borders, multiple colors,
or is larger than ~6mm ("ABCDE"). When in doubt, get it checked by a human.

## Known limitations

1. **Domain gap (dermatoscope → phone).** Trained on magnified, evenly-lit dermatoscopic
   images. Real phone photos differ in lighting, focus, and scale, so real-world accuracy
   will be **substantially lower** than the notebook's test numbers.

2. **Skin-tone bias.** Public dermatology datasets — HAM10000 included — skew heavily toward
   lighter skin. Models trained on them are well documented to **underperform on darker skin**.
   The reported metrics do **not** guarantee equitable performance across skin tones, and this
   project does not yet measure that. Treat fairness as unverified.

3. **Class imbalance.** With ~67% of data in one class, rare-but-dangerous classes (like
   melanoma) have far fewer examples to learn from, so their predictions are less reliable.

4. **Only 7 categories.** Real skin presents with hundreds of conditions. Anything outside
   these 7 classes will be **forced into a wrong bucket** — the model cannot say "I don't know"
   or "this isn't a lesion." It will confidently mislabel a photo of a wall.

5. **Overconfidence.** Softmax probabilities are **not** calibrated risk estimates. A "92%"
   output does not mean 92% real-world likelihood.

## Why melanoma recall matters most

For a cancer screen, the costly error is a **false negative** — telling someone a melanoma is
benign. That's why the notebook reports **recall** per class and calls out **melanoma recall**
specifically. A high overall accuracy with low melanoma recall is a *dangerous* model, not a
good one.

## Privacy

- Skin images can be sensitive personal/health data. Do not upload other people's photos
  without consent.
- This educational pipeline does not store user images; any future app version handling real
  user data would need encryption, secure storage, account deletion, and a privacy policy.

## Responsible framing if you demo this

Lead with: *"This is a learning project that classifies dermatoscopic images into 7 categories.
It is not a diagnostic tool, and here are the specific reasons it shouldn't be trusted as one…"*
Being able to articulate the limitations is more impressive than the accuracy number.

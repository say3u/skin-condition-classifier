# Data: HAM10000

This project trains on **HAM10000** ("Human Against Machine with 10,000 training images"),
a public dataset of **10,015 dermatoscopic images** labeled across 7 lesion categories.

> **Dermatoscopic** = captured with a *dermatoscope* (a magnifying lens pressed to the skin
> with controlled lighting). This is the single most important caveat in this whole project —
> see "The domain gap" below.

## How to download

### Option A — Kaggle API (what the notebook uses)

1. Make a free [Kaggle](https://www.kaggle.com) account.
2. Go to **Account → Settings → API → "Create New Token"**. This downloads `kaggle.json`.
3. In the Colab notebook, the data cell will prompt you to upload `kaggle.json`, then it runs:
   ```bash
   kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
   ```
4. The notebook unzips it automatically.

> 🔒 **Never commit `kaggle.json`** — it's an API secret. It's already in `.gitignore`.

### Option B — direct from the source

The official release lives on the Harvard Dataverse:
<https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T>
Download, unzip, and point the notebook's `DATA_DIR` variable at the folder.

## What's in the download

- `HAM10000_metadata.csv` — one row per image: `lesion_id`, `image_id`, `dx` (the label),
  `dx_type` (how it was confirmed), `age`, `sex`, `localization`.
- `HAM10000_images_part_1/` and `HAM10000_images_part_2/` — the `.jpg` images
  (filenames match `image_id`, e.g. `ISIC_0027419.jpg`).
- Some `hmnist_*.csv` pixel files — **we ignore these**; we use the real images.

## The labels (`dx`)

| Code | Full name | Malignancy |
|------|-----------|------------|
| `akiec` | Actinic keratoses & intraepithelial carcinoma | pre-malignant |
| `bcc` | Basal cell carcinoma | malignant |
| `bkl` | Benign keratosis-like lesions | benign |
| `df` | Dermatofibroma | benign |
| `mel` | Melanoma | **malignant** |
| `nv` | Melanocytic nevi | benign |
| `vasc` | Vascular lesions | benign |

## Two traps this dataset sets for you

### 1. Severe class imbalance
Roughly **67% of the images are `nv` (common moles)**, while `df` and `vasc` each have
only ~1%. A model that blindly predicts "nv" every time scores ~67% accuracy while being
useless. That's why this project uses **class-weighted loss** and reports **macro-F1** and
**per-class recall** instead of plain accuracy.

### 2. Multiple images of the same lesion
Many lesions were photographed more than once, so several rows share a `lesion_id`. If two
photos of the *same* lesion land in both train and test, your test score is inflated by
**data leakage**. The notebook splits **by `lesion_id`** (grouped split) to prevent this.

## The domain gap (read this before trusting any result)

HAM10000 is **dermatoscopic**. A phone-camera app sees ordinary photos: variable lighting,
focus, distance, and no magnification. A model trained only on HAM10000 will look great in
the notebook and then **degrade badly on real phone photos**. This is expected, not a bug —
closing that gap (with phone-captured data) is a deliberate item on the roadmap.

## Citation

> Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of
> multi-source dermatoscopic images of common pigmented skin lesions.* Sci. Data 5, 180161 (2018).

Check the dataset's license terms on Kaggle / Dataverse before any non-educational use.

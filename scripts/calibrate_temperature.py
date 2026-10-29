"""
Fit the optimal temperature T for post-hoc confidence calibration.

Usage (on a machine with the trained .pt and the HAM10000 dataset downloaded):
    python scripts/calibrate_temperature.py

What it does:
    1. Loads the trained model from skin_classifier_efficientnet_b0.pt
    2. Rebuilds the exact val split from the notebook (same seed, same lesion-group)
    3. Collects (logits, labels) on the val set
    4. Finds T* by minimizing negative log-likelihood via LBFGS
    5. Reports before/after ECE (expected calibration error) and NLL
    6. Writes calibration.json at the project root

Reference: Guo et al. 2017, "On Calibration of Modern Neural Networks" (ICML).
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "skin_classifier_efficientnet_b0.pt"
OUTPUT = ROOT / "calibration.json"
IMG_SIZE = 224
SEED = 42
BATCH_SIZE = 64


# ---------- Data loading (mirrors notebook 01's split) ----------

def build_val_split(data_dir: Path, classes: list[str]) -> pd.DataFrame:
    meta_path = next(data_dir.glob("**/HAM10000_metadata.csv"))
    df = pd.read_csv(meta_path)
    paths = {Path(p).stem: p for p in glob.glob(str(data_dir / "**" / "*.jpg"), recursive=True)}
    df["path"] = df["image_id"].map(paths)
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["label"] = df["dx"].map(class_to_idx)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(gss.split(df, groups=df["lesion_id"]))
    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx, _ = next(gss2.split(temp_df, groups=temp_df["lesion_id"]))
    return temp_df.iloc[val_idx].reset_index(drop=True)


# ---------- Calibration ----------

def collect_logits(model: nn.Module, df: pd.DataFrame, device, preprocess) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(df), BATCH_SIZE):
            chunk = df.iloc[i:i + BATCH_SIZE]
            imgs = torch.stack([preprocess(Image.open(r["path"]).convert("RGB")) for _, r in chunk.iterrows()]).to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_labels.extend(chunk["label"].tolist())
            print(f"  processed {min(i + BATCH_SIZE, len(df))} / {len(df)}", end="\r")
    print()
    return torch.cat(all_logits), torch.tensor(all_labels, dtype=torch.long)


def ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error - mean gap between confidence and accuracy, binned."""
    confidences, predictions = probs.max(dim=1)
    accuracies = (predictions == labels).float()
    boundaries = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.any():
            bin_acc = accuracies[in_bin].mean().item()
            bin_conf = confidences[in_bin].mean().item()
            total += abs(bin_acc - bin_conf) * in_bin.float().mean().item()
    return total


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Minimize NLL on the val set over a single scalar T."""
    T = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=200)
    criterion = nn.CrossEntropyLoss()

    def _closure():
        optimizer.zero_grad()
        loss = criterion(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(T.detach().item())


# ---------- Main ----------

def main():
    data_dir = ROOT / "data" / "ham10000"
    if not MODEL_PATH.exists():
        sys.exit(f"error: model not found at {MODEL_PATH}")
    if not any(data_dir.glob("**/HAM10000_metadata.csv")):
        sys.exit(f"error: HAM10000 not found under {data_dir}. See docs/DATA.md to download.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[calibrate] device: {device}")

    ckpt = torch.load(MODEL_PATH, map_location=device)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", IMG_SIZE)
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    preprocess = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("[calibrate] building val split...")
    val_df = build_val_split(data_dir, classes)
    print(f"[calibrate] val images: {len(val_df)}")

    print("[calibrate] collecting logits...")
    logits, labels = collect_logits(model, val_df, device, preprocess)

    nll_before = F.cross_entropy(logits, labels).item()
    ece_before = ece(F.softmax(logits, dim=1), labels)
    print(f"[calibrate] before: NLL={nll_before:.4f}, ECE={ece_before:.4f}")

    print("[calibrate] fitting T via LBFGS...")
    T = fit_temperature(logits, labels)
    nll_after = F.cross_entropy(logits / T, labels).item()
    ece_after = ece(F.softmax(logits / T, dim=1), labels)
    print(f"[calibrate] after:  NLL={nll_after:.4f}, ECE={ece_after:.4f}, T*={T:.4f}")

    OUTPUT.write_text(json.dumps({
        "temperature": round(T, 4),
        "notes": "Fit by minimizing NLL on the HAM10000 val split via LBFGS.",
        "computed_on": datetime.now().isoformat(timespec="seconds"),
        "val_nll_before": round(nll_before, 4),
        "val_nll_after":  round(nll_after, 4),
        "val_ece_before": round(ece_before, 4),
        "val_ece_after":  round(ece_after, 4),
    }, indent=2), encoding="utf-8")
    print(f"[calibrate] wrote {OUTPUT}")


if __name__ == "__main__":
    main()

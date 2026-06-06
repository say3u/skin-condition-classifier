"""
Populate app/examples/ with a handful of sample images so the desktop app has
click-to-try examples out of the box.

It tries, in order:
  1) Copy 4 random images from a local HAM10000 download (if found).
  2) Otherwise prints clear instructions for adding your own.

Run once:
    python app/get_examples.py
"""

from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "app" / "examples"
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Common spots people end up with HAM10000 on disk
CANDIDATE_HAM_DIRS = [
    PROJECT_ROOT / "data" / "ham10000",
    Path.home() / "Downloads" / "ham10000",
    Path.home() / "Downloads" / "skin-cancer-mnist-ham10000",
]


def find_local_ham_images():
    for base in CANDIDATE_HAM_DIRS:
        if base.exists():
            imgs = list(base.rglob("*.jpg"))
            if len(imgs) > 50:   # arbitrary "looks like the real dataset" threshold
                return base, imgs
    return None, []


def main():
    existing = list(EXAMPLES_DIR.glob("*.jpg")) + list(EXAMPLES_DIR.glob("*.png"))
    if len(existing) >= 4:
        print(f"already have {len(existing)} examples in {EXAMPLES_DIR}. nothing to do.")
        return

    base, imgs = find_local_ham_images()
    if imgs:
        print(f"found HAM10000 images at {base} ({len(imgs)} total). copying 4 random ones...")
        random.seed(7)
        for i, src in enumerate(random.sample(imgs, 4), start=1):
            dst = EXAMPLES_DIR / f"example_{i}.jpg"
            shutil.copy(src, dst)
            print(f"  -> {dst.name}")
        print("done. restart the app and you'll see click-to-try examples.")
        return

    print(
        "couldn't find a local HAM10000 download.\n"
        "two ways to add examples:\n"
        f"  1) drop 3-5 .jpg files into {EXAMPLES_DIR} yourself\n"
        "  2) download HAM10000 to data/ham10000/ (see docs/DATA.md), then re-run this script\n"
        "the app still works without examples - this is just for the click-to-try buttons."
    )


if __name__ == "__main__":
    main()

# Click-to-try examples for the desktop app

Drop 3–5 `.jpg` images here and the app will show them as click-to-try buttons under the
upload area. Anything goes — HAM10000 images, your own phone photos, whatever you want
visitors to try first.

Quickest way to fill this folder: from the project root, run

```
python app/get_examples.py
```

which copies 4 random images out of a local HAM10000 download (if you have one).
Or just drag images in manually.

Files in this folder are **gitignored** so you don't accidentally commit medical images.

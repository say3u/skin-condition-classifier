"""
Hugging Face Spaces entry point.

HF Spaces runs `python app.py` at the repo root. The real app lives in app/desktop_app.py;
this thin wrapper just imports the demo and launches it with HF-friendly settings
(no share=True, no auto-open-browser - HF provides its own URL).

For local development, run `python app/desktop_app.py` directly instead.
"""

from app.desktop_app import demo, theme, CUSTOM_CSS

demo.launch(theme=theme, css=CUSTOM_CSS)

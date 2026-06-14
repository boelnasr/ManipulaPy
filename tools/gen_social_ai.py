#!/usr/bin/env python3
"""Generate a ManipulaPy social-preview banner via the OpenAI Images API.

Zero third-party deps beyond Pillow (already installed). Reads the API key from
the OPENAI_API_KEY environment variable — nothing is hardcoded.

Usage:
    export OPENAI_API_KEY=sk-...
    python3 tools/gen_social_ai.py
    # optional: OPENAI_IMAGE_MODEL=dall-e-3 python3 tools/gen_social_ai.py

Output:
    assets/social-ai-raw.png   raw model output
    assets/social-preview.png  cropped/resized to GitHub's 1280x640 spec
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from io import BytesIO

from PIL import Image

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    sys.exit("error: set OPENAI_API_KEY in your environment first.")

MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
# gpt-image-1 landscape = 1536x1024; dall-e-3 landscape = 1792x1024 (closer to 2:1)
SIZE = "1792x1024" if MODEL == "dall-e-3" else "1536x1024"

PROMPT = (
    "A sleek, modern GitHub social-preview banner for an open-source robotics "
    "Python library called 'ManipulaPy'. Wide landscape, dark background "
    "(#0d1117 GitHub-dark) with a subtle blue technical grid. On the right, a "
    "clean 3D render of a 6-DOF robotic manipulator arm executing a smooth "
    "curved trajectory path glowing in cyan-blue, with a small target object on "
    "a reflective checkerboard floor. On the left, large bold white sans-serif "
    "title 'ManipulaPy' and a smaller muted subtitle 'GPU-Accelerated Robotic "
    "Manipulation, Perception & Control'. Minimalist, high-contrast, "
    "professional developer-tool aesthetic, soft blue accent lighting, lots of "
    "negative space, no clutter, flat vector-meets-3D style. Avoid gibberish "
    "text, watermarks, busy background, cartoonish look."
)

payload = {"model": MODEL, "prompt": PROMPT, "size": SIZE, "n": 1}
if MODEL == "dall-e-3":
    payload["quality"] = "hd"

req = urllib.request.Request(
    "https://api.openai.com/v1/images/generations",
    data=json.dumps(payload).encode(),
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    method="POST",
)

print(f"Requesting {MODEL} @ {SIZE} ...")
try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.load(resp)
except urllib.error.HTTPError as e:
    sys.exit(f"OpenAI API error {e.code}: {e.read().decode()[:500]}")

item = data["data"][0]
if item.get("b64_json"):
    raw = base64.b64decode(item["b64_json"])
else:  # dall-e-3 may return a URL
    with urllib.request.urlopen(item["url"], timeout=120) as r:
        raw = r.read()

os.makedirs("assets", exist_ok=True)
with open("assets/social-ai-raw.png", "wb") as f:
    f.write(raw)

# center-crop to 2:1, then resize to GitHub's 1280x640
im = Image.open(BytesIO(raw)).convert("RGB")
w, h = im.size
target = 2.0
if w / h > target:  # too wide -> trim sides
    nw = int(h * target)
    im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
else:  # too tall -> trim top/bottom
    nh = int(w / target)
    im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
im = im.resize((1280, 640), Image.LANCZOS)
im.save("assets/social-preview.png", "PNG", optimize=True)

kb = os.path.getsize("assets/social-preview.png") / 1024
print(f"OK -> assets/social-preview.png (1280x640, {kb:.0f} KB)")
print("     raw model output -> assets/social-ai-raw.png")

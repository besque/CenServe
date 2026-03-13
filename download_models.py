import urllib.request
import os
import sys

MODELS_DIR = "blurberry/models"
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = {
    "plate_best.pt": "https://github.com/besque/BlurBerry/releases/download/v0.1-models/plate_best.pt",
    "card_best.pt":  "https://github.com/besque/BlurBerry/releases/download/v0.1-models/card_best.pt",
    "plate_best.onnx": "https://github.com/besque/BlurBerry/releases/download/v0.1-models/plate_best.onnx",
    "card_best.onnx": "https://github.com/besque/BlurBerry/releases/download/v0.1-models/card_best.onnx"
}

def download(filename, url):
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest):
        print(f"  already exists: {dest}")
        return
    print(f"  downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"  saved {dest} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  manually download from: {url}")
        print(f"  and save to: {dest}")

print("BlurBerry — downloading model weights\n")
for name, url in MODELS.items():
    download(name, url)
print("\nDone. Run python main.py")
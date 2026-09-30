# offlineisbetter
# Copyright (c) 2026- offlineisbetter
# 
# Server for offlineisbetter

import json
import os
from pathlib import Path
import requests
import sys
import tarfile

from fastapi import FastAPI
import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer
import uvicorn

# Global variables
model = None
tokenizer = None
classes = None

# Cache directory
CACHE_DIR = Path.home() / ".cache" / "offlineisbetter"

# API application
app = FastAPI()

def load_model(model_triple):
    """
    Load a model from its triple.
    """
    global model
    global tokenizer
    global classes

    # Check if model exists already
    checkpoint = CACHE_DIR / model_triple
    tar = f"{model_triple}.tar"
    check = os.path.exists(checkpoint)

    if not check:
        # Download release metadata
        url = f"https://api.github.com/repos/offlineisbetter/offlineisbetter/releases/latest"
        response = requests.get(url)
        response.raise_for_status()
        metadata = response.json()

        # Find download URL
        download_url = None
        for asset in metadata.get("assets", []):
            if asset["name"] == tar:
                download_url = asset["browser_download_url"]
                break
        if download_url is None:
            raise ValueError(f"invalid model triple '{model_triple}'")
        
        # Stream to file
        checkpoint.mkdir(parents=True, exist_ok=True)
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(CACHE_DIR / tar, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Inflate the archive
        with tarfile.open(CACHE_DIR / tar, "r:*") as tar:
            tar.extractall(path=CACHE_DIR, filter="data")

    # Load model
    model = ort.InferenceSession(checkpoint / "model_quantized.pt", providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(
        str(checkpoint / "tokenizer.json")
    )
    with open(checkpoint / "offlineisbetter.json", "r") as f:
        classes = json.load(f)

@app.post("/classify")
def offlineisbetter(data: dict):
    """
    Run the offlineisbetter classification model.
    """
    global model
    global tokenizer
    global classes
    enc = tokenizer.encode(data["text"])
    tok = {
        "input_ids": np.array(enc.ids),
        "attention_mask": np.array(enc.attention_mask),
    }
    results = model.run(None, tok)[0]
    c = {v: k for k, v in classes.items()}
    return {
        "label": c[int(np.argmax(results))],
        "text": data["text"],
        "logits": [float(f) for f in results.flatten()],
    }

def main():
    """
    Run the offlineisbetter server.
    """
    if len(sys.argv) < 4:
        print(f"offlineisbetter [host] [port] [triple]")
        return

    # Get command line arguments
    _, host, port, triple = sys.argv[:4]
    try:
        port = int(port)
    except:
        print(f"offlineisbetter [host] [port] [triple]")
        print(f"[port] must be an integer")

    # Load model
    load_model(triple)

    # Run
    try:
        uvicorn.run(app, host=host, port=port)
    except:
        print(f"offlineisbetter [host] [port] [triple]")
        print(f"[host] must be a valid IP address")

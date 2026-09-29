# offlineisbetter
# Copyright (c) 2026- offlineisbetter
# 
# Server for offlineisbetter

import json
from pathlib import Path
import sys

from fastapi import FastAPI
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import uvicorn

# Global variables
model = None
tokenizer = None
classes = None

HOST = "127.0.0.1"
PORT = 8001

# API application
app = FastAPI()

def load_model(checkpoint):
    """
    Load a model, tokenizer, and class dict from checkpoint.
    """
    global model
    global tokenizer
    global classes
    model = ort.InferenceSession(checkpoint / "classifier_int8.pt", providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    with open(checkpoint / "classes.json", "r") as f:
        classes = json.load(f)

@app.post("/offlineisbetter")
def offlineisbetter(data: dict):
    """
    Run the offlineisbetter classification model.
    """
    global model
    global tokenizer
    global classes
    tok = tokenizer(data["text"])
    results = model.run(None, tok)[0]
    c = {v: k for k, v in classes.items()}
    return json.dumps({
        "label": c[int(np.argmax(results))],
        "text": data["text"],
        "logits": [float(f) for f in results.flatten()],
    })

def main():
    """
    Run the offlineisbetter server.
    """
    if len(sys.argv) < 4:
        print(f"offlineisbetter [host] [port] [checkpoint]")
        return

    # Get command line arguments
    _, host, port, checkpoint = sys.argv[:4]
    try:
        port = int(port)
    except:
        print(f"offlineisbetter [host] [port] [checkpoint")
        print(f"[port] must be an integer")

    # Load model
    load_model(Path(checkpoint))

    # Run
    try:
        uvicorn.run(app, host=host, port=port)
    except:
        print(f"offlineisbetter [host] [port] [checkpoint")
        print(f"[host] must be a valid IP address")

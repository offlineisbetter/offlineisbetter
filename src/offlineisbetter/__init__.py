# offlineisbetter
# Copyright (c) 2026- offlineisbetter
# 
# Server for offlineisbetter

import json
from pathlib import Path
import sys

from fastapi import FastAPI
import torch
import uvicorn

from .models import ClassificationModel

# Global variables
model = None
tokenizer = None
classes = None

HOST = "127.0.0.1"
PORT = 8001

# API application
app = FastAPI()

@app.post("/offlineisbetter")
def offlineisbetter(data: dict):
    """
    Run the offlineisbetter classification model.
    """
    global model
    global tokenizer
    global classes
    tok = tokenizer(data["text"], return_tensors="pt")
    results = model(**tok)
    c = {v: k for k, v in classes.items()}
    return json.dumps({
        "label": c[torch.argmax(results["logits"]).item()],
        "text": data["text"],
        "logits": [l.item() for l in results["logits"].squeeze()],
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
    global model
    global tokenizer
    global classes
    model, tokenizer, classes = ClassificationModel.load(Path(checkpoint))

    # Run
    try:
        uvicorn.run(app, host=host, port=port)
    except:
        print(f"offlineisbetter [host] [port] [checkpoint")
        print(f"[host] must be a valid IP address")

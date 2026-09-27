# offlineisbetter
# Copyright (c) 2026- offlineisbetter
# 
# Server for offlineisbetter

from fastapi import FastAPI
import uvicorn

from .models import ClassificationModel

HOST = "127.0.0.1"
PORT = 8001

app = FastAPI()

@app.post("/hello")
def hello(data: dict):
    return data | {"status": "ok"}

def main():
    """
    Run the offlineisbetter client.
    """
    uvicorn.run(app, host=HOST, port=PORT)

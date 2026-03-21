# Non-SAP backend — ai-trading-platform v2
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok", "message": "Non-SAP backend running"}

@app.get("/health")
def health():
    return {"healthy": True}

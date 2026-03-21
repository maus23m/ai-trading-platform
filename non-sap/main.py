# Non-SAP backend — ai-trading-platform
import os
from fastapi import FastAPI
from supabase import create_client

app = FastAPI()

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Non-SAP backend running"}

@app.get("/health")
def health():
    return {"healthy": True}

@app.get("/runs")
def get_runs():
    result = supabase.table("backtest_runs").select("*").execute()
    return result.data

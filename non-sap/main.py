# Non-SAP backend — ai-trading-platform v3
import os
from fastapi import FastAPI
from supabase import create_client
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import pytz
from pydantic import BaseModel
from agent import run_agent

app = FastAPI()

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)

trading_client = TradingClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
    paper=True
)

data_client = StockHistoricalDataClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"]
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

@app.get("/account")
def get_account():
    account = trading_client.get_account()
    return {
        "cash": float(account.cash),
        "portfolio_value": float(account.portfolio_value),
        "buying_power": float(account.buying_power),
        "paper": True
    }

@app.get("/bars/{symbol}")
def get_bars(symbol: str):
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime(2024, 1, 1, tzinfo=pytz.UTC),
        end=datetime(2024, 12, 31, tzinfo=pytz.UTC)
    )
    bars = data_client.get_stock_bars(request)
    df = bars.df
    return {"symbol": symbol, "bars": len(df), "data": df.reset_index().to_dict(orient="records")}

class AgentRequest(BaseModel):
    goal: str
    min_win_rate: float = 0.5
    max_drawdown: float = 0.2

@app.post("/run-agent")
def run_agent_endpoint(request: AgentRequest):
    try:
        result = run_agent(
            goal=request.goal,
            min_win_rate=request.min_win_rate,
            max_drawdown=request.max_drawdown
        )
        return {
            "termination_reason": result["termination_reason"],
            "iterations": result["iteration"],
            "constraints_met": result["constraints_met"],
            "best_result": result["best_result"],
            "all_results": result["results"]
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "detail": traceback.format_exc()
        }

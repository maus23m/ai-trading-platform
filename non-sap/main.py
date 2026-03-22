# Non-SAP backend — ai-trading-platform
import os
from fastapi import FastAPI
from supabase import create_client
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

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
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31)
    )
    bars = data_client.get_stock_bars(request)
    df = bars.df
    return {"symbol": symbol, "bars": len(df), "data": df.reset_index().to_dict(orient="records")}

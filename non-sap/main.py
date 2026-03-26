# Non-SAP backend — ai-trading-platform v6
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import pytz
from pydantic import BaseModel
from agent import run_agent
from debate import run_debate
from indicator_tester import test_indicators_from_text
from backtester import run_walkforward_from_text
from scoring_optimiser import run_optimisation_from_text

app = FastAPI()

# CORS — allows dashboard to call this API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# ─── DASHBOARD ────────────────────────────────────────────────────────────────

@app.get("/dashboard", include_in_schema=False)
def serve_dashboard():
    """Serve the trading dashboard HTML — open this URL in your browser."""
    return FileResponse("dashboard.html", media_type="text/html")


# ─── HEALTH & CONFIG ─────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Non-SAP backend running", "dashboard": "/dashboard"}

@app.get("/health")
def health():
    return {"healthy": True}

@app.get("/config", include_in_schema=False)
def get_config():
    """Serve public config to the dashboard — reads from Cloud Run env vars.
    Only exposes the Supabase anon key (safe to expose) and URL.
    Never exposes Alpaca or Anthropic keys."""
    return {
        "supabase_url": os.environ.get("SUPABASE_URL", ""),
        "supabase_anon_key": os.environ.get("SUPABASE_KEY", "")
    }


# ─── DATA ────────────────────────────────────────────────────────────────────

@app.get("/runs")
def get_runs():
    result = supabase.table("backtest_runs").select("*").order("id", desc=False).execute()
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
    return {
        "symbol": symbol,
        "bars": len(df),
        "data": df.reset_index().to_dict(orient="records")
    }


# ─── AGENT ───────────────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    goal: str
    min_win_rate: float = 0.5
    max_drawdown: float = 0.2
    auto_trade: bool = True      # if True and constraints met, place trade automatically
    auto_trade_qty: int = 1      # shares to buy when auto-trading

@app.post("/run-agent")
def run_agent_endpoint(request: AgentRequest):
    try:
        result = run_agent(
            goal=request.goal,
            min_win_rate=request.min_win_rate,
            max_drawdown=request.max_drawdown
        )

        auto_trade_result = None

        # ── AUTO-TRADE: if strategy passed constraints, place trade automatically ──
        if request.auto_trade and result.get("constraints_met"):
            best = result.get("best_result", {})
            symbol = best.get("symbol", "AAPL")
            try:
                order = trading_client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol,
                        qty=request.auto_trade_qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                )
                auto_trade_result = {
                    "triggered": True,
                    "symbol": symbol,
                    "qty": request.auto_trade_qty,
                    "order_id": str(order.id),
                    "status": str(order.status)
                }
            except Exception as e:
                auto_trade_result = {
                    "triggered": False,
                    "error": str(e)
                }
        else:
            auto_trade_result = {
                "triggered": False,
                "reason": "constraints not met" if not result.get("constraints_met") else "auto_trade disabled"
            }

        return {
            "termination_reason": result["termination_reason"],
            "iterations": result["iteration"],
            "constraints_met": result["constraints_met"],
            "best_result": result["best_result"],
            "all_results": result["results"],
            "supabase_error": result.get("supabase_error", ""),
            "auto_trade": auto_trade_result
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "detail": traceback.format_exc()
        }


# ─── MANUAL TRADE ────────────────────────────────────────────────────────────

class TradeRequest(BaseModel):
    symbol: str
    qty: int
    side: str

@app.post("/trade")
def place_trade(request: TradeRequest):
    try:
        side = OrderSide.BUY if request.side.lower() == "buy" else OrderSide.SELL
        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol=request.symbol,
                qty=request.qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
        )
        return {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "qty": order.qty,
            "side": str(order.side),
            "status": str(order.status),
            "paper": True
        }
    except Exception as e:
        return {"error": str(e)}


# ─── DEBATE ──────────────────────────────────────────────────────────────────

@app.get("/debate", include_in_schema=False)
def serve_debate():
    """Serve the dual-agent debate tool — open this URL in your browser."""
    return FileResponse("debate.html", media_type="text/html")

class DebateRequest(BaseModel):
    topic: str
    context: str = ""
    max_rounds: int = 3

@app.post("/run-debate")
def run_debate_endpoint(request: DebateRequest):
    try:
        result = run_debate(
            topic=request.topic,
            context=request.context,
            max_rounds=max(1, min(10, request.max_rounds))
        )
        return result
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "detail": traceback.format_exc()
        }


# ─── INDICATOR TESTER ────────────────────────────────────────────────────────

@app.get("/indicator-tester", include_in_schema=False)
def serve_indicator_tester():
    """Serve the indicator comparison tester UI."""
    return FileResponse("indicator_tester.html", media_type="text/html")

class IndicatorTestRequest(BaseModel):
    symbols: list = ["AAPL","MSFT","NVDA","JPM","GS","JNJ","UNH","AMZN","WMT","XOM"]
    start_year: int = 2020
    end_year: int = 2024

@app.post("/run-indicator-test")
def run_indicator_test(request: IndicatorTestRequest):
    try:
        result = test_indicators_from_text(
            symbols=request.symbols,
            start_year=request.start_year,
            end_year=request.end_year
        )
        return result
    except Exception as e:
        import traceback
        return {"error": str(e), "detail": traceback.format_exc()}


# ─── BACKTESTER ───────────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    architect_text: str
    symbols: list = ["AAPL","MSFT","NVDA","JPM","GS","JNJ","UNH","AMZN","WMT","XOM"]
    start_year: int = 2015
    end_year: int = 2024
    train_years: int = 2
    test_years: int = 1

@app.post("/run-backtest")
def run_backtest(request: BacktestRequest):
    try:
        result = run_walkforward_from_text(
            architect_text=request.architect_text,
            symbols=request.symbols,
            start_year=request.start_year,
            end_year=request.end_year,
            train_years=request.train_years,
            test_years=request.test_years
        )
        return {"result": result}
    except Exception as e:
        import traceback
        return {"error": str(e), "detail": traceback.format_exc()}


# ─── SCORING OPTIMISER ───────────────────────────────────────────────────────

class OptimiserRequest(BaseModel):
    architect_text: str
    symbols: list = ["AAPL","MSFT","NVDA","JPM","GS","JNJ","UNH","AMZN","WMT","XOM"]
    start_year: int = 2015
    end_year: int = 2024

@app.post("/run-optimiser")
def run_optimiser(request: OptimiserRequest):
    try:
        result = run_optimisation_from_text(
            architect_text=request.architect_text,
            symbols=request.symbols,
            start_year=request.start_year,
            end_year=request.end_year
        )
        return {"result": result}
    except Exception as e:
        import traceback
        return {"error": str(e), "detail": traceback.format_exc()}

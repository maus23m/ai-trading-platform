import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import pytz
from supabase import create_client

# Models
planner_llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

# Alpaca data client
data_client = StockHistoricalDataClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"]
)

# State definition
class BacktestState(TypedDict):
    goal: str
    iteration: int
    max_iterations: int
    results: list
    constraints_met: bool
    termination_reason: str
    symbol: str
    sma_short: int
    sma_long: int
    min_win_rate: float
    max_drawdown: float
    risk_score: float
    live_trading_enabled: bool
    best_result: dict
    supabase_error: str

# Node 1 — Planner
def planner(state: BacktestState) -> BacktestState:
    prompt = f"""You are a trading strategy planner.
Goal: {state['goal']}
Iteration: {state['iteration']}
Previous results: {state['results']}

Suggest SMA crossover parameters to test.
Respond in exactly this format:
SYMBOL: AAPL
SMA_SHORT: 10
SMA_LONG: 30

Pick different values from previous iterations if any failed."""

    response = planner_llm.invoke(prompt)
    text = response.content

    symbol = "AAPL"
    sma_short = 10
    sma_long = 30

    for line in text.split("\n"):
        if "SYMBOL:" in line:
            symbol = line.split(":")[1].strip()
        if "SMA_SHORT:" in line:
            sma_short = int(line.split(":")[1].strip())
        if "SMA_LONG:" in line:
            sma_long = int(line.split(":")[1].strip())

    return {**state, "symbol": symbol, "sma_short": sma_short, "sma_long": sma_long}

# Node 2 — Backtest
def backtest(state: BacktestState) -> BacktestState:
    try:
        request = StockBarsRequest(
            symbol_or_symbols=state["symbol"],
            timeframe=TimeFrame.Day,
            start=datetime(2024, 1, 1, tzinfo=pytz.UTC),
            end=datetime(2024, 12, 31, tzinfo=pytz.UTC)
        )
        bars = data_client.get_stock_bars(request)
        df = bars.df.reset_index()

        closes = df["close"].tolist()
        short = state["sma_short"]
        long = state["sma_long"]

        trades = []
        in_trade = False
        entry_price = 0

        for i in range(long, len(closes)):
            sma_s = sum(closes[i-short:i]) / short
            sma_l = sum(closes[i-long:i]) / long
            prev_sma_s = sum(closes[i-short-1:i-1]) / short
            prev_sma_l = sum(closes[i-long-1:i-1]) / long

            if prev_sma_s <= prev_sma_l and sma_s > sma_l and not in_trade:
                entry_price = closes[i]
                in_trade = True
            elif prev_sma_s >= prev_sma_l and sma_s < sma_l and in_trade:
                pnl = (closes[i] - entry_price) / entry_price
                trades.append(pnl)
                in_trade = False

        if not trades:
            result = {
                "symbol": state["symbol"],
                "sma_short": short,
                "sma_long": long,
                "total_return": 0,
                "win_rate": 0,
                "max_drawdown": 1,
                "trades": 0
            }
        else:
            wins = [t for t in trades if t > 0]
            win_rate = len(wins) / len(trades)
            total_return = sum(trades)
            equity = [1.0]
            for t in trades:
                equity.append(equity[-1] * (1 + t))
            peak = equity[0]
            max_dd = 0
            for e in equity:
                if e > peak:
                    peak = e
                dd = (peak - e) / peak
                if dd > max_dd:
                    max_dd = dd

            result = {
                "symbol": state["symbol"],
                "sma_short": short,
                "sma_long": long,
                "total_return": round(total_return, 4),
                "win_rate": round(win_rate, 4),
                "max_drawdown": round(max_dd, 4),
                "trades": len(trades)
            }

    except Exception as e:
        result = {
            "symbol": state["symbol"],
            "sma_short": state["sma_short"],
            "sma_long": state["sma_long"],
            "error": str(e),
            "total_return": 0,
            "win_rate": 0,
            "max_drawdown": 1,
            "trades": 0
        }

    new_results = state["results"] + [result]
    best = state.get("best_result", {})
    if result.get("total_return", 0) > best.get("total_return", -999):
        best = result

    return {**state, "results": new_results, "best_result": best}

# Node 3 — Evaluator
def evaluator(state: BacktestState) -> BacktestState:
    latest = state["results"][-1] if state["results"] else {}

    win_rate = latest.get("win_rate", 0)
    max_dd = latest.get("max_drawdown", 1)

    constraints_met = (
        win_rate >= state["min_win_rate"] and
        max_dd <= state["max_drawdown"]
    )

    return {
        **state,
        "constraints_met": constraints_met,
        "iteration": state["iteration"] + 1
    }

# Node 4 — Reporter
def reporter(state: BacktestState) -> BacktestState:
    if state["constraints_met"]:
        reason = "success"
    elif state["iteration"] >= state["max_iterations"]:
        reason = "max_iterations_reached"
    else:
        reason = "unknown"

    try:
        supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_KEY"]
        )
        best = state.get("best_result", {})
        supabase.table("backtest_runs").insert({
            "goal": state["goal"],
            "iteration": state["iteration"],
            "parameters": {
                "symbol": best.get("symbol"),
                "sma_short": best.get("sma_short"),
                "sma_long": best.get("sma_long")
            },
            "total_return": best.get("total_return", 0),
            "max_drawdown": best.get("max_drawdown", 0),
            "win_rate": best.get("win_rate", 0),
            "constraints_met": state["constraints_met"],
            "termination_reason": reason
        }).execute()
        return {**state, "termination_reason": reason, "supabase_error": ""}
    except Exception as e:
        return {**state, "termination_reason": reason, "supabase_error": str(e)}

# Routing logic
def should_continue(state: BacktestState) -> str:
    if state["constraints_met"]:
        return "reporter"
    if state["iteration"] >= state["max_iterations"]:
        return "reporter"
    return "planner"

# Build the graph
def build_graph():
    graph = StateGraph(BacktestState)
    graph.add_node("planner", planner)
    graph.add_node("backtest", backtest)
    graph.add_node("evaluator", evaluator)
    graph.add_node("reporter", reporter)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "backtest")
    graph.add_edge("backtest", "evaluator")
    graph.add_conditional_edges("evaluator", should_continue)
    graph.add_edge("reporter", END)
    return graph.compile()

# Run function
def run_agent(goal: str, min_win_rate: float = 0.5, max_drawdown: float = 0.2):
    app = build_graph()
    initial_state = BacktestState(
        goal=goal,
        iteration=0,
        max_iterations=3,
        results=[],
        constraints_met=False,
        termination_reason="",
        symbol="AAPL",
        sma_short=10,
        sma_long=30,
        min_win_rate=min_win_rate,
        max_drawdown=max_drawdown,
        risk_score=0.0,
        live_trading_enabled=False,
        best_result={},
        supabase_error=""
    )
    final_state = app.invoke(
        initial_state,
        config={"recursion_limit": 10}
    )
    return final_state

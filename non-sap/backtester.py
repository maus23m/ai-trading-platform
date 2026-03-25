# backtester.py — Gate 3: Walk-Forward Backtester Tool
# Called automatically by debate graph after each architect turn
# Extracts strategy parameters from free text, runs walk-forward backtest,
# returns human-readable results for injection into critic/architect context

import os
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pytz
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

data_client = StockHistoricalDataClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"]
)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

SLIPPAGE       = 0.0005  # 0.05% per side
COMMISSION     = 0.0010  # 0.10% round-trip
ROUND_TRIP_COST = SLIPPAGE * 2 + COMMISSION

DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "JPM", "GS",
    "JNJ",  "UNH",  "AMZN", "WMT", "XOM"
]

# ─── PARAMETER EXTRACTOR ──────────────────────────────────────────────────────

def extract_parameters(text: str) -> Dict:
    """
    Extract strategy parameters from architect free text.
    Returns dict with: indicators, hold_periods, entry_rules, exit_rules
    """
    text_lower = text.lower()
    params = {
        "indicators": [],
        "hold_periods": [],
        "entry_threshold": 0.0,
        "stop_loss": None,
        "raw_text": text
    }

    # Known indicators to look for
    known = [
        "roc", "momentum", "macd", "rsi", "kama", "frama",
        "ema", "sma", "obv", "mfi", "roc20", "roc12", "roc10"
    ]
    for ind in known:
        if re.search(r'\b' + re.escape(ind) + r'\b', text_lower):
            # Normalise roc20 → roc with period 20
            if ind not in params["indicators"]:
                params["indicators"].append(ind)

    # Extract hold periods — look for numbers near "day", "period", "hold"
    hold_matches = re.findall(
        r'(\d+)\s*[-\s]?\s*(?:day|days|period|bar|bars|hold)',
        text_lower
    )
    for m in hold_matches:
        p = int(m)
        if 3 <= p <= 60 and p not in params["hold_periods"]:
            params["hold_periods"].append(p)

    # Extract ROC period if specified (e.g. "ROC 20", "ROC(20)", "20-period ROC")
    roc_period = re.findall(
        r'roc\s*[\(\[]?\s*(\d+)\s*[\)\]]?|(\d+)\s*[-\s]period\s+roc',
        text_lower
    )
    for m in roc_period:
        p = int(m[0] or m[1])
        if ("roc", p) not in params.get("roc_periods", []):
            params.setdefault("roc_periods", []).append(p)

    # Extract stop loss
    stop_matches = re.findall(
        r'stop\s*(?:loss)?\s*(?:at|of|[-:])?\s*[-]?(\d+(?:\.\d+)?)\s*%',
        text_lower
    )
    if stop_matches:
        params["stop_loss"] = float(stop_matches[0]) / 100

    # Default hold periods if none found
    if not params["hold_periods"]:
        params["hold_periods"] = [15]  # default

    return params

# ─── DATA FETCHER ─────────────────────────────────────────────────────────────

def fetch_bars(symbol: str, start_year: int, end_year: int) -> List[dict]:
    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=datetime(start_year, 1, 1, tzinfo=pytz.UTC),
            end=datetime(end_year, 12, 31, tzinfo=pytz.UTC)
        )
        bars = data_client.get_stock_bars(request)
        df = bars.df.reset_index()
        return df.to_dict(orient="records")
    except Exception:
        return []

# ─── INDICATORS ───────────────────────────────────────────────────────────────

def compute_roc(closes: List[float], period: int = 20) -> List[Optional[float]]:
    result = [None] * len(closes)
    for i in range(period, len(closes)):
        if closes[i - period] != 0:
            result[i] = (closes[i] - closes[i - period]) / closes[i - period]
    return result

def compute_momentum(closes: List[float], period: int = 20) -> List[Optional[float]]:
    """Raw price momentum — close minus close N periods ago"""
    result = [None] * len(closes)
    for i in range(period, len(closes)):
        result[i] = closes[i] - closes[i - period]
    return result

def compute_macd(closes: List[float], fast: int = 12, slow: int = 26) -> List[Optional[float]]:
    k_f, k_s = 2/(fast+1), 2/(slow+1)
    ema_f = [None]*len(closes)
    ema_s = [None]*len(closes)
    for i in range(len(closes)):
        if i < fast-1: continue
        ema_f[i] = sum(closes[:fast])/fast if i == fast-1 else closes[i]*k_f + ema_f[i-1]*(1-k_f)
    for i in range(len(closes)):
        if i < slow-1: continue
        ema_s[i] = sum(closes[:slow])/slow if i == slow-1 else closes[i]*k_s + ema_s[i-1]*(1-k_s)
    return [f-s if f and s else None for f,s in zip(ema_f, ema_s)]

def compute_rsi(closes: List[float], period: int = 14) -> List[Optional[float]]:
    result = [None]*len(closes)
    if len(closes) < period+1: return result
    gains, losses = [], []
    for i in range(1, period+1):
        d = closes[i]-closes[i-1]
        gains.append(max(d,0)); losses.append(max(-d,0))
    ag, al = sum(gains)/period, sum(losses)/period
    for i in range(period, len(closes)):
        d = closes[i]-closes[i-1]
        ag = (ag*(period-1)+max(d,0))/period
        al = (al*(period-1)+max(-d,0))/period
        result[i] = 100 if al==0 else 100-(100/(1+ag/al))
    return result

def get_signals(indicator: str, closes: List[float],
                roc_period: int = 20) -> List[int]:
    """Convert indicator values to signals: +1 buy, -1 sell, 0 hold"""
    if indicator in ("roc", "roc20", "roc12", "roc10"):
        period = roc_period
        if "roc" in indicator and len(indicator) > 3:
            try: period = int(indicator[3:])
            except: pass
        values = compute_roc(closes, period)
        return [1 if v and v > 0 else (-1 if v and v < 0 else 0) for v in values]

    elif indicator == "momentum":
        values = compute_momentum(closes, roc_period)
        return [1 if v and v > 0 else (-1 if v and v < 0 else 0) for v in values]

    elif indicator == "macd":
        values = compute_macd(closes)
        signals = [0]*len(values)
        for i in range(1, len(values)):
            if values[i] and values[i-1]:
                if values[i] > 0 and values[i-1] <= 0: signals[i] = 1
                elif values[i] < 0 and values[i-1] >= 0: signals[i] = -1
        return signals

    elif indicator == "rsi":
        values = compute_rsi(closes)
        return [1 if v and v < 30 else (-1 if v and v > 70 else 0) for v in values]

    elif indicator in ("ema", "sma", "kama", "frama"):
        period = 20
        ema = [None]*len(closes)
        k = 2/(period+1)
        for i in range(len(closes)):
            if i < period-1: continue
            ema[i] = sum(closes[:period])/period if i==period-1 else closes[i]*k+ema[i-1]*(1-k)
        signals = [0]*len(closes)
        for i in range(1, len(closes)):
            if ema[i] and ema[i-1]:
                if closes[i] > ema[i] and closes[i-1] <= ema[i-1]: signals[i] = 1
                elif closes[i] < ema[i] and closes[i-1] >= ema[i-1]: signals[i] = -1
        return signals

    return [0]*len(closes)

# ─── SINGLE WINDOW BACKTEST ───────────────────────────────────────────────────

def backtest_window(closes: List[float], signals: List[int],
                    hold_period: int,
                    stop_loss: Optional[float] = None) -> Dict:
    """Backtest a single train/test window with realistic costs"""
    trades = []
    i = 0
    while i < len(signals) - hold_period:
        if signals[i] == 1:
            entry = closes[i]
            # Apply stop loss if specified
            exit_price = closes[i + hold_period]
            if stop_loss:
                for j in range(i+1, i+hold_period+1):
                    if (closes[j] - entry) / entry <= -stop_loss:
                        exit_price = closes[j]
                        break
            # Deduct realistic costs
            pnl = (exit_price - entry) / entry - ROUND_TRIP_COST
            trades.append(pnl)
            i += hold_period
        else:
            i += 1

    if not trades:
        return {"trades": 0, "sharpe": 0, "win_rate": 0,
                "total_return": 0, "max_drawdown": 0, "profit_factor": 0}

    wins   = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    win_rate     = len(wins) / len(trades)
    total_return = sum(trades)
    avg_r  = float(np.mean(trades))
    std_r  = float(np.std(trades)) if len(trades) > 1 else 1
    sharpe = (avg_r / std_r) * np.sqrt(252/hold_period) if std_r > 0 else 0

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss   = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss

    # Max drawdown
    equity = [1.0]
    for t in trades:
        equity.append(equity[-1] * (1+t))
    peak, max_dd = equity[0], 0
    for e in equity:
        if e > peak: peak = e
        dd = (peak-e)/peak
        if dd > max_dd: max_dd = dd

    return {
        "trades":        len(trades),
        "win_rate":      round(win_rate, 4),
        "total_return":  round(total_return, 4),
        "sharpe":        round(sharpe, 4),
        "max_drawdown":  round(max_dd, 4),
        "profit_factor": round(profit_factor, 4)
    }

# ─── WALK-FORWARD ENGINE ──────────────────────────────────────────────────────

def walk_forward(bars: List[dict], indicator: str,
                 hold_period: int, stop_loss: Optional[float],
                 train_years: int = 2, test_years: int = 1,
                 roc_period: int = 20) -> Dict:
    """
    Rolling walk-forward: train on N years, test on next M years, roll forward.
    Returns aggregated out-of-sample metrics.
    """
    if len(bars) < 252 * (train_years + test_years):
        return {"error": "insufficient data for walk-forward"}

    closes = [b["close"] for b in bars]
    signals_full = get_signals(indicator, closes, roc_period)

    train_bars  = int(252 * train_years)
    test_bars   = int(252 * test_years)
    step        = test_bars

    oos_trades = []  # all out-of-sample trades
    windows    = []

    start = 0
    while start + train_bars + test_bars <= len(closes):
        test_start = start + train_bars
        test_end   = test_start + test_bars

        test_closes  = closes[test_start:test_end]
        test_signals = signals_full[test_start:test_end]

        result = backtest_window(test_closes, test_signals, hold_period, stop_loss)
        if result["trades"] > 0:
            windows.append(result)

        start += step

    if not windows:
        return {"error": "no trades in any walk-forward window"}

    # Aggregate across windows
    all_sharpes  = [w["sharpe"]        for w in windows]
    all_returns  = [w["total_return"]  for w in windows]
    all_winrates = [w["win_rate"]      for w in windows]
    all_dds      = [w["max_drawdown"]  for w in windows]
    all_pfs      = [w["profit_factor"] for w in windows]
    all_trades   = [w["trades"]        for w in windows]

    return {
        "windows":          len(windows),
        "total_trades":     sum(all_trades),
        "avg_sharpe":       round(float(np.mean(all_sharpes)), 4),
        "std_sharpe":       round(float(np.std(all_sharpes)), 4),
        "avg_win_rate":     round(float(np.mean(all_winrates)), 4),
        "avg_return":       round(float(np.mean(all_returns)), 4),
        "avg_drawdown":     round(float(np.mean(all_dds)), 4),
        "avg_profit_factor":round(float(np.mean(all_pfs)), 4),
        "sharpe_stability": round(1 - float(np.std(all_sharpes)) / (abs(float(np.mean(all_sharpes))) + 0.001), 4),
        "window_results":   windows
    }

# ─── MAIN TOOL FUNCTION ───────────────────────────────────────────────────────

def run_walkforward_from_text(
    architect_text: str,
    symbols: List[str] = None,
    start_year: int = 2015,
    end_year: int = 2024,
    train_years: int = 2,
    test_years: int = 1
) -> str:
    """
    Main tool function called by debate graph.
    Extracts strategy parameters from architect free text,
    runs walk-forward backtest, returns human-readable results.
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    params = extract_parameters(architect_text)
    indicators   = params["indicators"] or ["roc"]
    hold_periods = params["hold_periods"] or [15]
    stop_loss    = params["stop_loss"]
    roc_period   = params.get("roc_periods", [20])[0] if params.get("roc_periods") else 20

    lines = [
        f"WALK-FORWARD BACKTEST RESULTS",
        f"Period: {start_year}–{end_year} | Train: {train_years}yr | Test: {test_years}yr",
        f"Costs: {ROUND_TRIP_COST*100:.2f}% round-trip (slippage + commission)",
        f"Symbols: {len(symbols)} | Indicators found: {indicators} | Hold periods: {hold_periods}",
        "=" * 70
    ]

    if stop_loss:
        lines.append(f"Stop loss: {stop_loss*100:.1f}%")

    all_results = []

    for indicator in indicators:
        for hold in hold_periods:
            symbol_results = []
            for symbol in symbols:
                bars = fetch_bars(symbol, start_year, end_year)
                if not bars:
                    continue
                result = walk_forward(
                    bars, indicator, hold, stop_loss,
                    train_years, test_years, roc_period
                )
                if "error" not in result:
                    symbol_results.append(result)

            if not symbol_results:
                lines.append(f"{indicator.upper()} | hold={hold}d: no data")
                continue

            # Aggregate across symbols
            agg = {
                "indicator":     indicator,
                "hold_period":   hold,
                "symbols":       len(symbol_results),
                "avg_sharpe":    round(float(np.mean([r["avg_sharpe"]        for r in symbol_results])), 4),
                "std_sharpe":    round(float(np.mean([r["std_sharpe"]        for r in symbol_results])), 4),
                "avg_win_rate":  round(float(np.mean([r["avg_win_rate"]      for r in symbol_results])), 4),
                "avg_return":    round(float(np.mean([r["avg_return"]        for r in symbol_results])), 4),
                "avg_drawdown":  round(float(np.mean([r["avg_drawdown"]      for r in symbol_results])), 4),
                "avg_pf":        round(float(np.mean([r["avg_profit_factor"] for r in symbol_results])), 4),
                "stability":     round(float(np.mean([r["sharpe_stability"]  for r in symbol_results])), 4),
                "total_trades":  sum(r["total_trades"] for r in symbol_results),
            }
            all_results.append(agg)

            lines.append(
                f"{indicator.upper()} | hold={hold}d | "
                f"Sharpe={agg['avg_sharpe']:.3f}±{agg['std_sharpe']:.3f} | "
                f"WinRate={agg['avg_win_rate']*100:.1f}% | "
                f"Return={agg['avg_return']*100:.1f}% | "
                f"Drawdown={agg['avg_drawdown']*100:.1f}% | "
                f"ProfitFactor={agg['avg_pf']:.2f} | "
                f"Stability={agg['stability']:.3f} | "
                f"Trades={agg['total_trades']}"
            )

    # Best combination
    if all_results:
        best = max(all_results, key=lambda x: x["avg_sharpe"])
        lines.append("")
        lines.append("BEST COMBINATION (by out-of-sample Sharpe):")
        lines.append(
            f"  {best['indicator'].upper()} with {best['hold_period']}-day hold | "
            f"Sharpe={best['avg_sharpe']:.3f} | "
            f"WinRate={best['avg_win_rate']*100:.1f}% | "
            f"Return={best['avg_return']*100:.1f}% | "
            f"Stability={best['stability']:.3f}"
        )

        # Gate check
        lines.append("")
        lines.append("GATE CHECK (from debate output):")
        lines.append(f"  Sharpe >= 1.0:        {'PASS' if best['avg_sharpe'] >= 1.0 else 'FAIL'} ({best['avg_sharpe']:.3f})")
        lines.append(f"  Win rate >= 50%:      {'PASS' if best['avg_win_rate'] >= 0.5 else 'FAIL'} ({best['avg_win_rate']*100:.1f}%)")
        lines.append(f"  Profit factor >= 1.2: {'PASS' if best['avg_pf'] >= 1.2 else 'FAIL'} ({best['avg_pf']:.2f})")
        lines.append(f"  Drawdown <= 35%:      {'PASS' if best['avg_drawdown'] <= 0.35 else 'FAIL'} ({best['avg_drawdown']*100:.1f}%)")
        lines.append(f"  Trades >= 100:        {'PASS' if best['total_trades'] >= 100 else 'FAIL'} ({best['total_trades']})")

    lines.append("")
    lines.append("Use these out-of-sample results to validate or refine the proposed strategy.")
    lines.append("Sharpe stability (closer to 1.0) indicates consistent performance across time windows.")

    return "\n".join(lines)

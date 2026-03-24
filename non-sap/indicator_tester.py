# indicator_tester.py — Gate 2: Indicator Tool
# Called automatically by the debate graph each round
# Extracts indicator names from free text, runs backtests, returns results

import os
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import pytz
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

data_client = StockHistoricalDataClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"]
)

# ─── KNOWN INDICATORS ─────────────────────────────────────────────────────────
# Parser looks for these names in architect free text

KNOWN_INDICATORS = {
    "kama": "trend",
    "frama": "trend",
    "ema": "trend",
    "sma": "trend",
    "dema": "trend",
    "tema": "trend",
    "macd": "momentum",
    "roc": "momentum",
    "rsi": "momentum",
    "cci": "momentum",
    "stochastic": "momentum",
    "momentum": "momentum",
    "obv": "volume",
    "mfi": "volume",
    "vwap": "volume",
    "adl": "volume",
    "cmf": "volume",
}

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "JPM", "GS", "JNJ", "UNH", "AMZN", "WMT", "XOM"]

# ─── PARSER ───────────────────────────────────────────────────────────────────

def extract_indicators(text: str) -> List[str]:
    """Extract indicator names from architect free text."""
    text_lower = text.lower()
    found = []
    for name in KNOWN_INDICATORS:
        # Match whole word only
        pattern = r'\b' + re.escape(name) + r'\b'
        if re.search(pattern, text_lower):
            if name not in found:
                found.append(name)
    return found

# ─── DATA FETCHER ─────────────────────────────────────────────────────────────

def fetch_bars(symbol: str, start_year: int = 2020, end_year: int = 2024) -> List[dict]:
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

def compute_indicator(name: str, closes: List[float],
                      volumes: List[float], highs: List[float],
                      lows: List[float]) -> List[Optional[float]]:

    n = len(closes)

    if name == "kama":
        period, fast_sc, slow_sc = 10, 2/(2+1), 2/(30+1)
        result = [None] * n
        if n < period + 1:
            return result
        result[period] = closes[period]
        for i in range(period + 1, n):
            direction = abs(closes[i] - closes[i - period])
            volatility = sum(abs(closes[j] - closes[j-1]) for j in range(i-period+1, i+1))
            er = direction / volatility if volatility != 0 else 0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            result[i] = result[i-1] + sc * (closes[i] - result[i-1])
        return result

    elif name == "frama":
        period = 16
        half = period // 2
        result = [None] * n
        if n < period + 1:
            return result
        result[period - 1] = sum(closes[:period]) / period
        for i in range(period, n):
            h1 = max(closes[i-half:i])
            l1 = min(closes[i-half:i])
            h2 = max(closes[i-period:i-half])
            l2 = min(closes[i-period:i-half])
            h3 = max(closes[i-period:i])
            l3 = min(closes[i-period:i])
            n1 = (h1-l1)/half if half > 0 else 0
            n2 = (h2-l2)/half if half > 0 else 0
            n3 = (h3-l3)/period if period > 0 else 0
            if n1+n2 > 0 and n3 > 0:
                d = (np.log(n1+n2) - np.log(n3)) / np.log(2)
            else:
                d = 1
            alpha = max(0.01, min(1.0, float(np.exp(-4.6*(d-1)))))
            result[i] = alpha*closes[i] + (1-alpha)*result[i-1]
        return result

    elif name in ("ema", "sma", "dema", "tema"):
        period = 20
        result = [None] * n
        k = 2 / (period + 1)
        for i in range(n):
            if i < period - 1:
                continue
            if i == period - 1:
                result[i] = sum(closes[:period]) / period
            else:
                result[i] = closes[i]*k + result[i-1]*(1-k)
        return result

    elif name == "macd":
        fast, slow = 12, 26
        k_f, k_s = 2/(fast+1), 2/(slow+1)
        ema_f = [None]*n
        ema_s = [None]*n
        for i in range(n):
            if i < fast-1:
                continue
            if i == fast-1:
                ema_f[i] = sum(closes[:fast])/fast
            else:
                ema_f[i] = closes[i]*k_f + ema_f[i-1]*(1-k_f)
        for i in range(n):
            if i < slow-1:
                continue
            if i == slow-1:
                ema_s[i] = sum(closes[:slow])/slow
            else:
                ema_s[i] = closes[i]*k_s + ema_s[i-1]*(1-k_s)
        return [
            (f - s) if f is not None and s is not None else None
            for f, s in zip(ema_f, ema_s)
        ]

    elif name == "roc":
        period = 12
        result = [None] * n
        for i in range(period, n):
            if closes[i-period] != 0:
                result[i] = (closes[i] - closes[i-period]) / closes[i-period]
        return result

    elif name == "rsi":
        period = 14
        result = [None] * n
        if n < period + 1:
            return result
        gains, losses = [], []
        for i in range(1, period+1):
            d = closes[i] - closes[i-1]
            gains.append(max(d, 0))
            losses.append(max(-d, 0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        for i in range(period, n):
            d = closes[i] - closes[i-1]
            avg_gain = (avg_gain*(period-1) + max(d,0)) / period
            avg_loss = (avg_loss*(period-1) + max(-d,0)) / period
            if avg_loss == 0:
                result[i] = 100
            else:
                result[i] = 100 - (100 / (1 + avg_gain/avg_loss))
        return result

    elif name in ("cci", "stochastic", "momentum"):
        period = 14
        result = [None] * n
        for i in range(period, n):
            result[i] = closes[i] - closes[i-period]
        return result

    elif name == "obv":
        result = [0.0]
        for i in range(1, n):
            if closes[i] > closes[i-1]:
                result.append(result[-1] + volumes[i])
            elif closes[i] < closes[i-1]:
                result.append(result[-1] - volumes[i])
            else:
                result.append(result[-1])
        return result

    elif name == "mfi":
        period = 14
        result = [None] * n
        typical = [(h+l+c)/3 for h,l,c in zip(highs, lows, closes)]
        for i in range(period, n):
            pos = sum(typical[j]*volumes[j] for j in range(i-period, i) if typical[j] > typical[j-1])
            neg = sum(typical[j]*volumes[j] for j in range(i-period, i) if typical[j] < typical[j-1])
            result[i] = 100 if neg == 0 else 100 - (100/(1+pos/neg))
        return result

    elif name in ("vwap", "adl", "cmf"):
        # Simplified volume-weighted signal
        result = [None] * n
        period = 14
        for i in range(period, n):
            tp = [(highs[j]+lows[j]+closes[j])/3 for j in range(i-period, i)]
            vols = volumes[i-period:i]
            total_vol = sum(vols)
            result[i] = sum(t*v for t,v in zip(tp,vols))/total_vol if total_vol > 0 else None
        return result

    return [None] * n

# ─── SIGNAL GENERATOR ─────────────────────────────────────────────────────────

def signals_from_values(values: List, name: str, closes: List[float]) -> List[int]:
    signals = [0] * len(values)
    for i in range(1, len(values)):
        v = values[i]
        vp = values[i-1]
        if v is None or vp is None:
            continue
        if name in ("kama", "frama", "ema", "sma", "dema", "tema", "vwap", "adl", "cmf"):
            # Price vs MA crossover
            if closes[i] > v and closes[i-1] <= vp:
                signals[i] = 1
            elif closes[i] < v and closes[i-1] >= vp:
                signals[i] = -1
        elif name in ("macd", "roc", "momentum"):
            if v > 0 and vp <= 0:
                signals[i] = 1
            elif v < 0 and vp >= 0:
                signals[i] = -1
        elif name == "rsi":
            if v < 30 and vp >= 30:
                signals[i] = 1   # oversold
            elif v > 70 and vp <= 70:
                signals[i] = -1  # overbought
        elif name in ("cci", "stochastic"):
            if v > 0 and vp <= 0:
                signals[i] = 1
            elif v < 0 and vp >= 0:
                signals[i] = -1
        elif name == "obv":
            if v > vp:
                signals[i] = 1
            elif v < vp:
                signals[i] = -1
        elif name == "mfi":
            if v < 20:
                signals[i] = 1
            elif v > 80:
                signals[i] = -1
    return signals

# ─── BACKTEST ONE INDICATOR ON ONE SYMBOL ────────────────────────────────────

def backtest_one(bars: List[dict], name: str) -> dict:
    if len(bars) < 60:
        return {"error": "insufficient data"}

    closes  = [b["close"] for b in bars]
    volumes = [b.get("volume", 0) for b in bars]
    highs   = [b.get("high", b["close"]) for b in bars]
    lows    = [b.get("low",  b["close"]) for b in bars]

    values  = compute_indicator(name, closes, volumes, highs, lows)
    signals = signals_from_values(values, name, closes)

    hold_period = 15
    trades = []
    i = 0
    while i < len(signals) - hold_period:
        if signals[i] == 1:
            pnl = (closes[i+hold_period] - closes[i]) / closes[i]
            trades.append(pnl)
            i += hold_period
        else:
            i += 1

    if not trades:
        return {"indicator": name, "trades": 0, "win_rate": 0,
                "sharpe": 0, "max_drawdown": 0, "ic": 0, "total_return": 0}

    wins     = [t for t in trades if t > 0]
    win_rate = len(wins) / len(trades)
    avg_r    = float(np.mean(trades))
    std_r    = float(np.std(trades)) if len(trades) > 1 else 1
    sharpe   = (avg_r / std_r) * np.sqrt(252/hold_period) if std_r > 0 else 0

    equity = [1.0]
    for t in trades:
        equity.append(equity[-1] * (1+t))
    peak   = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak: peak = e
        dd = (peak-e)/peak
        if dd > max_dd: max_dd = dd

    fwd, sig_vals = [], []
    for i in range(len(signals)-hold_period):
        if signals[i] != 0:
            fwd.append((closes[i+hold_period]-closes[i])/closes[i])
            sig_vals.append(signals[i])
    ic = 0.0
    if len(fwd) > 2:
        c = float(np.corrcoef(sig_vals, fwd)[0,1])
        ic = 0.0 if np.isnan(c) else c

    return {
        "indicator":    name,
        "trades":       len(trades),
        "win_rate":     round(win_rate, 4),
        "total_return": round(sum(trades), 4),
        "sharpe":       round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "ic":           round(ic, 4),
    }

# ─── MAIN TOOL FUNCTION ───────────────────────────────────────────────────────

def test_indicators_from_text(
    architect_text: str,
    symbols: List[str] = None,
    start_year: int = 2020,
    end_year: int = 2024
) -> str:
    """
    Main tool function called by debate graph.
    Extracts indicator names from architect free text,
    runs backtests, returns human-readable results string
    for injection into critic/architect context.
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS[:5]  # use 5 symbols for speed during debate

    # Extract indicators from text
    indicators = extract_indicators(architect_text)

    if not indicators:
        return "No known indicators found in the proposal. Architect should name specific indicators (e.g. KAMA, MACD, OBV, RSI, FRAMA, ROC, MFI)."

    # Run backtests
    aggregated: Dict[str, Dict] = {ind: {
        "win_rates": [], "sharpes": [], "ics": [],
        "drawdowns": [], "returns": [], "trades": 0
    } for ind in indicators}

    for symbol in symbols:
        bars = fetch_bars(symbol, start_year, end_year)
        if not bars:
            continue
        for ind in indicators:
            res = backtest_one(bars, ind)
            if "error" not in res and res["trades"] > 0:
                aggregated[ind]["win_rates"].append(res["win_rate"])
                aggregated[ind]["sharpes"].append(res["sharpe"])
                aggregated[ind]["ics"].append(res["ic"])
                aggregated[ind]["drawdowns"].append(res["max_drawdown"])
                aggregated[ind]["returns"].append(res["total_return"])
                aggregated[ind]["trades"] += res["trades"]

    # Build readable results
    lines = [
        f"BACKTEST RESULTS ({start_year}–{end_year}, {len(symbols)} symbols, 15-day hold)",
        "=" * 60
    ]

    scored = []
    for ind, data in aggregated.items():
        if not data["win_rates"]:
            lines.append(f"{ind.upper()}: no data")
            continue
        avg_wr  = float(np.mean(data["win_rates"]))
        avg_sh  = float(np.mean(data["sharpes"]))
        avg_ic  = float(np.mean(data["ics"]))
        avg_dd  = float(np.mean(data["drawdowns"]))
        avg_ret = float(np.mean(data["returns"]))
        composite = avg_ic*0.4 + avg_sh*0.1 + (avg_wr-0.5)*0.5
        group = KNOWN_INDICATORS.get(ind, "unknown")

        lines.append(
            f"{ind.upper()} ({group}): "
            f"IC={avg_ic:.3f} | Sharpe={avg_sh:.3f} | "
            f"WinRate={avg_wr*100:.1f}% | Drawdown={avg_dd*100:.1f}% | "
            f"Return={avg_ret*100:.1f}% | Trades={data['trades']} | "
            f"Score={composite:.3f}"
        )
        scored.append((ind, composite, group))

    # Winners per group
    lines.append("")
    lines.append("WINNERS BY GROUP:")
    groups_seen = {}
    for ind, score, group in sorted(scored, key=lambda x: -x[1]):
        if group not in groups_seen:
            groups_seen[group] = ind
            lines.append(f"  {group.upper()}: {ind.upper()} (score={score:.3f})")

    lines.append("")
    lines.append("Use these results to evaluate whether your proposed indicators are justified by data.")

    return "\n".join(lines)

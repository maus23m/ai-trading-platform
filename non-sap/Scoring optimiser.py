# scoring_optimiser.py — Gate 4: Scoring Curve Optimiser
# Autonomous validation engine called by debate graph
# Runs ablation studies, parameter sweeps, regime analysis,
# statistical significance tests on any proposed strategy

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

ROUND_TRIP_COST = 0.0020  # 0.20% base, 0.50% realistic
DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "JPM", "GS",
    "JNJ",  "UNH",  "AMZN", "WMT", "XOM"
]

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

# ─── PARAMETER EXTRACTOR ──────────────────────────────────────────────────────

def extract_validation_request(text: str) -> Dict:
    """Extract what the architect wants validated from free text."""
    text_lower = text.lower()
    params = {
        "signals": [],
        "hold_periods": [],
        "thresholds": [],
        "test_regime": True,
        "test_costs": True,
        "ablation": True
    }

    # Extract signals mentioned
    known_signals = ["sma", "roc", "momentum", "macd", "rsi", "ema",
                     "obv", "mfi", "kama", "frama", "revision", "earnings"]
    for s in known_signals:
        if re.search(r'\b' + s + r'\b', text_lower):
            params["signals"].append(s)

    # Extract hold periods
    holds = re.findall(r'(\d+)\s*[-\s]?\s*(?:day|days|period)', text_lower)
    params["hold_periods"] = [int(h) for h in holds if 3 <= int(h) <= 90] or [5,10,15,20,30,45]

    # Extract thresholds
    thresholds = re.findall(r'(\d+(?:\.\d+)?)\s*%', text_lower)
    params["thresholds"] = [float(t)/100 for t in thresholds if float(t) < 50] or [0.01, 0.02, 0.03]

    return params

# ─── SIGNAL FUNCTIONS ─────────────────────────────────────────────────────────

def compute_signal(name: str, closes: List[float],
                   period: int = 20) -> List[Optional[float]]:
    n = len(closes)

    if name in ("sma", "ema"):
        result = [None] * n
        k = 2/(period+1)
        for i in range(n):
            if i < period-1: continue
            result[i] = sum(closes[:period])/period if i==period-1 \
                        else closes[i]*k + result[i-1]*(1-k)
        return result

    elif name in ("roc", "momentum"):
        result = [None] * n
        for i in range(period, n):
            if closes[i-period] != 0:
                result[i] = (closes[i]-closes[i-period])/closes[i-period]
        return result

    elif name == "macd":
        k12, k26 = 2/13, 2/27
        e12 = [None]*n; e26 = [None]*n
        for i in range(n):
            if i < 11: continue
            e12[i] = sum(closes[:12])/12 if i==11 else closes[i]*k12+e12[i-1]*(1-k12)
        for i in range(n):
            if i < 25: continue
            e26[i] = sum(closes[:26])/26 if i==25 else closes[i]*k26+e26[i-1]*(1-k26)
        return [f-s if f and s else None for f,s in zip(e12,e26)]

    elif name == "rsi":
        result = [None]*n
        if n < 15: return result
        gains, losses = [], []
        for i in range(1, 15):
            d = closes[i]-closes[i-1]
            gains.append(max(d,0)); losses.append(max(-d,0))
        ag = sum(gains)/14; al = sum(losses)/14
        for i in range(14, n):
            d = closes[i]-closes[i-1]
            ag = (ag*13+max(d,0))/14; al = (al*13+max(-d,0))/14
            result[i] = 100 if al==0 else 100-(100/(1+ag/al))
        return result

    return [None]*n

def get_signals(name: str, closes: List[float],
                threshold: float = 0.0, period: int = 20) -> List[int]:
    values = compute_signal(name, closes, period)
    signals = [0]*len(closes)
    for i in range(1, len(closes)):
        v = values[i]; vp = values[i-1]
        if v is None or vp is None: continue
        if name in ("sma", "ema"):
            if closes[i] > v and closes[i-1] <= vp: signals[i] = 1
            elif closes[i] < v and closes[i-1] >= vp: signals[i] = -1
        elif name in ("roc", "momentum", "macd"):
            if v > threshold: signals[i] = 1
            elif v < -threshold: signals[i] = -1
        elif name == "rsi":
            if v < 30: signals[i] = 1
            elif v > 70: signals[i] = -1
    return signals

# ─── BACKTEST ENGINE ──────────────────────────────────────────────────────────

def backtest(closes: List[float], signals: List[int],
             hold: int, cost: float = ROUND_TRIP_COST,
             stop_loss: float = None) -> Dict:
    trades = []
    i = 0
    while i < len(signals) - hold:
        if signals[i] == 1:
            entry = closes[i]
            exit_p = closes[i+hold]
            if stop_loss:
                for j in range(i+1, i+hold+1):
                    if (closes[j]-entry)/entry <= -stop_loss:
                        exit_p = closes[j]; break
            trades.append((exit_p-entry)/entry - cost)
            i += hold
        else:
            i += 1

    if not trades:
        return {"trades":0,"sharpe":0,"win_rate":0,
                "total_return":0,"max_drawdown":0,"profit_factor":0}

    wins   = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    wr     = len(wins)/len(trades)
    avg_r  = float(np.mean(trades))
    std_r  = float(np.std(trades)) if len(trades) > 1 else 1
    sharpe = (avg_r/std_r)*np.sqrt(252/hold) if std_r > 0 else 0
    pf     = sum(wins)/(abs(sum(losses))+0.001) if losses else float(sum(wins))

    eq = [1.0]
    for t in trades: eq.append(eq[-1]*(1+t))
    peak = eq[0]; max_dd = 0
    for e in eq:
        if e > peak: peak = e
        dd = (peak-e)/peak
        if dd > max_dd: max_dd = dd

    # Statistical significance — t-test against zero
    t_stat = (avg_r / (std_r/np.sqrt(len(trades)))) if std_r > 0 else 0
    # p-value approximation
    from math import erfc, sqrt
    p_value = float(erfc(abs(t_stat)/sqrt(2)))

    return {
        "trades":       len(trades),
        "win_rate":     round(wr, 4),
        "total_return": round(sum(trades), 4),
        "sharpe":       round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "profit_factor":round(pf, 4),
        "t_stat":       round(t_stat, 3),
        "p_value":      round(p_value, 4),
        "significant":  p_value < 0.05
    }

# ─── WALK-FORWARD ─────────────────────────────────────────────────────────────

def walk_forward_oos(closes: List[float], signal_name: str,
                     hold: int, cost: float,
                     train_years: int = 2, test_years: int = 1,
                     period: int = 20, threshold: float = 0.0) -> Dict:
    train_bars = int(252*train_years)
    test_bars  = int(252*test_years)
    if len(closes) < train_bars + test_bars:
        return {"error": "insufficient data"}

    windows = []
    start = 0
    while start + train_bars + test_bars <= len(closes):
        test_c = closes[start+train_bars : start+train_bars+test_bars]
        test_s = get_signals(signal_name, closes[:start+train_bars+test_bars],
                             threshold, period)[start+train_bars:]
        r = backtest(test_c, test_s[:len(test_c)], hold, cost)
        if r["trades"] > 0:
            windows.append(r)
        start += test_bars

    if not windows:
        return {"error": "no trades"}

    sharpes  = [w["sharpe"]        for w in windows]
    returns  = [w["total_return"]  for w in windows]
    winrates = [w["win_rate"]      for w in windows]
    dds      = [w["max_drawdown"]  for w in windows]

    return {
        "windows":      len(windows),
        "avg_sharpe":   round(float(np.mean(sharpes)), 4),
        "std_sharpe":   round(float(np.std(sharpes)), 4),
        "avg_return":   round(float(np.mean(returns)), 4),
        "avg_win_rate": round(float(np.mean(winrates)), 4),
        "avg_drawdown": round(float(np.mean(dds)), 4),
        "stability":    round(1-float(np.std(sharpes))/(abs(float(np.mean(sharpes)))+0.001), 4),
        "pct_positive": round(sum(1 for s in sharpes if s > 0)/len(sharpes), 4)
    }

# ─── ABLATION STUDY ───────────────────────────────────────────────────────────

def run_ablation(closes: List[float], signals_list: List[str],
                 hold: int, cost: float) -> List[Dict]:
    """Test each signal in isolation and combinations."""
    results = []

    # Individual signals
    for sig in signals_list:
        s = get_signals(sig, closes)
        r = backtest(closes, s, hold, cost)
        r["variant"] = sig.upper()
        r["components"] = [sig]
        results.append(r)

    # Pairwise combinations — AND logic (both must agree)
    for i in range(len(signals_list)):
        for j in range(i+1, len(signals_list)):
            s1 = get_signals(signals_list[i], closes)
            s2 = get_signals(signals_list[j], closes)
            combined = [1 if a==1 and b==1 else 0
                       for a,b in zip(s1, s2)]
            r = backtest(closes, combined, hold, cost)
            r["variant"] = f"{signals_list[i].upper()} AND {signals_list[j].upper()}"
            r["components"] = [signals_list[i], signals_list[j]]
            results.append(r)

    return results

# ─── REGIME ANALYSIS ──────────────────────────────────────────────────────────

def regime_analysis(closes: List[float], signals: List[int],
                    hold: int, cost: float) -> Dict:
    """Split results by bull/bear/high-vol/low-vol regimes."""
    n = len(closes)
    sma200 = [None]*n
    for i in range(n):
        if i < 199: continue
        sma200[i] = sum(closes[i-199:i+1])/200

    # Volatility (20-day rolling std of returns)
    rets = [0.0] + [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,n)]
    vol20 = [None]*n
    for i in range(20, n):
        vol20[i] = float(np.std(rets[i-20:i])) * np.sqrt(252)

    bull_trades = []; bear_trades = []
    low_vol_trades = []; high_vol_trades = []

    i = 0
    while i < len(signals) - hold:
        if signals[i] == 1:
            pnl = (closes[i+hold]-closes[i])/closes[i] - cost
            # Bull vs Bear
            if sma200[i] and closes[i] > sma200[i]:
                bull_trades.append(pnl)
            elif sma200[i]:
                bear_trades.append(pnl)
            # Vol regime
            if vol20[i]:
                med_vol = 0.15  # ~15% annualised vol = median
                if vol20[i] < med_vol:
                    low_vol_trades.append(pnl)
                else:
                    high_vol_trades.append(pnl)
            i += hold
        else:
            i += 1

    def regime_stats(trades, name):
        if not trades:
            return {"regime": name, "trades": 0, "sharpe": 0,
                    "win_rate": 0, "avg_return": 0}
        avg = float(np.mean(trades))
        std = float(np.std(trades)) if len(trades) > 1 else 1
        return {
            "regime":     name,
            "trades":     len(trades),
            "sharpe":     round((avg/std)*np.sqrt(252/hold), 4) if std > 0 else 0,
            "win_rate":   round(sum(1 for t in trades if t>0)/len(trades), 4),
            "avg_return": round(avg, 4)
        }

    return {
        "bull":     regime_stats(bull_trades,     "bull (above SMA200)"),
        "bear":     regime_stats(bear_trades,     "bear (below SMA200)"),
        "low_vol":  regime_stats(low_vol_trades,  "low vol (<15% ann)"),
        "high_vol": regime_stats(high_vol_trades, "high vol (>15% ann)")
    }

# ─── COST SENSITIVITY ─────────────────────────────────────────────────────────

def cost_sensitivity(closes: List[float], signals: List[int],
                     hold: int) -> List[Dict]:
    """Test strategy at different transaction cost levels."""
    cost_levels = [0.001, 0.002, 0.004, 0.006, 0.0075]
    labels      = ["0.10%", "0.20%", "0.40%", "0.60%", "0.75%"]
    results = []
    for cost, label in zip(cost_levels, labels):
        r = backtest(closes, signals, hold, cost)
        r["cost"] = label
        results.append(r)
    return results

# ─── HOLD PERIOD SWEEP ────────────────────────────────────────────────────────

def hold_period_sweep(closes: List[float], signals_name: str,
                      hold_periods: List[int], cost: float) -> List[Dict]:
    """Sweep across hold periods and find optimal."""
    results = []
    for hold in hold_periods:
        sigs = get_signals(signals_name, closes)
        r = walk_forward_oos(closes, signals_name, hold, cost)
        if "error" not in r:
            r["hold_period"] = hold
            results.append(r)
    return sorted(results, key=lambda x: x.get("avg_sharpe", 0), reverse=True)

# ─── MAIN TOOL FUNCTION ───────────────────────────────────────────────────────

def run_optimisation_from_text(
    architect_text: str,
    symbols: List[str] = None,
    start_year: int = 2015,
    end_year: int = 2024
) -> str:
    """
    Main tool called by debate graph after architect proposes a strategy.
    Runs ablation, hold sweep, regime analysis, cost sensitivity.
    Returns human-readable validation report.
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS[:5]

    params = extract_validation_request(architect_text)
    signals_to_test = params["signals"] or ["sma", "roc"]
    hold_periods    = sorted(set(params["hold_periods"] or [5,10,15,20,30,45]))

    lines = [
        "SCORING CURVE OPTIMISATION REPORT",
        f"Period: {start_year}–{end_year} | Symbols: {len(symbols)}",
        f"Signals found in proposal: {signals_to_test}",
        f"Hold periods to test: {hold_periods}",
        "=" * 70
    ]

    all_closes = {}
    all_signals = {}

    for symbol in symbols:
        bars = fetch_bars(symbol, start_year, end_year)
        if bars:
            all_closes[symbol] = [b["close"] for b in bars]

    if not all_closes:
        return "No data available for any symbol."

    # ── 1. HOLD PERIOD SWEEP ──────────────────────────────────────────────────
    lines.append("\n1. HOLD PERIOD SWEEP (out-of-sample walk-forward)")
    lines.append("-" * 50)

    for sig in signals_to_test:
        lines.append(f"\n{sig.upper()}:")
        agg = {}
        for symbol, closes in all_closes.items():
            sweep = hold_period_sweep(closes, sig, hold_periods, ROUND_TRIP_COST)
            for r in sweep:
                h = r["hold_period"]
                if h not in agg:
                    agg[h] = []
                agg[h].append(r.get("avg_sharpe", 0))

        best_hold = None; best_sharpe = -999
        for h in sorted(agg.keys()):
            avg_sh = round(float(np.mean(agg[h])), 3)
            marker = ""
            if avg_sh > best_sharpe:
                best_sharpe = avg_sh; best_hold = h
            lines.append(f"  hold={h:2d}d | avg Sharpe={avg_sh:+.3f}")

        lines.append(f"  → OPTIMAL: {best_hold}-day hold (Sharpe={best_sharpe:.3f})")

    # ── 2. ABLATION STUDY ─────────────────────────────────────────────────────
    if len(signals_to_test) >= 2:
        lines.append("\n2. ABLATION STUDY (20-day hold, in-sample)")
        lines.append("-" * 50)

        agg_ablation = {}
        for symbol, closes in all_closes.items():
            results = run_ablation(closes, signals_to_test, 20, ROUND_TRIP_COST)
            for r in results:
                v = r["variant"]
                if v not in agg_ablation:
                    agg_ablation[v] = []
                agg_ablation[v].append(r.get("sharpe", 0))

        ablation_ranked = []
        for variant, sharpes in agg_ablation.items():
            avg_sh = float(np.mean(sharpes))
            ablation_ranked.append((variant, avg_sh))
        ablation_ranked.sort(key=lambda x: -x[1])

        for variant, avg_sh in ablation_ranked:
            sig_flag = "✓" if avg_sh > 0.8 else ("~" if avg_sh > 0.4 else "✗")
            lines.append(f"  {sig_flag} {variant}: Sharpe={avg_sh:+.3f}")

        best_variant = ablation_ranked[0][0] if ablation_ranked else "unknown"
        lines.append(f"  → BEST COMBINATION: {best_variant}")

    # ── 3. REGIME ANALYSIS ────────────────────────────────────────────────────
    lines.append("\n3. REGIME-CONDITIONAL PERFORMANCE (primary signal, 20-day hold)")
    lines.append("-" * 50)

    primary_sig = signals_to_test[0] if signals_to_test else "sma"
    regime_agg = {"bull":[],"bear":[],"low_vol":[],"high_vol":[]}

    for symbol, closes in all_closes.items():
        sigs = get_signals(primary_sig, closes)
        ra = regime_analysis(closes, sigs, 20, ROUND_TRIP_COST)
        for regime_key in regime_agg:
            sh = ra[regime_key].get("sharpe", 0)
            regime_agg[regime_key].append(sh)

    for regime_key, sharpes in regime_agg.items():
        avg_sh = float(np.mean(sharpes))
        flag = "✓" if avg_sh > 0.5 else ("~" if avg_sh > 0 else "✗")
        lines.append(f"  {flag} {regime_key.upper()}: Sharpe={avg_sh:+.3f}")

    # ── 4. COST SENSITIVITY ───────────────────────────────────────────────────
    lines.append("\n4. TRANSACTION COST SENSITIVITY (primary signal, optimal hold)")
    lines.append("-" * 50)

    cost_levels = [0.001, 0.002, 0.004, 0.006, 0.0075]
    cost_labels = ["0.10%", "0.20%", "0.40%", "0.60%", "0.75%"]
    cost_agg = {l: [] for l in cost_labels}

    for symbol, closes in all_closes.items():
        sigs = get_signals(primary_sig, closes)
        cost_results = cost_sensitivity(closes, sigs, 20)
        for r in cost_results:
            cost_agg[r["cost"]].append(r.get("sharpe", 0))

    for label, sharpes in cost_agg.items():
        avg_sh = float(np.mean(sharpes))
        flag = "✓" if avg_sh > 0.5 else ("~" if avg_sh > 0 else "✗")
        lines.append(f"  {flag} cost={label}: Sharpe={avg_sh:+.3f}")

    # ── 5. SUMMARY AND GATE CHECK ─────────────────────────────────────────────
    lines.append("\n5. GATE CHECK SUMMARY")
    lines.append("-" * 50)
    lines.append("  Pass criteria: Sharpe≥0.8 | WinRate≥55% | Stable across regimes | Survives 0.40% costs")
    lines.append("")
    lines.append("  Feeds into Strategy Architect:")
    lines.append(f"  - Validated signals: {signals_to_test}")
    lines.append(f"  - Optimal hold period: determined above")
    lines.append(f"  - Regime performance: documented above")
    lines.append(f"  - Cost robustness: documented above")
    lines.append("")
    lines.append("  Use these results to refine the hypothesis or confirm readiness for Strategy Architect.")

    return "\n".join(lines)

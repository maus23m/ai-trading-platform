# debate.py — Dual-agent debate tool v2
# Architect proposes → indicator tool runs automatically → Critic challenges on real data
# Fully autonomous — no human input needed between rounds

import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from indicator_tester import test_indicators_from_text
from backtester import run_walkforward_from_text

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

# ─── STATE ────────────────────────────────────────────────────────────────────

class DebateState(TypedDict):
    topic: str
    context: str
    max_rounds: int
    round: int
    history: List[dict]
    architect_position: str
    backtest_results: str       # indicator test results
    walkforward_results: str    # walk-forward backtest results
    critic_objections: str
    consensus_reached: bool
    final_verdict: str
    decision: str

# ─── NODE 1 — ARCHITECT ───────────────────────────────────────────────────────

def architect(state: DebateState) -> DebateState:
    is_first = state["round"] == 0

    if is_first:
        prompt = f"""You are the Architect of a systematic S&P 500 trading strategy.
Topic: {state['topic']}
Context: {state['context'] or 'S&P 500 systematic trading strategy.'}

You are a quantitative researcher with deep knowledge of mathematics, statistics,
financial theory, and market microstructure.

Do NOT start from indicators. Start from the market.

Step 1 — INEFFICIENCY: Identify a specific market inefficiency or risk premium that exists
in S&P 500 stocks. Explain precisely why it exists and why it has persisted despite being known.

Step 2 — MATHEMATICS: Derive the mathematical formula or transformation that best captures
this inefficiency from available market data (price, volume, fundamentals, macro).
Do not name a known indicator — derive the formula from the theory.

Step 3 — TESTABLE HYPOTHESIS: State the hypothesis in falsifiable form.
What exact signal, threshold, and holding period would confirm or deny the edge?
Name any mathematical functions precisely so they can be computed and tested.

Structure your response as:
INEFFICIENCY: [what market inefficiency you are exploiting and why it persists]
MATHEMATICS: [the exact formula or transformation derived from the theory]
HYPOTHESIS: [falsifiable statement — if X then Y, testable with real data]
EXPECTED EDGE: [why this should produce positive risk-adjusted returns]"""
    else:
        prompt = f"""You are the Architect of a systematic S&P 500 trading strategy.
Topic: {state['topic']}
Round: {state['round']} of {state['max_rounds']}

Your previous proposal was tested with real data. Here are the backtest results:

{state['backtest_results']}

The Critic challenged your proposal with:
{state['critic_objections']}

You are a quantitative researcher reasoning from evidence.

The backtest data has tested your hypothesis. Now reason from what the data shows.

If the data validates your hypothesis — explain the mechanism more precisely.
If the data rejects your hypothesis — do not simply swap to a different indicator.
Instead, ask: what does the failure tell us about the underlying market inefficiency?
Then derive a refined hypothesis from that understanding.

Always reason from:
  inefficiency → mathematics → testable hypothesis → evidence

Never reason from:
  "the data failed, let me try a different named indicator"

Structure your response as:
WHAT THE DATA SHOWS: [interpret the backtest results — what do they tell us about the inefficiency]
REFINED HYPOTHESIS: [updated mathematical formulation derived from evidence]
RESPONSES TO OBJECTIONS: [address each challenge using theory and data together]
POSITION: [MAINTAINED / REFINED — with the theoretical justification]"""

    response = llm.invoke(prompt)
    position = response.content

    new_history = state["history"] + [{
        "role": "Architect",
        "round": state["round"] + 1,
        "content": position
    }]

    return {**state, "architect_position": position, "history": new_history}

# ─── NODE 2 — RESEARCH TOOL (automatic) ──────────────────────────────────────

def run_indicator_tool(state: DebateState) -> DebateState:
    """
    Automatically called after every architect turn.
    Runs indicator comparison AND walk-forward backtest.
    Injects combined real data results for critic to use.
    """
    indicator_results = test_indicators_from_text(
        architect_text=state["architect_position"],
        symbols=["AAPL", "MSFT", "NVDA", "JPM", "GS"],
        start_year=2020,
        end_year=2024
    )

    walkforward_results = run_walkforward_from_text(
        architect_text=state["architect_position"],
        symbols=["AAPL", "MSFT", "NVDA", "JPM", "GS"],
        start_year=2015,
        end_year=2024,
        train_years=2,
        test_years=1
    )

    combined = (
        "=== INDICATOR COMPARISON (in-sample) ===\n" + indicator_results +
        "\n\n=== WALK-FORWARD BACKTEST (out-of-sample) ===\n" + walkforward_results
    )

    new_history = state["history"] + [{
        "role": "Research Tool",
        "round": state["round"] + 1,
        "content": combined
    }]

    return {
        **state,
        "backtest_results":    combined,
        "walkforward_results": walkforward_results,
        "history":             new_history
    }

# ─── NODE 3 — CRITIC ──────────────────────────────────────────────────────────

def critic(state: DebateState) -> DebateState:
    prompt = f"""You are the Critic reviewing a proposed trading strategy design decision.
Topic: {state['topic']}
Round: {state['round'] + 1} of {state['max_rounds']}

Architect's proposal:
{state['architect_position']}

ACTUAL BACKTEST RESULTS for the proposed indicators:
{state['backtest_results']}

You are a quantitative researcher and statistician with deep scepticism.
Your job is to stress-test the Architect's hypothesis at two levels:

Level 1 — THEORY: Is the proposed market inefficiency real and well-reasoned?
- Does the inefficiency have a credible behavioural or structural explanation?
- Is it likely to persist or has it been arbitraged away?
- Is the mathematical formulation a clean measure of the claimed inefficiency?
- Or is it just a transformation of price that happens to fit recent data?

Level 2 — EVIDENCE: Does the backtest data support the hypothesis?
- Is the information coefficient meaningful or noise?
- Is the Sharpe robust across different time windows or concentrated in one period?
- Are there enough trades to be statistically significant?
- Does the out-of-sample walk-forward confirm or contradict the in-sample results?
- Are there signs of overfitting — too many parameters, too few trades, suspiciously clean results?

Be intellectually honest. If the theory is sound and the data confirms it, concede.
If either the theory or the data is weak, challenge it precisely.

Structure your response as:
THEORY CRITIQUE: [is the inefficiency real and the mathematics sound?]
DATA ANALYSIS: [what the backtest numbers actually show]
OBJECTIONS: [numbered — only raise objections grounded in theory or data]
CONCESSIONS: [where the Architect is right]
VERDICT: [CHALLENGED / PARTIALLY SATISFIED / SATISFIED]"""

    response = llm.invoke(prompt)
    objections = response.content

    new_history = state["history"] + [{
        "role": "Critic",
        "round": state["round"] + 1,
        "content": objections
    }]

    return {
        **state,
        "critic_objections": objections,
        "history": new_history,
        "round": state["round"] + 1
    }

# ─── NODE 4 — JUDGE ───────────────────────────────────────────────────────────

def judge(state: DebateState) -> DebateState:
    prompt = f"""You are an independent Judge evaluating a strategy design debate.
Topic: {state['topic']}
Rounds completed: {state['round']} of {state['max_rounds']}

Full debate transcript including backtest results:
{_format_history(state['history'])}

You are an independent Judge — part quantitative researcher, part risk manager.

Evaluate the debate at three levels:

1. THEORY: Was the proposed market inefficiency credible and well-reasoned?
   Is it a genuine edge or a data mining artefact?

2. EVIDENCE: Does the backtest data — both in-sample and out-of-sample walk-forward —
   confirm the hypothesis? Is the edge robust across time periods and market regimes?

3. COMPLETENESS: Does this output feed into a complete trading strategy?
   What has been validated? What remains to be tested before the Strategy Architect
   can assemble a deployable strategy?

Your verdict feeds directly into the Strategy Architect — be precise about
what is proven, what is promising but unproven, and what should be discarded.

Structure your response as:
CONSENSUS: [YES / NO / PARTIAL]
VALIDATED EDGE: [what market inefficiency is confirmed by theory + data]
MATHEMATICAL FORMULATION: [the exact formula to implement — derived from the debate]
REJECTED HYPOTHESES: [what failed and the theoretical reason why]
FEEDS INTO STRATEGY ARCHITECT: [what this output contributes to the full strategy]
REMAINING GAPS: [what the Strategy Architect still needs before deploying]
CONFIDENCE: [HIGH / MEDIUM / LOW]"""

    response = llm.invoke(prompt)
    verdict = response.content

    consensus = False
    if "CONSENSUS:" in verdict.upper():
        after = verdict.upper().split("CONSENSUS:")[1][:30]
        consensus = "YES" in after

    decision = ""
    if "FINAL DECISION:" in verdict.upper():
        idx = verdict.upper().find("FINAL DECISION:")
        rest = verdict[idx + len("FINAL DECISION:"):].strip()
        decision = rest.split("\n")[0].strip()

    return {
        **state,
        "final_verdict": verdict,
        "consensus_reached": consensus,
        "decision": decision
    }

# ─── ROUTING ──────────────────────────────────────────────────────────────────

def should_continue(state: DebateState) -> str:
    if state["round"] >= state["max_rounds"]:
        return "judge"
    return "architect"

# ─── UTILS ───────────────────────────────────────────────────────────────────

def _format_history(history: List[dict]) -> str:
    if not history:
        return "No history yet."
    lines = []
    for entry in history:
        lines.append(f"[Round {entry['round']} — {entry['role']}]")
        lines.append(entry["content"])
        lines.append("")
    return "\n".join(lines)

# ─── BUILD GRAPH ──────────────────────────────────────────────────────────────

def build_debate_graph():
    graph = StateGraph(DebateState)

    graph.add_node("architect",           architect)
    graph.add_node("indicator_tool",      run_indicator_tool)
    graph.add_node("critic",              critic)
    graph.add_node("judge",               judge)

    graph.set_entry_point("architect")
    graph.add_edge("architect",      "indicator_tool")   # tool always runs after architect
    graph.add_edge("indicator_tool", "critic")           # critic always sees real data
    graph.add_conditional_edges("critic", should_continue)
    graph.add_edge("judge",          END)

    return graph.compile()

# ─── RUN FUNCTION ─────────────────────────────────────────────────────────────

def run_debate(topic: str, context: str = "", max_rounds: int = 3) -> dict:
    app = build_debate_graph()

    initial_state: DebateState = {
        "topic":              topic,
        "context":            context,
        "max_rounds":         max_rounds,
        "round":              0,
        "history":            [],
        "architect_position": "",
        "backtest_results":   "",
        "critic_objections":  "",
        "consensus_reached":  False,
        "final_verdict":      "",
        "decision":           ""
    }

    final_state = app.invoke(
        initial_state,
        config={"recursion_limit": 60}
    )

    return {
        "topic":             final_state["topic"],
        "rounds_completed":  final_state["round"],
        "consensus_reached": final_state["consensus_reached"],
        "final_verdict":     final_state["final_verdict"],
        "decision":          final_state["decision"],
        "history":           final_state["history"]
    }

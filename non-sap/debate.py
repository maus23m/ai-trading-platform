# debate.py — Dual-agent debate tool v2
# Architect proposes → indicator tool runs automatically → Critic challenges on real data
# Fully autonomous — no human input needed between rounds

import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from indicator_tester import test_indicators_from_text

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
    backtest_results: str       # injected automatically after each architect turn
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

Propose a specific design decision. Name the exact indicators, parameters, or rules you recommend.
Propose from first principles. Name indicators by their standard industry names so they can be identified and tested precisely.

Structure your response as:
PROPOSAL: [your specific proposal in 1-2 sentences naming exact indicators/rules]
REASONING: [3-5 bullet points justifying each choice]
EXPECTED OUTCOME: [what this achieves for the strategy]"""
    else:
        prompt = f"""You are the Architect of a systematic S&P 500 trading strategy.
Topic: {state['topic']}
Round: {state['round']} of {state['max_rounds']}

Your previous proposal was tested with real data. Here are the backtest results:

{state['backtest_results']}

The Critic challenged your proposal with:
{state['critic_objections']}

Based on the actual backtest data, respond to the Critic.
If the data supports your proposal, defend it with the numbers.
If the data contradicts your proposal, refine it — name specific alternative indicators to test.
Name any indicators by their standard industry names so they can be identified and tested precisely.

Structure your response as:
REFINED PROPOSAL: [updated or defended proposal with specific indicator names]
DATA RESPONSE: [how the backtest results support or change your position]
RESPONSES TO OBJECTIONS: [address each objection using the data]
POSITION: [MAINTAINED / REFINED — and why]"""

    response = llm.invoke(prompt)
    position = response.content

    new_history = state["history"] + [{
        "role": "Architect",
        "round": state["round"] + 1,
        "content": position
    }]

    return {**state, "architect_position": position, "history": new_history}

# ─── NODE 2 — INDICATOR TOOL (automatic) ─────────────────────────────────────

def run_indicator_tool(state: DebateState) -> DebateState:
    """
    Automatically called after every architect turn.
    Extracts indicator names from architect's proposal,
    runs backtests, injects results into state.
    """
    results = test_indicators_from_text(
        architect_text=state["architect_position"],
        symbols=["AAPL", "MSFT", "NVDA", "JPM", "GS"],  # 5 symbols for speed
        start_year=2020,
        end_year=2024
    )

    new_history = state["history"] + [{
        "role": "Indicator Tool",
        "round": state["round"] + 1,
        "content": results
    }]

    return {**state, "backtest_results": results, "history": new_history}

# ─── NODE 3 — CRITIC ──────────────────────────────────────────────────────────

def critic(state: DebateState) -> DebateState:
    prompt = f"""You are the Critic reviewing a proposed trading strategy design decision.
Topic: {state['topic']}
Round: {state['round'] + 1} of {state['max_rounds']}

Architect's proposal:
{state['architect_position']}

ACTUAL BACKTEST RESULTS for the proposed indicators:
{state['backtest_results']}

Use the backtest data to challenge or validate the Architect's proposal.
Look for:
- Indicators with low IC (< 0.05) — weak predictive power
- Indicators with negative Sharpe — destroys value
- Win rates below 50% — coin-flip or worse
- High drawdowns — unacceptable risk
- Better alternatives in the same group that scored higher
- Overfitting concerns — too few trades, suspiciously good numbers

If the data strongly supports the proposal, concede on those points.
Be intellectually honest — data beats opinion.

Structure your response as:
DATA ANALYSIS: [what the backtest numbers actually show]
OBJECTIONS: [numbered list — only raise objections supported by the data]
CONCESSIONS: [points where the data validates the Architect]
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

Based on the full debate and the actual backtest data:
1. Which indicators are justified by the data?
2. Which should be rejected and why?
3. What is the final recommended indicator stack?

Structure your response as:
CONSENSUS: [YES / NO / PARTIAL]
FINAL DECISION: [exact indicator stack to implement — name each indicator]
DATA JUSTIFICATION: [which backtest metrics support this decision]
REJECTED INDICATORS: [what was tested and failed and why]
REMAINING RISKS: [what needs further validation]
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

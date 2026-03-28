# strategy_architect.py — Stage 2: Strategy Architect Agent
# Reads ALL accumulated debate results from Supabase
# Designs a complete, deployable trading strategy from validated evidence
# Owns sole authority to declare a strategy ready for deployment
# No other agent can do this

import os
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from supabase import create_client

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)

# ─── STATE ────────────────────────────────────────────────────────────────────

class ArchitectState(TypedDict):
    debate_results: List[Dict]      # all rows from debate_results table
    backtest_results: List[Dict]    # all rows from backtest_runs table
    analysis: str                   # architect's analysis of accumulated evidence
    strategy_spec: str              # complete strategy specification
    deployment_decision: str        # READY / NOT_READY / NEEDS_MORE_RESEARCH
    deployment_reason: str          # why this decision was made
    gaps: List[str]                 # what still needs to be validated
    next_debate_topics: List[str]   # what the debate tool should tackle next

# ─── NODE 1 — LOAD EVIDENCE ───────────────────────────────────────────────────

def load_evidence(state: ArchitectState) -> ArchitectState:
    """Load all accumulated evidence from Supabase."""

    # Load all debate results
    try:
        debate_rows = supabase.table("debate_results").select("*").order(
            "created_at", desc=False
        ).execute()
        debate_results = debate_rows.data or []
    except Exception as e:
        debate_results = []

    # Load all backtest runs
    try:
        backtest_rows = supabase.table("backtest_runs").select("*").order(
            "id", desc=False
        ).execute()
        backtest_results = backtest_rows.data or []
    except Exception as e:
        backtest_results = []

    return {
        **state,
        "debate_results":  debate_results,
        "backtest_results": backtest_results
    }

# ─── NODE 2 — ANALYSE EVIDENCE ────────────────────────────────────────────────

def analyse_evidence(state: ArchitectState) -> ArchitectState:
    """
    Architect reads all accumulated evidence and produces a structured analysis.
    Not constrained by any single debate — reasons across all of them.
    """
    debate_results  = state["debate_results"]
    backtest_results = state["backtest_results"]

    # Format debate evidence
    debate_summary = ""
    for i, row in enumerate(debate_results):
        debate_summary += f"""
DEBATE {i+1} — Topic: {row.get('topic', 'unknown')}
  Consensus: {row.get('consensus_reached')}
  Confidence: {row.get('confidence', '')}
  Validated edge: {row.get('validated_edge', '')[:500]}
  Mathematical formulation: {row.get('mathematical_formulation', '')[:500]}
  Feeds into architect: {row.get('feeds_into_architect', '')[:500]}
  Remaining gaps: {row.get('remaining_gaps', '')[:300]}
"""

    # Format backtest evidence
    backtest_summary = ""
    for row in backtest_results[-10:]:  # last 10 runs
        p = row.get("parameters", {})
        backtest_summary += (
            f"  symbol={p.get('symbol','?')} "
            f"sma={p.get('sma_short','?')}/{p.get('sma_long','?')} "
            f"return={row.get('total_return',0):.3f} "
            f"win_rate={row.get('win_rate',0):.3f} "
            f"drawdown={row.get('max_drawdown',0):.3f} "
            f"passed={row.get('constraints_met')}\n"
        )

    prompt = f"""You are the Strategy Architect — the highest authority in this trading system.

Your role is to read ALL accumulated research evidence and determine:
1. What market inefficiencies have been confirmed across multiple debates?
2. What mathematical formulations are validated and ready to implement?
3. What is contradicted or unproven and should be discarded?
4. What critical gaps remain before a complete strategy can be deployed?

You have access to {len(debate_results)} debate(s) and {len(backtest_results)} backtest run(s).

ACCUMULATED DEBATE EVIDENCE:
{debate_summary if debate_summary else "No debate results yet."}

ACCUMULATED BACKTEST EVIDENCE:
{backtest_summary if backtest_summary else "No backtest runs yet."}

Analyse this evidence as a senior quantitative researcher would.
Look for:
- Consistent findings across multiple debates (high confidence)
- Contradictions between debates (needs resolution)
- Gaps in the three-layer framework (macro/fundamental/technical)
- What is validated vs what is still theoretical

Structure your analysis as:
CONFIRMED EDGES: [what is proven across multiple debates and backtests]
CONFIRMED REJECTIONS: [what has been empirically rejected]
FRAMEWORK COVERAGE: [which of the three layers have been addressed]
CRITICAL GAPS: [what is missing before a complete strategy exists]
SYNTHESIS: [how the confirmed edges fit together into a coherent whole]"""

    response = llm.invoke(prompt)
    return {**state, "analysis": response.content}

# ─── NODE 3 — DESIGN STRATEGY ─────────────────────────────────────────────────

def design_strategy(state: ArchitectState) -> ArchitectState:
    """
    Based on the analysis, design the most complete strategy possible
    from currently validated evidence. Be explicit about what is
    proven vs assumed.
    """
    prompt = f"""You are the Strategy Architect designing a complete trading strategy.

Based on this analysis of accumulated research:
{state['analysis']}

Design the most complete, deployable trading strategy that can be justified
by the current evidence. Be explicit about:
- What is empirically proven (high confidence)
- What is theoretically sound but needs more validation (medium confidence)
- What parameters are validated vs assumed

Do NOT invent edges that haven't been tested. Do NOT claim higher confidence
than the evidence supports. A partial but honest strategy is better than
a complete but fabricated one.

The strategy must cover:
1. SIGNAL GENERATION: exact mathematical formula for entry signals
2. POSITION SIZING: how much to allocate per signal
3. RISK MANAGEMENT: stop losses, drawdown limits, position limits
4. EXIT RULES: when to close positions
5. UNIVERSE: which stocks to trade and any filters
6. REGIME FILTER: when to reduce or stop trading

Structure as:
STRATEGY NAME: [descriptive name]
CORE HYPOTHESIS: [the market inefficiency being exploited]
SIGNAL: [exact formula — only what is validated]
POSITION SIZING: [rules — mark as VALIDATED or ASSUMED]
EXIT RULES: [rules — mark as VALIDATED or ASSUMED]
UNIVERSE: [stock selection criteria]
REGIME FILTER: [when to trade vs not trade]
EXPECTED PERFORMANCE: [based on backtest evidence only]
CONFIDENCE: [HIGH/MEDIUM/LOW per component]"""

    response = llm.invoke(prompt)
    return {**state, "strategy_spec": response.content}

# ─── NODE 4 — DEPLOYMENT DECISION ────────────────────────────────────────────

def deployment_decision(state: ArchitectState) -> ArchitectState:
    """
    Make the final deployment decision.
    Only this agent can declare a strategy ready.
    Be conservative — paper trading Sharpe must be validated first.
    """
    prompt = f"""You are the Strategy Architect making the final deployment decision.

Strategy specification:
{state['strategy_spec']}

Analysis of evidence:
{state['analysis']}

Number of debates completed: {len(state['debate_results'])}
Number of backtest runs: {len(state['backtest_results'])}

Make a deployment decision. The bar is HIGH:
- READY: Strategy is complete, all components empirically validated,
  paper trading results confirm live readiness. Only declare READY
  if every component is validated and risks are fully understood.

- NOT_READY: Strategy has validated core signals but missing components
  (position sizing, regime filter, universe definition, etc.).
  Specify exactly what is needed before READY.

- NEEDS_MORE_RESEARCH: Core signal is unproven or contradicted.
  Specify what debates or backtests are needed next.

Also provide:
- The 3 highest priority topics for the next debate rounds
- What specific validation is needed before the next review

Structure as:
DECISION: [READY / NOT_READY / NEEDS_MORE_RESEARCH]
REASON: [why this decision — be specific]
WHAT IS PROVEN: [bullet list]
WHAT IS MISSING: [bullet list — specific gaps]
NEXT DEBATE TOPICS: [3 specific questions for the debate tool]
NEXT REVIEW CRITERIA: [what needs to be true to upgrade the decision]"""

    response = llm.invoke(prompt)
    verdict = response.content

    # Extract decision
    decision = "NEEDS_MORE_RESEARCH"
    if "DECISION:" in verdict.upper():
        after = verdict.upper().split("DECISION:")[1][:50]
        if "READY" in after and "NOT_READY" not in after:
            decision = "READY"
        elif "NOT_READY" in after:
            decision = "NOT_READY"

    # Extract next debate topics
    topics = []
    if "NEXT DEBATE TOPICS:" in verdict.upper():
        idx = verdict.upper().find("NEXT DEBATE TOPICS:")
        rest = verdict[idx:].split("\n")
        for line in rest[1:6]:
            line = line.strip().lstrip("•-123456789. ")
            if line and len(line) > 10:
                topics.append(line)

    # Extract gaps
    gaps = []
    if "WHAT IS MISSING:" in verdict.upper():
        idx = verdict.upper().find("WHAT IS MISSING:")
        rest = verdict[idx:].split("\n")
        for line in rest[1:8]:
            line = line.strip().lstrip("•-123456789. ")
            if line and len(line) > 5:
                gaps.append(line)

    return {
        **state,
        "deployment_decision": decision,
        "deployment_reason":   verdict,
        "next_debate_topics":  topics[:3],
        "gaps":                gaps
    }

# ─── BUILD GRAPH ──────────────────────────────────────────────────────────────

def build_architect_graph():
    graph = StateGraph(ArchitectState)

    graph.add_node("load_evidence",      load_evidence)
    graph.add_node("analyse_evidence",   analyse_evidence)
    graph.add_node("design_strategy",    design_strategy)
    graph.add_node("deployment_decision", deployment_decision)

    graph.set_entry_point("load_evidence")
    graph.add_edge("load_evidence",      "analyse_evidence")
    graph.add_edge("analyse_evidence",   "design_strategy")
    graph.add_edge("design_strategy",    "deployment_decision")
    graph.add_edge("deployment_decision", END)

    return graph.compile()

# ─── RUN FUNCTION ─────────────────────────────────────────────────────────────

def run_strategy_architect() -> dict:
    """
    Run the Strategy Architect.
    Reads all accumulated evidence from Supabase and produces:
    - Complete strategy specification
    - Deployment decision
    - Next steps for the debate tool
    """
    app = build_architect_graph()

    initial_state: ArchitectState = {
        "debate_results":      [],
        "backtest_results":    [],
        "analysis":            "",
        "strategy_spec":       "",
        "deployment_decision": "",
        "deployment_reason":   "",
        "gaps":                [],
        "next_debate_topics":  []
    }

    final_state = app.invoke(
        initial_state,
        config={"recursion_limit": 25}
    )

    return {
        "deployment_decision":  final_state["deployment_decision"],
        "strategy_spec":        final_state["strategy_spec"],
        "analysis":             final_state["analysis"],
        "deployment_reason":    final_state["deployment_reason"],
        "gaps":                 final_state["gaps"],
        "next_debate_topics":   final_state["next_debate_topics"],
        "debates_read":         len(final_state["debate_results"]),
        "backtests_read":       len(final_state["backtest_results"])
    }

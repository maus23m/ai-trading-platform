# strategy_architect.py — Stage 2: Strategy Architect Agent
# Autonomous bounded loop — reads evidence, fires debates, re-reads, repeats
# Terminates when: READY | max_research_rounds reached | no progress | no new gaps
# RULE: Only this agent can declare a strategy deployable

import os
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from supabase import create_client
from debate import run_debate

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)

# ─── BOUNDS ───────────────────────────────────────────────────────────────────

MAX_RESEARCH_ROUNDS  = 5   # max autonomous debate rounds
MAX_DEBATES_PER_ROUND = 3  # max debates fired per round
DEBATE_ROUNDS = 2          # rounds per debate (keep short for speed)

# ─── STATE ────────────────────────────────────────────────────────────────────

class ArchitectState(TypedDict):
    # Evidence
    debate_results: List[Dict]
    backtest_results: List[Dict]
    # Analysis
    analysis: str
    strategy_spec: str
    deployment_decision: str
    deployment_reason: str
    gaps: List[str]
    next_debate_topics: List[str]
    # Loop control
    research_round: int
    max_research_rounds: int
    previous_gaps: List[str]     # to detect no-progress
    debates_fired: List[str]     # topics already debated this run
    termination_reason: str
    # Log
    research_log: List[str]

# ─── NODE 1 — LOAD EVIDENCE ───────────────────────────────────────────────────

def load_evidence(state: ArchitectState) -> ArchitectState:
    try:
        debate_rows = supabase.table("debate_results").select("*").order(
            "created_at", desc=False
        ).execute()
        debate_results = debate_rows.data or []
    except Exception:
        debate_results = []

    try:
        backtest_rows = supabase.table("backtest_runs").select("*").order(
            "id", desc=False
        ).execute()
        backtest_results = backtest_rows.data or []
    except Exception:
        backtest_results = []

    log = state.get("research_log", [])
    log.append(
        f"Round {state['research_round']+1}: loaded {len(debate_results)} debates, "
        f"{len(backtest_results)} backtests"
    )

    return {
        **state,
        "debate_results":  debate_results,
        "backtest_results": backtest_results,
        "research_log": log
    }

# ─── NODE 2 — ANALYSE EVIDENCE ────────────────────────────────────────────────

def analyse_evidence(state: ArchitectState) -> ArchitectState:
    debate_summary = ""
    for i, row in enumerate(state["debate_results"]):
        debate_summary += f"""
DEBATE {i+1} — Topic: {row.get('topic', 'unknown')}
  Consensus: {row.get('consensus_reached')}
  Confidence: {row.get('confidence', '')}
  Validated edge: {row.get('validated_edge', '')[:400]}
  Mathematical formulation: {row.get('mathematical_formulation', '')[:400]}
  Feeds into architect: {row.get('feeds_into_architect', '')[:400]}
  Remaining gaps: {row.get('remaining_gaps', '')[:300]}
"""

    backtest_summary = ""
    for row in state["backtest_results"][-10:]:
        p = row.get("parameters", {})
        backtest_summary += (
            f"  symbol={p.get('symbol','?')} "
            f"return={row.get('total_return',0):.3f} "
            f"win_rate={row.get('win_rate',0):.3f} "
            f"drawdown={row.get('max_drawdown',0):.3f} "
            f"passed={row.get('constraints_met')}\n"
        )

    prompt = f"""You are the Strategy Architect — the highest authority in this trading system.
Research round: {state['research_round']+1} of {state['max_research_rounds']}
Debates completed: {len(state['debate_results'])}
Backtests completed: {len(state['backtest_results'])}

ACCUMULATED DEBATE EVIDENCE:
{debate_summary or "No debate results yet."}

ACCUMULATED BACKTEST EVIDENCE:
{backtest_summary or "No backtest runs yet."}

Analyse this evidence as a senior quantitative researcher.
Look for:
- Confirmed edges across multiple debates (high confidence)
- Contradictions between debates (needs resolution)
- Which of the three layers (macro/fundamental/technical) have been addressed
- What is validated vs still theoretical
- Whether gaps from the previous round have been closed

Structure your analysis as:
CONFIRMED EDGES: [what is proven]
CONFIRMED REJECTIONS: [what has been empirically rejected]
FRAMEWORK COVERAGE: [macro/fundamental/technical — what is done vs missing]
CLOSED GAPS: [gaps from previous round that are now resolved]
REMAINING GAPS: [what is still missing — be specific and numbered]
SYNTHESIS: [how confirmed edges fit together]"""

    response = llm.invoke(prompt)
    return {**state, "analysis": response.content}

# ─── NODE 3 — DESIGN STRATEGY ─────────────────────────────────────────────────

def design_strategy(state: ArchitectState) -> ArchitectState:
    prompt = f"""You are the Strategy Architect designing a complete trading strategy.
Research round: {state['research_round']+1} of {state['max_research_rounds']}

Based on this analysis of accumulated research:
{state['analysis']}

Design the most complete, deployable trading strategy justified by current evidence.
Mark each component as VALIDATED, ASSUMED, or REJECTED.

Structure as:
STRATEGY NAME: [descriptive name]
CORE HYPOTHESIS: [market inefficiency being exploited]
SIGNAL: [exact formula — only what is validated]
POSITION SIZING: [rules — mark VALIDATED or ASSUMED]
EXIT RULES: [rules — mark VALIDATED or ASSUMED]
UNIVERSE: [stock selection criteria]
REGIME FILTER: [when to trade vs not — mark VALIDATED or ASSUMED]
EXPECTED PERFORMANCE: [based on backtest evidence only]
CONFIDENCE PER COMPONENT: [HIGH/MEDIUM/LOW per component]"""

    response = llm.invoke(prompt)
    return {**state, "strategy_spec": response.content}

# ─── NODE 4 — DEPLOYMENT DECISION ────────────────────────────────────────────

def make_deployment_decision(state: ArchitectState) -> ArchitectState:
    prompt = f"""You are the Strategy Architect making the final deployment decision.
Research round: {state['research_round']+1} of {state['max_research_rounds']}
Debates completed: {len(state['debate_results'])}

Strategy specification:
{state['strategy_spec']}

Analysis:
{state['analysis']}

Make a deployment decision:
- READY: All components empirically validated. Deploy to paper trading.
- NOT_READY: Core signal proven but gaps remain. Specify exactly what is missing.
- NEEDS_MORE_RESEARCH: Core signal unproven. Specify what debates are needed.

Also provide up to {MAX_DEBATES_PER_ROUND} specific debate topics to close the most critical gaps.
Each topic must be a clear open question that the debate tool can answer.
Do NOT name specific indicators or answers in the topics — let the agents discover.

Structure as:
DECISION: [READY / NOT_READY / NEEDS_MORE_RESEARCH]
REASON: [specific justification]
WHAT IS PROVEN: [bullet list]
WHAT IS MISSING: [bullet list — numbered, specific]
NEXT DEBATE TOPICS: [up to {MAX_DEBATES_PER_ROUND} open questions — one per line, no numbering prefix]
NEXT REVIEW CRITERIA: [what must be true to upgrade decision]"""

    response = llm.invoke(prompt)
    verdict = response.content

    # Extract decision
    decision = "NEEDS_MORE_RESEARCH"
    if "DECISION:" in verdict.upper():
        after = verdict.upper().split("DECISION:")[1][:50]
        if "NOT_READY" in after:
            decision = "NOT_READY"
        elif "READY" in after:
            decision = "READY"

    # Extract next debate topics
    topics = []
    if "NEXT DEBATE TOPICS:" in verdict.upper():
        idx = verdict.upper().find("NEXT DEBATE TOPICS:")
        rest = verdict[idx:].split("\n")
        for line in rest[1:MAX_DEBATES_PER_ROUND+4]:
            line = line.strip().lstrip("•-*123456789. ")
            if line and len(line) > 15 and "?" in line:
                # Skip if already debated this run
                if not any(line[:30] in d for d in state.get("debates_fired", [])):
                    topics.append(line)
            if len(topics) >= MAX_DEBATES_PER_ROUND:
                break

    # Extract gaps
    gaps = []
    if "WHAT IS MISSING:" in verdict.upper():
        idx = verdict.upper().find("WHAT IS MISSING:")
        rest = verdict[idx:].split("\n")
        for line in rest[1:8]:
            line = line.strip().lstrip("•-*123456789. ")
            if line and len(line) > 5:
                gaps.append(line)

    return {
        **state,
        "deployment_decision": decision,
        "deployment_reason":   verdict,
        "next_debate_topics":  topics,
        "gaps":                gaps
    }

# ─── NODE 5 — FIRE DEBATES ────────────────────────────────────────────────────

def fire_debates(state: ArchitectState) -> ArchitectState:
    """
    Autonomously fire debates on identified gaps.
    Bounded by MAX_DEBATES_PER_ROUND.
    Saves results to Supabase so next load_evidence picks them up.
    """
    log = state.get("research_log", [])
    topics = state.get("next_debate_topics", [])[:MAX_DEBATES_PER_ROUND]
    debates_fired = state.get("debates_fired", [])

    if not topics:
        log.append(f"Round {state['research_round']+1}: no new topics to debate")
        return {**state, "research_log": log}

    log.append(f"Round {state['research_round']+1}: firing {len(topics)} debate(s) autonomously")

    for topic in topics:
        if topic in debates_fired:
            log.append(f"  skipped (already debated): {topic[:60]}")
            continue

        log.append(f"  debating: {topic[:80]}...")

        try:
            result = run_debate(
                topic=topic,
                context="S&P 500 systematic trading strategy research.",
                max_rounds=DEBATE_ROUNDS
            )

            # Save to Supabase
            verdict = result.get("final_verdict", "")

            def extract_section(text, label):
                upper = text.upper()
                if label.upper() + ":" not in upper:
                    return ""
                idx = upper.find(label.upper() + ":")
                rest = text[idx + len(label) + 1:].strip()
                lines = rest.split("\n")
                section = []
                for line in lines:
                    if any(line.upper().startswith(h) for h in [
                        "CONSENSUS:", "VALIDATED EDGE:", "MATHEMATICAL FORMULATION:",
                        "REJECTED HYPOTHESES:", "FEEDS INTO STRATEGY ARCHITECT:",
                        "REMAINING GAPS:", "CONFIDENCE:", "FINAL DECISION:"
                    ]):
                        break
                    section.append(line)
                return "\n".join(section).strip()[:2000]

            supabase.table("debate_results").insert({
                "topic":                    topic,
                "rounds_completed":         result.get("rounds_completed", 0),
                "consensus_reached":        result.get("consensus_reached", False),
                "validated_edge":           extract_section(verdict, "VALIDATED EDGE"),
                "mathematical_formulation": extract_section(verdict, "MATHEMATICAL FORMULATION"),
                "feeds_into_architect":     extract_section(verdict, "FEEDS INTO STRATEGY ARCHITECT"),
                "remaining_gaps":           extract_section(verdict, "REMAINING GAPS"),
                "confidence":               extract_section(verdict, "CONFIDENCE"),
                "final_verdict":            verdict[:5000],
                "decision":                 result.get("decision", "")
            }).execute()

            debates_fired.append(topic)
            log.append(f"  ✓ saved: {topic[:60]}")

        except Exception as e:
            log.append(f"  ✗ failed: {topic[:60]} — {str(e)[:80]}")

    return {
        **state,
        "debates_fired": debates_fired,
        "research_log":  log
    }

# ─── ROUTING ──────────────────────────────────────────────────────────────────

def should_continue(state: ArchitectState) -> str:
    """
    Decide whether to loop back for another research round or terminate.
    Bounded by MAX_RESEARCH_ROUNDS and progress detection.
    """
    decision = state.get("deployment_decision", "")
    round_num = state.get("research_round", 0)
    current_gaps = set(state.get("gaps", []))
    previous_gaps = set(state.get("previous_gaps", []))

    # Termination condition 1 — success
    if decision == "READY":
        return "end"

    # Termination condition 2 — safety limit
    if round_num >= state.get("max_research_rounds", MAX_RESEARCH_ROUNDS) - 1:
        return "end"

    # Termination condition 3 — no new topics to debate
    if not state.get("next_debate_topics"):
        return "end"

    # Termination condition 4 — no progress (same gaps as last round)
    if current_gaps and previous_gaps and current_gaps == previous_gaps:
        return "end"

    # Continue — increment round and loop back
    return "continue"

def increment_round(state: ArchitectState) -> ArchitectState:
    """Increment round counter and save current gaps for progress detection."""
    return {
        **state,
        "research_round":  state["research_round"] + 1,
        "previous_gaps":   state.get("gaps", [])
    }

# ─── BUILD GRAPH ──────────────────────────────────────────────────────────────

def build_architect_graph():
    graph = StateGraph(ArchitectState)

    graph.add_node("load_evidence",         load_evidence)
    graph.add_node("analyse_evidence",      analyse_evidence)
    graph.add_node("design_strategy",       design_strategy)
    graph.add_node("deployment_decision",   make_deployment_decision)
    graph.add_node("fire_debates",          fire_debates)
    graph.add_node("increment_round",       increment_round)

    graph.set_entry_point("load_evidence")
    graph.add_edge("load_evidence",       "analyse_evidence")
    graph.add_edge("analyse_evidence",    "design_strategy")
    graph.add_edge("design_strategy",     "deployment_decision")
    graph.add_conditional_edges(
        "deployment_decision",
        should_continue,
        {
            "end":      "end_node",
            "continue": "fire_debates"
        }
    )
    graph.add_edge("fire_debates",        "increment_round")
    graph.add_edge("increment_round",     "load_evidence")  # loop back

    # End node
    graph.add_node("end_node", lambda s: {
        **s,
        "termination_reason": (
            "READY" if s["deployment_decision"] == "READY"
            else f"max_rounds_reached ({s['research_round']+1})"
            if s["research_round"] >= s["max_research_rounds"]-1
            else "no_progress" if set(s.get("gaps",[])) == set(s.get("previous_gaps",[]))
            else "no_new_topics"
        )
    })
    graph.add_edge("end_node", END)

    return graph.compile()

# ─── RUN FUNCTION ─────────────────────────────────────────────────────────────

def run_strategy_architect(
    max_research_rounds: int = MAX_RESEARCH_ROUNDS
) -> dict:
    """
    Run the fully autonomous Strategy Architect.
    Bounded loop: reads evidence → decides → fires debates → re-reads → repeats.
    Stops when READY or safety limits reached.
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
        "next_debate_topics":  [],
        "research_round":      0,
        "max_research_rounds": max_research_rounds,
        "previous_gaps":       [],
        "debates_fired":       [],
        "termination_reason":  "",
        "research_log":        []
    }

    # recursion_limit = nodes(6) × max_rounds(5) × 2 + buffer = 70
    final_state = app.invoke(
        initial_state,
        config={"recursion_limit": 70}
    )

    return {
        "deployment_decision":  final_state["deployment_decision"],
        "strategy_spec":        final_state["strategy_spec"],
        "analysis":             final_state["analysis"],
        "deployment_reason":    final_state["deployment_reason"],
        "gaps":                 final_state["gaps"],
        "next_debate_topics":   final_state["next_debate_topics"],
        "termination_reason":   final_state["termination_reason"],
        "research_rounds":      final_state["research_round"] + 1,
        "debates_fired":        final_state["debates_fired"],
        "research_log":         final_state["research_log"],
        "debates_read":         len(final_state["debate_results"]),
        "backtests_read":       len(final_state["backtest_results"])
    }


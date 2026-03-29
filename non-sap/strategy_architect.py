# strategy_architect.py — Stage 2: Strategy Architect Agent
#
# DESIGN: Single-step per HTTP call. Cloud Scheduler drives the loop.
# Each call: reads evidence → analyses → decides → fires ONE debate if needed → saves state
# Loop terminates when: READY | max_rounds reached | no progress
#
# RULE: Only this agent can declare a strategy deployable.
# RULE: No other agent can say "ready to deploy".

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

MAX_ROUNDS_TOTAL = 10      # absolute max calls before giving up
DEBATE_ROUNDS    = 2       # rounds per debate

# ─── STATE TABLE ──────────────────────────────────────────────────────────────
# Persisted in Supabase so each call picks up where the last left off

def load_architect_state() -> Dict:
    """Load persisted architect state from Supabase."""
    import json
    try:
        result = supabase.table("architect_state").select("*").order(
            "id", desc=True
        ).limit(1).execute()
        if result.data:
            row = result.data[0]
            # Parse JSON strings back to lists
            for field in ["debates_fired", "previous_gaps"]:
                val = row.get(field, [])
                if isinstance(val, str):
                    try:
                        row[field] = json.loads(val)
                    except Exception:
                        row[field] = []
                elif val is None:
                    row[field] = []
            return row
    except Exception:
        pass
    return {
        "round": 0,
        "deployment_decision": "NEEDS_MORE_RESEARCH",
        "debates_fired": [],
        "previous_gaps": [],
        "terminated": False,
        "termination_reason": ""
    }

def save_architect_state(state: Dict):
    """Save architect state to Supabase."""
    import json
    try:
        supabase.table("architect_state").insert({
            "round":               int(state.get("round", 0)),
            "deployment_decision": str(state.get("deployment_decision", "")),
            "debates_fired":       json.dumps(state.get("debates_fired", [])),
            "previous_gaps":       json.dumps(state.get("previous_gaps", [])),
            "terminated":          bool(state.get("terminated", False)),
            "termination_reason":  str(state.get("termination_reason", "")),
            "strategy_spec":       str(state.get("strategy_spec", ""))[:3000],
            "analysis":            str(state.get("analysis", ""))[:3000]
        }).execute()
    except Exception as e:
        print(f"architect_state save error: {e}")  # log but don't fail

# ─── STATE ────────────────────────────────────────────────────────────────────

class ArchitectState(TypedDict):
    debate_results: List[Dict]
    backtest_results: List[Dict]
    analysis: str
    strategy_spec: str
    deployment_decision: str
    deployment_reason: str
    gaps: List[str]
    next_debate_topic: str       # ONE topic per run
    round: int
    previous_gaps: List[str]
    debates_fired: List[str]
    terminated: bool
    termination_reason: str
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
        f"Round {state['round']+1}: loaded {len(debate_results)} debates, "
        f"{len(backtest_results)} backtests"
    )

    return {
        **state,
        "debate_results":   debate_results,
        "backtest_results": backtest_results,
        "research_log":     log
    }

# ─── NODE 2 — ANALYSE ─────────────────────────────────────────────────────────

def analyse_evidence(state: ArchitectState) -> ArchitectState:
    debate_summary = ""
    for i, row in enumerate(state["debate_results"]):
        debate_summary += f"""
DEBATE {i+1} — {row.get('topic','?')}
  Consensus: {row.get('consensus_reached')} | Confidence: {row.get('confidence','')}
  Validated: {row.get('validated_edge','')[:300]}
  Formula:   {row.get('mathematical_formulation','')[:300]}
  Gaps:      {row.get('remaining_gaps','')[:200]}
"""

    backtest_summary = "\n".join([
        f"  {r.get('parameters',{}).get('symbol','?')} "
        f"return={r.get('total_return',0):.3f} "
        f"wr={r.get('win_rate',0):.3f} "
        f"passed={r.get('constraints_met')}"
        for r in state["backtest_results"][-10:]
    ])

    prompt = f"""You are the Strategy Architect.
Round {state['round']+1}. Debates: {len(state['debate_results'])}. Backtests: {len(state['backtest_results'])}.
Previously fired debates: {state.get('debates_fired', [])}

DEBATE EVIDENCE:
{debate_summary or "None yet."}

BACKTEST EVIDENCE:
{backtest_summary or "None yet."}

Analyse all evidence. Identify:
- CONFIRMED EDGES: empirically proven across debates
- CONFIRMED REJECTIONS: empirically disproven
- FRAMEWORK COVERAGE: macro/fundamental/technical — what is done vs missing
- REMAINING GAPS: numbered, specific, what is still missing

Structure as:
CONFIRMED EDGES: ...
CONFIRMED REJECTIONS: ...
FRAMEWORK COVERAGE: ...
REMAINING GAPS: [numbered list]
SYNTHESIS: [how pieces fit together]"""

    response = llm.invoke(prompt)
    return {**state, "analysis": response.content}

# ─── NODE 3 — DESIGN ──────────────────────────────────────────────────────────

def design_strategy(state: ArchitectState) -> ArchitectState:
    prompt = f"""You are the Strategy Architect. Round {state['round']+1}.

Analysis:
{state['analysis']}

Design the most complete strategy justified by current evidence.
Mark each component: VALIDATED / ASSUMED / REJECTED.

STRATEGY NAME:
CORE HYPOTHESIS:
SIGNAL: [exact formula — validated only]
POSITION SIZING: [VALIDATED or ASSUMED]
EXIT RULES: [VALIDATED or ASSUMED]
UNIVERSE:
REGIME FILTER: [VALIDATED or ASSUMED]
EXPECTED PERFORMANCE: [from backtest evidence only]
CONFIDENCE: [HIGH/MEDIUM/LOW per component]"""

    response = llm.invoke(prompt)
    return {**state, "strategy_spec": response.content}

# ─── NODE 4 — DECIDE ──────────────────────────────────────────────────────────

def make_decision(state: ArchitectState) -> ArchitectState:
    already_fired = state.get("debates_fired", [])

    prompt = f"""You are the Strategy Architect. Round {state['round']+1} of {MAX_ROUNDS_TOTAL}.

Strategy:
{state['strategy_spec']}

Analysis:
{state['analysis']}

Already debated topics (do NOT suggest these again):
{already_fired}

Make deployment decision:
READY: all components empirically validated — deploy to paper trading
NOT_READY: core signal proven but gaps remain
NEEDS_MORE_RESEARCH: core signal unproven

Identify the SINGLE most critical open question to debate next.
It must be different from already-debated topics.
It must be an open question — no indicator names or answers in the question.

DECISION: [READY / NOT_READY / NEEDS_MORE_RESEARCH]
REASON: [specific justification]
WHAT IS PROVEN: [bullet list]
WHAT IS MISSING: [bullet list]
NEXT DEBATE TOPIC: [single most critical open question — one line only]
REVIEW CRITERIA: [what must be true to upgrade decision]"""

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

    # Extract single next debate topic
    next_topic = ""
    if "NEXT DEBATE TOPIC:" in verdict.upper():
        idx = verdict.upper().find("NEXT DEBATE TOPIC:")
        rest = verdict[idx + len("NEXT DEBATE TOPIC:"):].strip()
        # Take first non-empty line
        for line in rest.split("\n"):
            line = line.strip().lstrip("•-*123456789. ")
            if line and len(line) > 15:
                next_topic = line
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

    log = state.get("research_log", [])
    log.append(f"Round {state['round']+1}: decision={decision} | next_topic={next_topic[:60]}")

    return {
        **state,
        "deployment_decision": decision,
        "deployment_reason":   verdict,
        "next_debate_topic":   next_topic,
        "gaps":                gaps,
        "research_log":        log
    }

# ─── NODE 5 — FIRE ONE DEBATE ─────────────────────────────────────────────────

def fire_one_debate(state: ArchitectState) -> ArchitectState:
    """Fire exactly ONE debate on the most critical gap. Save to Supabase."""
    log = state.get("research_log", [])
    topic = state.get("next_debate_topic", "")
    debates_fired = list(state.get("debates_fired", []))

    if not topic:
        log.append(f"Round {state['round']+1}: no topic to debate — terminating")
        return {**state, "research_log": log, "terminated": True,
                "termination_reason": "no_topic"}

    if topic in debates_fired:
        log.append(f"Round {state['round']+1}: topic already debated — terminating")
        return {**state, "research_log": log, "terminated": True,
                "termination_reason": "no_new_topics"}

    log.append(f"Round {state['round']+1}: firing debate → {topic[:80]}")

    try:
        result = run_debate(
            topic=topic,
            context="S&P 500 systematic trading strategy research.",
            max_rounds=DEBATE_ROUNDS
        )

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
        log.append(f"  ✓ debate saved to Supabase")

    except Exception as e:
        log.append(f"  ✗ debate failed: {str(e)[:80]}")

    return {
        **state,
        "debates_fired": debates_fired,
        "research_log":  log,
        "round":         state["round"] + 1,
        "previous_gaps": state.get("gaps", [])
    }

# ─── NODE 6 — CHECK TERMINATION ───────────────────────────────────────────────

def check_termination(state: ArchitectState) -> ArchitectState:
    """Check if we should stop. Save state to Supabase for next call."""
    decision  = state.get("deployment_decision", "")
    round_num = state.get("round", 0)
    current_gaps  = set(state.get("gaps", []))
    previous_gaps = set(state.get("previous_gaps", []))

    terminated = False
    reason = ""

    if decision == "READY":
        terminated = True; reason = "READY"
    elif round_num >= MAX_ROUNDS_TOTAL:
        terminated = True; reason = f"max_rounds_reached ({round_num})"
    elif not state.get("next_debate_topic"):
        terminated = True; reason = "no_new_topics"
    elif current_gaps and previous_gaps and current_gaps == previous_gaps:
        terminated = True; reason = "no_progress"
    elif state.get("terminated"):
        reason = state.get("termination_reason", "unknown")

    # Save state to Supabase for next scheduled call
    save_architect_state({
        "round":               round_num,
        "deployment_decision": decision,
        "debates_fired":       state.get("debates_fired", []),
        "previous_gaps":       list(current_gaps),
        "terminated":          terminated,
        "termination_reason":  reason,
        "strategy_spec":       state.get("strategy_spec", ""),
        "analysis":            state.get("analysis", "")
    })

    log = state.get("research_log", [])
    log.append(f"Termination check: terminated={terminated} reason={reason}")

    return {
        **state,
        "terminated":         terminated,
        "termination_reason": reason,
        "research_log":       log
    }

# ─── ROUTING ──────────────────────────────────────────────────────────────────

def route_after_decision(state: ArchitectState) -> str:
    if state.get("deployment_decision") == "READY":
        return "end"
    if state.get("round", 0) >= MAX_ROUNDS_TOTAL:
        return "end"
    if not state.get("next_debate_topic"):
        return "end"
    return "fire"

def route_after_fire(state: ArchitectState) -> str:
    if state.get("terminated"):
        return "end"
    return "end"  # Always end after one debate — next call picks up

# ─── BUILD GRAPH ──────────────────────────────────────────────────────────────

def build_architect_graph():
    graph = StateGraph(ArchitectState)

    graph.add_node("load_evidence",   load_evidence)
    graph.add_node("analyse",         analyse_evidence)
    graph.add_node("design",          design_strategy)
    graph.add_node("decide",          make_decision)
    graph.add_node("fire_debate",     fire_one_debate)
    graph.add_node("check_terminate", check_termination)
    graph.add_node("end_node",        lambda s: s)

    graph.set_entry_point("load_evidence")
    graph.add_edge("load_evidence", "analyse")
    graph.add_edge("analyse",       "design")
    graph.add_edge("design",        "decide")
    graph.add_conditional_edges("decide", route_after_decision,
                                 {"end": "end_node", "fire": "fire_debate"})
    graph.add_edge("fire_debate",     "check_terminate")
    graph.add_edge("check_terminate", "end_node")
    graph.add_edge("end_node",        END)

    return graph.compile()

# ─── RUN FUNCTION ─────────────────────────────────────────────────────────────

def run_strategy_architect() -> dict:
    """
    Single-step Strategy Architect.
    Each call: reads evidence → analyses → decides → fires ONE debate → saves state.
    Cloud Scheduler calls this repeatedly to drive the research loop.
    Call manually from UI to see current state at any time.
    """
    # Load persisted state from previous calls
    persisted = load_architect_state()

    app = build_architect_graph()

    initial_state: ArchitectState = {
        "debate_results":      [],
        "backtest_results":    [],
        "analysis":            "",
        "strategy_spec":       "",
        "deployment_decision": persisted.get("deployment_decision", "NEEDS_MORE_RESEARCH"),
        "deployment_reason":   "",
        "gaps":                [],
        "next_debate_topic":   "",
        "round":               persisted.get("round", 0),
        "previous_gaps":       persisted.get("previous_gaps", []),
        "debates_fired":       persisted.get("debates_fired", []),
        "terminated":          persisted.get("terminated", False),
        "termination_reason":  persisted.get("termination_reason", ""),
        "research_log":        [f"Resuming from round {persisted.get('round',0)}"]
    }

    # recursion_limit = 7 nodes × 2 + buffer = 24
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
        "next_debate_topic":    final_state["next_debate_topic"],
        "termination_reason":   final_state["termination_reason"],
        "terminated":           final_state["terminated"],
        "round":                final_state["round"],
        "debates_fired":        final_state["debates_fired"],
        "research_log":         final_state["research_log"],
        "debates_read":         len(final_state["debate_results"]),
        "backtests_read":       len(final_state["backtest_results"])
    }

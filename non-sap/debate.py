# debate.py — Dual-agent debate tool
# Architect vs Critic debate on trading strategy design decisions
# Plugs into existing LangGraph framework as a standalone graph

import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

# ─── STATE ────────────────────────────────────────────────────────────────────

class DebateState(TypedDict):
    topic: str                  # what we are debating
    context: str                # optional background context
    max_rounds: int             # adjustable — default 3
    round: int                  # current round number
    history: List[dict]         # list of {role, content} per round
    architect_position: str     # current architect proposal
    critic_objections: str      # current critic objections
    consensus_reached: bool     # judge decision
    final_verdict: str          # judge summary
    decision: str               # final agreed decision

# ─── NODE 1 — ARCHITECT ───────────────────────────────────────────────────────

def architect(state: DebateState) -> DebateState:
    is_first_round = state["round"] == 0

    if is_first_round:
        prompt = f"""You are the Architect of a systematic S&P 500 trading strategy.
Your job is to propose a clear, well-reasoned design decision on the following topic.

Topic: {state['topic']}
Context: {state['context'] or 'No additional context provided.'}

Propose your design decision. Be specific and justify each choice with logic or evidence.
Structure your response as:
PROPOSAL: [your specific proposal in 1-2 sentences]
REASONING: [3-5 bullet points justifying this approach]
EXPECTED OUTCOME: [what this achieves for the strategy]"""
    else:
        last_objections = state["critic_objections"]
        prompt = f"""You are the Architect of a systematic S&P 500 trading strategy.
Topic: {state['topic']}
Round: {state['round']} of {state['max_rounds']}

Your previous proposal was challenged by the Critic with these objections:
{last_objections}

Respond to each objection specifically. Either defend your original proposal with stronger evidence,
or refine it to address valid concerns. Do not abandon your position without good reason.

Structure your response as:
REFINED PROPOSAL: [your updated or defended proposal]
RESPONSES TO OBJECTIONS: [address each objection point by point]
POSITION: [MAINTAINED / REFINED — and why]"""

    response = llm.invoke(prompt)
    position = response.content

    new_history = state["history"] + [{
        "role": "Architect",
        "round": state["round"] + 1,
        "content": position
    }]

    return {**state, "architect_position": position, "history": new_history}

# ─── NODE 2 — CRITIC ──────────────────────────────────────────────────────────

def critic(state: DebateState) -> DebateState:
    prompt = f"""You are the Critic reviewing a proposed trading strategy design decision.
Your job is to stress-test the Architect's proposal and find weaknesses.

Topic: {state['topic']}
Round: {state['round'] + 1} of {state['max_rounds']}

Architect's current proposal:
{state['architect_position']}

Debate history so far:
{_format_history(state['history'][:-1])}

Critically evaluate this proposal. Look for:
- Overfitting or curve-fitting risks
- Data snooping or look-ahead bias
- Unrealistic assumptions
- Missing edge cases
- Better alternatives that were not considered
- Logical inconsistencies

If the proposal is genuinely strong and previous objections have been addressed well,
you may concede on specific points. Be intellectually honest.

Structure your response as:
OBJECTIONS: [numbered list of specific challenges]
CONCESSIONS: [any points where the Architect is correct]
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

# ─── NODE 3 — JUDGE ───────────────────────────────────────────────────────────

def judge(state: DebateState) -> DebateState:
    prompt = f"""You are an independent Judge evaluating a strategy design debate.
Topic: {state['topic']}
Rounds completed: {state['round']} of {state['max_rounds']}

Full debate transcript:
{_format_history(state['history'])}

Based on the full debate:
1. Has a defensible consensus been reached?
2. What is the final recommended decision?
3. What risks or caveats remain?

Structure your response as:
CONSENSUS: [YES / NO / PARTIAL]
FINAL DECISION: [the specific design decision to implement]
RATIONALE: [why this is the right decision based on the debate]
REMAINING RISKS: [what to monitor or validate going forward]
CONFIDENCE: [HIGH / MEDIUM / LOW]"""

    response = llm.invoke(prompt)
    verdict = response.content

    # Check if consensus reached
    consensus = "YES" in verdict.upper().split("CONSENSUS:")[1][:20] if "CONSENSUS:" in verdict.upper() else False

    # Extract final decision
    decision = ""
    if "FINAL DECISION:" in verdict.upper():
        parts = verdict.upper().split("FINAL DECISION:")
        if len(parts) > 1:
            decision = verdict.split(verdict.upper().split("FINAL DECISION:")[0])[-1]
            decision = decision.split("FINAL DECISION:")[-1].split("\n")[0].strip()

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
    if state["consensus_reached"]:
        return "judge"
    return "architect"

# ─── UTILS ───────────────────────────────────────────────────────────────────

def _format_history(history: List[dict]) -> str:
    if not history:
        return "No history yet."
    lines = []
    for entry in history:
        lines.append(f"[Round {entry['round']} — {entry['role']}]")
        lines.append(entry['content'])
        lines.append("")
    return "\n".join(lines)

# ─── BUILD GRAPH ──────────────────────────────────────────────────────────────

def build_debate_graph():
    graph = StateGraph(DebateState)

    graph.add_node("architect", architect)
    graph.add_node("critic", critic)
    graph.add_node("judge", judge)

    graph.set_entry_point("architect")
    graph.add_edge("architect", "critic")
    graph.add_conditional_edges("critic", should_continue)
    graph.add_edge("judge", END)

    return graph.compile()

# ─── RUN FUNCTION ─────────────────────────────────────────────────────────────

def run_debate(topic: str, context: str = "", max_rounds: int = 3) -> dict:
    app = build_debate_graph()

    initial_state: DebateState = {
        "topic": topic,
        "context": context,
        "max_rounds": max_rounds,
        "round": 0,
        "history": [],
        "architect_position": "",
        "critic_objections": "",
        "consensus_reached": False,
        "final_verdict": "",
        "decision": ""
    }

    final_state = app.invoke(
        initial_state,
        config={"recursion_limit": 50}
    )

    return {
        "topic": final_state["topic"],
        "rounds_completed": final_state["round"],
        "consensus_reached": final_state["consensus_reached"],
        "final_verdict": final_state["final_verdict"],
        "decision": final_state["decision"],
        "history": final_state["history"]
    }

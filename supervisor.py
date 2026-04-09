from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from agents import critic_agent, planner_agent, researcher_agent
from config import SUPERVISOR_PROMPT, get_settings
from schemas import CritiqueResult, ResearchPlan
from tools import save_report
from langchain_core.tools import tool
settings = get_settings()

MAX_REVISION_ROUNDS = 2


def _extract_text_from_state(state: Any) -> str:
    if isinstance(state, dict):
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if text:
                            parts.append(str(text))
                    elif isinstance(item, str):
                        parts.append(item)
                if parts:
                    return "\n".join(parts)
        structured = state.get("structured_response")
        if structured is not None:
            return str(structured)

    if isinstance(state, AIMessage):
        return str(state.content)

    return str(state)


def _safe_filename_from_request(request: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", request.lower()).strip("_")
    if not slug:
        slug = "research_report"
    slug = slug[:60]
    return f"{slug}.md"


def plan(request: str) -> ResearchPlan:
    print(f"\n[Supervisor → Planner]\n plan({request!r})")
    result = planner_agent.invoke({"messages": [{"role": "user", "content": request}]})
    plan_obj: ResearchPlan = result["structured_response"]
    print(f" {json.dumps(plan_obj.model_dump(), ensure_ascii=False, indent=2)}")
    return plan_obj


def research(request: str) -> str:
    print(f"\n[Supervisor → Researcher]\n research({request!r})")
    result = researcher_agent.invoke({"messages": [{"role": "user", "content": request}]})
    findings = _extract_text_from_state(result)
    preview = findings.replace("\n", " ")[:300]
    suffix = "..." if len(findings) > 300 else ""
    print(f" {preview}{suffix}")
    return findings


def critique(
    *,
    original_request: str,
    plan_obj: ResearchPlan,
    findings: str,
) -> CritiqueResult:
    critique_input = f"""
Original user request:
{original_request}

Approved research plan:
{json.dumps(plan_obj.model_dump(), ensure_ascii=False, indent=2)}

Current findings:
{findings}
""".strip()

    preview = findings[:180]
    print(f"\n[Supervisor → Critic]\n critique({preview!r}{'...' if len(findings) > 180 else ''})")

    result = critic_agent.invoke({"messages": [{"role": "user", "content": critique_input}]})
    critique_obj: CritiqueResult = result["structured_response"]
    print(f" {json.dumps(critique_obj.model_dump(), ensure_ascii=False, indent=2)}")
    return critique_obj


def _build_research_request(
    *,
    original_request: str,
    plan_obj: ResearchPlan,
    round_index: int,
    critique_obj: CritiqueResult | None = None,
    previous_findings: str | None = None,
) -> str:
    if critique_obj is not None and critique_obj.revision_requests:
        queries = _dedupe_queries(critique_obj.revision_requests, limit=2)
    else:
        queries = _dedupe_queries(plan_obj.search_queries, limit=3)
    lines = [
        f"Original user request:\n{original_request}",
        "",
        f"Research goal:\n{plan_obj.goal}",
        "",
        "Search queries:",
    ]

    for q in plan_obj.search_queries:
        lines.append(f"- {q}")

    lines.extend(
        [
            "",
            f"Preferred sources: {', '.join(plan_obj.sources_to_check)}",
            f"Expected output format: {plan_obj.output_format}",
            "",
            f"Current round: {round_index}",
        ]
    )

    if previous_findings:
        lines.extend(
            [
                "",
                "Previous findings summary:",
                previous_findings[:4000],
            ]
        )

    if critique_obj and critique_obj.revision_requests:
        lines.extend(
            [
                "",
                "You are revising existing research. Focus only on these critique requests:",
            ]
        )
        for item in critique_obj.revision_requests:
            lines.append(f"- {item}")

        if critique_obj.gaps:
            lines.extend(["", "Known gaps to address:"])
            for gap in critique_obj.gaps:
                lines.append(f"- {gap}")

        lines.extend(
            [
                "",
                "Do not restart from scratch. Improve the existing findings and fill only the missing essential gaps.",
            ]
        )

    return "\n".join(lines)


def _build_final_report(
    *,
    original_request: str,
    plan_obj: ResearchPlan,
    findings: str,
    final_critique: CritiqueResult | None,
    revision_rounds_used: int,
) -> str:
    lines = [
        f"# Research Report",
        "",
        f"## User Request",
        original_request,
        "",
        f"## Research Goal",
        plan_obj.goal,
        "",
        f"## Findings",
        findings.strip(),
        "",
        "## Process Summary",
        f"- Revision rounds used: {revision_rounds_used}",
    ]

    if final_critique is not None:
        lines.extend(
            [
                f"- Final critic verdict: {final_critique.verdict}",
                f"- Freshness: {'yes' if final_critique.is_fresh else 'no'}",
                f"- Completeness: {'yes' if final_critique.is_complete else 'no'}",
                f"- Structure: {'yes' if final_critique.is_well_structured else 'no'}",
            ]
        )

        if final_critique.strengths:
            lines.extend(["", "## Strengths"])
            for item in final_critique.strengths:
                lines.append(f"- {item}")

        if final_critique.gaps:
            lines.extend(["", "## Remaining Limitations"])
            for item in final_critique.gaps:
                lines.append(f"- {item}")

    return "\n".join(lines).strip() + "\n"


model = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
    temperature=0.1,
    timeout=settings.request_timeout_seconds,
)

checkpointer = InMemorySaver()
supervisor_save_report = tool(
    "save_report",
    description="Save the final approved markdown report to disk and return the saved file path.",
)(save_report)
save_agent = create_agent(
    model=model,
    tools=[supervisor_save_report],
    system_prompt=SUPERVISOR_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "save_report": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                }
            },
            description_prefix="Report save pending approval",
        )
    ],
    checkpointer=checkpointer,
)


def run_supervisor(user_request: str) -> dict[str, Any]:
    plan_obj = plan(user_request)

    findings = research(
        _build_research_request(
            original_request=user_request,
            plan_obj=plan_obj,
            round_index=1,
        )
    )

    final_critique: CritiqueResult | None = None
    revision_rounds_used = 0

    for revision_round in range(MAX_REVISION_ROUNDS + 1):
        critique_obj = critique(
            original_request=user_request,
            plan_obj=plan_obj,
            findings=findings,
        )
        final_critique = critique_obj

        if critique_obj.verdict == "APPROVE":
            revision_rounds_used = revision_round
            break

        if revision_round == MAX_REVISION_ROUNDS:
            print("\n[Supervisor] Max revision rounds reached. Proceeding with current findings.")
            revision_rounds_used = revision_round
            break

        findings = research(
            _build_research_request(
                original_request=user_request,
                plan_obj=plan_obj,
                round_index=revision_round + 2,
                critique_obj=critique_obj,
                previous_findings=findings,
            )
        )
        revision_rounds_used = revision_round + 1

    final_report = _build_final_report(
        original_request=user_request,
        plan_obj=plan_obj,
        findings=findings,
        final_critique=final_critique,
        revision_rounds_used=revision_rounds_used,
    )

    return {
        "filename": _safe_filename_from_request(user_request),
        "content": final_report,
        "plan": plan_obj,
        "findings": findings,
        "critique": final_critique,
        "revision_rounds_used": revision_rounds_used,
    }

def request_save_report(report: dict[str, Any], thread_id: str):
    return save_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Save this approved report.\n\n"
                        f"Filename: {report['filename']}\n\n"
                        f"Content:\n{report['content']}"
                    ),
                }
            ]
        },
        config={"configurable": {"thread_id": thread_id}},
    )


def new_thread_id() -> str:
    return str(uuid.uuid4())

def _dedupe_queries(queries: list[str], limit: int = 3) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for q in queries:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(q.strip())
        if len(result) >= limit:
            break
    return result
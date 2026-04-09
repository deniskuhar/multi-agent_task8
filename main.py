from __future__ import annotations

from typing import Any

from langgraph.types import Command

from config import APP_TITLE, SEPARATOR
from supervisor import (
    new_thread_id,
    request_save_report,
    run_supervisor,
)
from tools import save_report

def _extract_final_text(result: Any) -> str:
    state = getattr(result, "value", result)

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

    return str(state)


def _get_interrupts(result: Any) -> list[Any]:
    interrupts = getattr(result, "interrupts", None)
    if not interrupts:
        return []
    return list(interrupts)


def _show_interrupt(interrupt: Any) -> None:
    data = getattr(interrupt, "value", {}) or {}
    action_requests = data.get("action_requests", [])

    print("\n" + SEPARATOR)
    print("  ACTION REQUIRES APPROVAL")
    print(SEPARATOR)

    for request in action_requests:
        print(f"  Tool:  {request.get('name')}")
        print(f"  Args:  {request.get('arguments')}")
        description = request.get("description")
        if description:
            print(f"  Note:  {description}")


def _resume_from_interrupt(thread_id: str) -> tuple[str, str | None]:
    while True:
        decision = input("\n approve / edit / reject: ").strip().lower()

        if decision not in {"approve", "edit", "reject"}:
            print("Please enter approve, edit, or reject.")
            continue

        if decision == "approve":
            command = Command(resume={"decisions": [{"type": "approve"}]})
            return "approve", command

        if decision == "edit":
            feedback = input("✏️  Your feedback: ").strip()
            command = Command(
                resume={
                    "decisions": [
                        {
                            "type": "edit",
                            "edited_action": {
                                "feedback": feedback,
                            },
                        }
                    ]
                }
            )
            return "edit", command

        reason = input(" Reason (optional): ").strip() or "User rejected the save action."
        command = Command(
            resume={
                "decisions": [
                    {
                        "type": "reject",
                        "message": reason,
                    }
                ]
            }
        )
        return "reject", command


def _handle_save_flow(report: dict[str, Any], thread_id: str) -> str:
    while True:
        print("\n" + SEPARATOR)
        print("  ACTION REQUIRES APPROVAL")
        print(SEPARATOR)
        print(f"  Tool:  save_report")
        print(f"  Filename:  {report['filename']}")
        preview = report["content"][:800]
        print(f"  Preview:\n{preview}")
        if len(report["content"]) > 800:
            print("\n  ...")

        decision = input("\n approve / edit / reject: ").strip().lower()

        if decision not in {"approve", "edit", "reject"}:
            print("Please enter approve, edit, or reject.")
            continue

        if decision == "approve":
            result = save_report(
                filename=report["filename"],
                content=report["content"],
            )
            return result

        if decision == "edit":
            feedback = input("✏️  Your feedback: ").strip()
            from supervisor import revise_report_with_feedback
            report = revise_report_with_feedback(report, feedback)
            print("\n[Supervisor] Report revised based on your feedback.")
            continue

        return "Report saving was cancelled."
    

def main() -> None:
    thread_id = new_thread_id()

    print(SEPARATOR)
    print(APP_TITLE)
    print("Type 'exit' or 'quit' to leave. Type 'new' to reset the session.")
    print(SEPARATOR)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if user_input.lower() == "new":
            thread_id = new_thread_id()
            print("Started a new session.")
            continue

        try:
            report = run_supervisor(user_input)

            print("\n" + SEPARATOR)
            print("Draft report prepared")
            print(SEPARATOR)
            print(f"Filename: {report['filename']}")
            preview = report["content"][:1500]
            print(preview)
            if len(report["content"]) > 1500:
                print("\n...")

            final_message = _handle_save_flow(report, thread_id)
            print(f"\nAgent: {final_message}")

        except Exception as exc:
            print(f"\nAgent error: {exc}")


if __name__ == "__main__":
    main()
"""Student LLM client — calls LM Studio's OpenAI-compatible API."""

import json
import os
import traceback as tb_module
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
BASE_DIR = Path(__file__).parent.parent


def _read_system_prompt() -> str:
    path = BASE_DIR / "student_config" / "system_prompt.md"
    return path.read_text(encoding="utf-8")


def _read_tools() -> list:
    path = BASE_DIR / "student_config" / "tools.json"
    content = path.read_text(encoding="utf-8").strip()
    return json.loads(content) if content else []


def _call_lm_studio(messages: list, tools: Optional[list] = None, temperature: float = 0.7) -> dict:
    """Send a request to LM Studio and return the raw response dict."""
    payload: dict = {
        "model": "local-model",  # LM Studio ignores this but requires it
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    with httpx.Client(timeout=None) as client:
        resp = client.post(
            f"{LM_STUDIO_URL}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


def ask_student(
    question: str,
    conversation_history: list,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    tools: Optional[list] = None,
    tools_dir: Optional[Path] = None,
) -> dict:
    """
    Ask the student LLM a question.

    Returns:
        {
            "answer": str,
            "tool_calls_made": list,
            "execution_trace": list,   # full record of every step + errors
            "had_errors": bool,
        }
    """
    from backend.tool_executor import execute_tool  # lazy import to avoid circular

    execution_trace = []
    effective_system_prompt = system_prompt if system_prompt is not None else _read_system_prompt()
    effective_tools = tools if tools is not None else _read_tools()

    messages = [{"role": "system", "content": effective_system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": question})

    # Initial LM Studio call
    response = _call_lm_studio(
        messages, tools=effective_tools if effective_tools else None, temperature=temperature
    )
    choice = response["choices"][0]
    message = choice["message"]

    tool_calls_made = []
    final_answer = message.get("content") or ""

    execution_trace.append(
        {
            "step": "initial_response",
            "content": final_answer,
            "tool_calls": message.get("tool_calls") or [],
        }
    )

    # Handle tool calls if any
    if message.get("tool_calls"):
        tool_calls_made = message["tool_calls"]
        messages.append(message)

        for tc in message["tool_calls"]:
            tool_name = tc["function"]["name"]
            try:
                arguments = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                arguments = {}

            try:
                result = execute_tool(tool_name, arguments, tools_dir=tools_dir)
                execution_trace.append(
                    {
                        "step": "tool_execution",
                        "tool": tool_name,
                        "arguments": arguments,
                        "status": "success",
                        "result": result,
                    }
                )
                tool_result_content = result
            except Exception as exc:
                execution_trace.append(
                    {
                        "step": "tool_execution",
                        "tool": tool_name,
                        "arguments": arguments,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": tb_module.format_exc(),
                    }
                )
                tool_result_content = f"ERROR: {type(exc).__name__}: {exc}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result_content,
                }
            )

        # Second LM Studio call with tool results
        try:
            response2 = _call_lm_studio(
                messages, tools=effective_tools if effective_tools else None, temperature=temperature
            )
            message2 = response2["choices"][0]["message"]
            final_answer = message2.get("content") or ""
            execution_trace.append(
                {
                    "step": "final_response_after_tools",
                    "content": final_answer,
                }
            )
        except Exception as exc:
            execution_trace.append(
                {
                    "step": "final_response_after_tools",
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    had_errors = any(t.get("status") == "error" for t in execution_trace)

    return {
        "answer": final_answer,
        "tool_calls_made": tool_calls_made,
        "execution_trace": execution_trace,
        "had_errors": had_errors,
    }


def ask_student_direct(
    question: str,
    context: str = "",
    system_prompt: Optional[str] = None,
) -> str:
    """
    Ask the student a direct question from the master during evaluation.
    Uses the student's current system prompt but no tools (pure Q&A).
    """
    effective_system_prompt = system_prompt if system_prompt is not None else _read_system_prompt()
    full_question = f"{context}\n\n{question}" if context else question

    messages = [
        {"role": "system", "content": effective_system_prompt},
        {"role": "user", "content": full_question},
    ]

    response = _call_lm_studio(messages, tools=None)
    return response["choices"][0]["message"].get("content") or ""


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is 2 + 2?"
    print(f"Question: {question}")
    print("Calling LM Studio...")
    result = ask_student(question, [])
    print(f"\nAnswer: {result['answer']}")
    print(f"Had errors: {result['had_errors']}")
    print(f"Execution trace steps: {[t['step'] for t in result['execution_trace']]}")

"""Master LLM client — evaluates student answers via Claude Opus 4.6."""

import json
import os
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

BASE_DIR = Path(__file__).parent.parent
MODEL = "claude-opus-4-6"

MASTER_SYSTEM_PROMPT = """You are a master AI teacher supervising a beginner student LLM. The student is a small local model with limited knowledge and capabilities. Your job:

1. EVALUATE: Judge whether the student's answer is good enough to show the user.
2. DIAGNOSE: If not, figure out WHY it failed.
3. ACT: Choose the right intervention to fix it.

You have these intervention tools available:

- edit_system_prompt: Rewrite the student's permanent system prompt. Use when you see a pattern that better instructions would fix (e.g., student keeps hallucinating, being too verbose, not using tools properly).

- add_tool: Create a new tool the student can use. Provide the tool name, description, parameters schema (JSON), and the Python implementation. Use when the student lacks a capability it needs (e.g., can't look up weather → create a weather API tool).

- remove_tool: Remove a tool that's causing problems.

- retry_with_hint: Don't change anything permanent — just retry the question with a one-time hint appended. Use for one-off mistakes.

- edit_code: Edit any file in the project. Use sparingly and only when the system itself needs a fix (not just the student's behavior). Provide the file path and the new content or a diff.

- ask_student: Ask the student a direct question. The student will answer and you'll receive the response to continue your evaluation. Use this to:
  - Probe the student's reasoning ("Why did you say X?")
  - Test the student's knowledge ("Do you know what Y is?")
  - Check if the student understands its limitations ("Are you sure this is current info?")
  - Explore what the student thinks it needs ("What would help you answer this better?")
  Payload: {"question": "your question to the student"}
  IMPORTANT: You can ask up to 3 questions per evaluation. Each ask_student action pauses evaluation — you'll receive the student's response and can then continue deciding on your verdict and other actions.

- pass_answer: The answer is good enough. Deliver it to the user.

- fail_with_explanation: After max retries, give up and explain to the user what happened. Provide your own best answer.

IMPORTANT RULES:
- Always start by evaluating. Don't over-intervene.
- For time-sensitive or factual questions, you can use your own knowledge to verify.
- Log your reasoning in the teaching journal.
- If you add a tool, make sure the tool implementation is complete, working Python code.
- Prefer the lightest intervention that solves the problem.
- Keep a teaching_journal entry for every FAIL — what went wrong, what you did, and what to watch for.

ERROR HANDLING RULES:
- You will receive the student's full execution trace, including any tool call attempts and their results or errors.
- If you see tool errors:
  - Read the traceback carefully to understand what went wrong.
  - If it's a bug in tool code YOU previously wrote, fix it using the edit_code action.
  - If the student called the tool with wrong parameters, consider improving the tool description or the system prompt to guide proper usage.
  - If a tool is fundamentally broken (e.g., missing external dependency), remove it and try a different approach.

Respond in this exact JSON format:
{
  "verdict": "PASS" | "FAIL",
  "reasoning": "Your analysis of the student's answer",
  "quality_scores": {
    "factual_accuracy": 1-5,
    "completeness": 1-5,
    "honesty": 1-5,
    "usefulness": 1-5
  },
  "actions": [
    {
      "type": "pass_answer" | "retry_with_hint" | "edit_system_prompt" | "add_tool" | "remove_tool" | "edit_code" | "ask_student" | "fail_with_explanation",
      "payload": { }
    }
  ],
  "teaching_journal_entry": "Optional note to add to the teaching journal"
}

Action payloads:
- pass_answer: {"final_answer": "the student's answer, possibly lightly edited"}
- retry_with_hint: {"hint": "text to append to the user's question as context"}
- edit_system_prompt: {"new_prompt": "the full new system prompt text"}
- add_tool: {"name": "tool_name", "description": "what it does", "parameters": {...json schema...}, "implementation": "full Python code as string"}
- remove_tool: {"name": "tool_name"}
- edit_code: {"file_path": "relative path", "new_content": "full file content"}
- ask_student: {"question": "your question to the student"}
- fail_with_explanation: {"explanation": "what to tell the user", "master_answer": "your best answer"}

IMPORTANT: Your entire response must be valid JSON. Do not include any text before or after the JSON object."""


def _read_system_prompt() -> str:
    path = BASE_DIR / "student_config" / "system_prompt.md"
    return path.read_text(encoding="utf-8")


def _read_tools() -> list:
    path = BASE_DIR / "student_config" / "tools.json"
    content = path.read_text(encoding="utf-8").strip()
    return json.loads(content) if content else []


def _read_journal() -> str:
    path = BASE_DIR / "master_config" / "teaching_journal.md"
    return path.read_text(encoding="utf-8")


def _extract_json(text: str) -> dict:
    """Extract JSON from the response, handling markdown code blocks."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from master response:\n{text}")


def _build_evaluation_prompt(
    question: str,
    student_answer: str,
    execution_trace: list,
    had_errors: bool,
    system_prompt_content: str,
    tools_content: list,
    journal: str,
    attempt: int,
) -> str:
    return f"""## Evaluation Request (Attempt {attempt}/3)

### User's Question:
{question}

### Student's Answer:
{student_answer}

### Execution Trace:
{json.dumps(execution_trace, indent=2)}

### Errors Occurred: {"Yes" if had_errors else "No"}

### Student's Current System Prompt:
{system_prompt_content}

### Student's Available Tools:
{json.dumps(tools_content, indent=2)}

### Teaching Journal (your past observations):
{journal}

Evaluate this answer and decide what to do."""


def evaluate(question: str, student_result: dict, attempt: int) -> dict:
    """
    Send the student's result to the master for evaluation.
    Supports multi-turn dialogue: if the master asks the student a question,
    the response is fed back for the master to continue evaluating.

    Returns the parsed evaluation dict with keys:
        verdict, reasoning, quality_scores, actions,
        teaching_journal_entry, student_dialogue
    """
    from backend.student import ask_student_direct  # avoid circular at module level

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system_prompt_content = _read_system_prompt()
    tools_content = _read_tools()
    journal = _read_journal()

    student_answer = student_result.get("answer", "")
    execution_trace = student_result.get("execution_trace", [])
    had_errors = student_result.get("had_errors", False)

    initial_prompt = _build_evaluation_prompt(
        question,
        student_answer,
        execution_trace,
        had_errors,
        system_prompt_content,
        tools_content,
        journal,
        attempt,
    )

    master_messages = [{"role": "user", "content": initial_prompt}]

    max_student_questions = 3
    questions_asked = 0
    student_dialogue = []

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=MASTER_SYSTEM_PROMPT,
            messages=master_messages,
        )

        raw_text = response.content[0].text
        result = _extract_json(raw_text)

        # Check if master wants to ask the student something
        ask_actions = [a for a in result.get("actions", []) if a["type"] == "ask_student"]

        if ask_actions and questions_asked < max_student_questions:
            ask_action = ask_actions[0]
            student_question = ask_action["payload"]["question"]

            student_response = ask_student_direct(
                student_question,
                context=f"The master teacher is asking you a follow-up about your answer to: {question}",
            )

            questions_asked += 1
            student_dialogue.append(
                {
                    "master_question": student_question,
                    "student_response": student_response,
                }
            )

            # Feed response back to master and loop
            master_messages.append({"role": "assistant", "content": raw_text})
            master_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Student's response to your question '{student_question}':\n\n"
                        f"{student_response}\n\n"
                        f"Continue your evaluation. You have "
                        f"{max_student_questions - questions_asked} questions remaining. "
                        "Provide your final verdict and actions."
                    ),
                }
            )
        else:
            # Final evaluation — strip any remaining ask_student actions
            result["actions"] = [a for a in result.get("actions", []) if a["type"] != "ask_student"]
            result["student_dialogue"] = student_dialogue
            return result


# ---------------------------------------------------------------------------
# Teacher's Lounge — direct chat with the master
# ---------------------------------------------------------------------------

MASTER_DIRECT_CHAT_SYSTEM_PROMPT = """You are the master teacher in a master-student LLM teaching system. You're now in a direct conversation with the human operator (the person who runs this system).

You have full context of the current system state provided below, including:
- The student's current system prompt
- The student's available tools
- Your teaching journal
- Recent conversation logs between the user and the student

## Your Role in This Chat

You're having a collegial conversation with the operator about the student's development. You should:
- Be insightful and opinionated about the student's progress
- Discuss your teaching strategy openly
- Explain your past decisions when asked
- Accept directives from the operator and incorporate them into your approach
- Proactively suggest improvements you've been thinking about

## Taking Actions

If the conversation leads to a decision to change something, you can take action right here. Embed an action block anywhere in your response using this exact format:

```action
{"type": "edit_system_prompt", "payload": {"new_prompt": "..."}}
```

Supported action types and their payloads:
- edit_system_prompt: {"new_prompt": "the full new system prompt text"}
- add_tool: {"name": "tool_name", "description": "what it does", "parameters": {...json schema...}, "implementation": "full Python code as string"}
- remove_tool: {"name": "tool_name"}
- edit_code: {"file_path": "relative path", "new_content": "full file content"}

For example, if the operator says "add a calculator tool", discuss the approach AND embed an action block to create it in the same response. Always explain what you're doing and why.

You can also just chat without taking any actions — not every conversation needs to result in a change.

## Tone

Be natural and conversational. You're a thoughtful mentor discussing a student with a colleague. Share your observations, concerns, and ideas. Feel free to be candid about the student's limitations — the operator knows it's a small model and expects honest assessment.

Don't use the structured JSON evaluation format here — that's for the automated evaluation loop. Just talk naturally, and embed action blocks only when actually making changes."""


def try_extract_actions(response_text: str) -> list:
    """Extract ```action blocks from the master's conversational response."""
    pattern = r"```action\s*\n(.*?)\n```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    actions = []
    for match in matches:
        try:
            actions.append(json.loads(match.strip()))
        except json.JSONDecodeError:
            continue
    return actions


def compress_journal(journal_text: str) -> str:
    """
    Ask the master to summarize one session's teaching journal.
    Returns a concise summary suitable for the cross-session journal.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=(
            "You are the master teacher in a master-student LLM teaching system. "
            "You maintain a teaching journal that tracks student progress, issues, and interventions."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize this teaching session journal in exactly 3-4 bullet points. "
                    "One sentence per bullet. Be terse and specific — no preamble, no headers, just bullets. "
                    "Cover: behavior patterns observed, any interventions made (tools added, prompt edits, etc.), "
                    "and the single most important thing to watch for next session.\n\n"
                    f"Teaching Journal:\n{journal_text}\n\n"
                    "Write the bullet points now (start with -):"
                ),
            }
        ],
    )
    return response.content[0].text


def direct_chat(history: list, context_block: str) -> str:
    """
    One turn of the master direct chat.
    history: list of {"role": "user"|"assistant", "content": str} messages
    context_block: current system state (injected into the system prompt)
    Returns the master's response text.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=MASTER_DIRECT_CHAT_SYSTEM_PROMPT + "\n\n" + context_block,
        messages=history,
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("Testing master evaluator with a hardcoded student result...")

    test_question = "What is the capital of France?"
    test_result = {
        "answer": "The capital of France is Paris.",
        "execution_trace": [
            {"step": "initial_response", "content": "The capital of France is Paris.", "tool_calls": []}
        ],
        "had_errors": False,
    }

    result = evaluate(test_question, test_result, attempt=1)
    print(json.dumps(result, indent=2))

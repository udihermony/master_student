"""FastAPI orchestrator — the master-student teaching loop."""

import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure backend package is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(Path(__file__).parent.parent / ".env")

from backend.master import compress_journal, direct_chat, evaluate, try_extract_actions
from backend.student import ask_student
from backend.tool_executor import add_tool, remove_tool

BASE_DIR = Path(__file__).parent.parent
SYSTEM_PROMPT_PATH = BASE_DIR / "student_config" / "system_prompt.md"
TOOLS_JSON_PATH = BASE_DIR / "student_config" / "tools.json"
JOURNAL_PATH = BASE_DIR / "master_config" / "teaching_journal.md"
LOG_PATH = BASE_DIR / "logs" / "conversation_log.jsonl"
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"
MASTER_CHAT_HISTORY_PATH = BASE_DIR / "master_config" / "master_chat_history.json"
CROSS_SESSION_JOURNAL_PATH = BASE_DIR / "master_config" / "cross_session_journal.md"
SESSIONS_DIR = BASE_DIR / "master_config" / "sessions"
TEACHER_CONFIG_PATH = BASE_DIR / "master_config" / "teacher_config.json"

FRESH_CROSS_SESSION_TEXT = (
    "# Cross-Session Journal\n\n"
    "*Compressed summaries of each teaching session, maintained by the master across resets.*\n\n"
    "---\n\n"
    "(No sessions compressed yet.)\n"
)

_DEFAULT_TEACHER_CONFIG = {
    "master_model": "claude-opus-4-6",
    "max_attempts": 3,
    "max_student_questions": 3,
    "student_temperature": 0.7,
}


def _read_teacher_config() -> dict:
    if TEACHER_CONFIG_PATH.exists():
        try:
            return json.loads(TEACHER_CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            pass
    return dict(_DEFAULT_TEACHER_CONFIG)

FRESH_JOURNAL_TEXT = (
    "# Teaching Journal\n\n"
    "*This journal is maintained by the master to track the student's progress, "
    "recurring issues, and interventions made.*\n\n---\n\n"
    "(No entries yet. The journey begins.)\n"
)

INITIAL_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's questions to the best of your ability.\n"
    "Be concise and accurate. If you're not sure about something, say so.\n"
)

app = FastAPI(title="Master-Student LLM Experiment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket connections for status streaming
_ws_connections: list[WebSocket] = []

# Conversation history (in-memory, per session — cleared on /reset)
_conversation_history: list[dict] = []

# Master direct-chat history (persisted to disk)
def _load_master_chat_history() -> list:
    if MASTER_CHAT_HISTORY_PATH.exists():
        try:
            return json.loads(MASTER_CHAT_HISTORY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return []
    return []

_master_chat_history: list = _load_master_chat_history()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    attempts: int
    verdict: str
    scores: Optional[dict] = None
    reasoning: Optional[str] = None
    explanation: Optional[str] = None
    actions_taken: list = []
    execution_trace: list = []
    student_dialogue: list = []


class SystemPromptUpdate(BaseModel):
    prompt: str


class TeacherConfig(BaseModel):
    master_model: str = "claude-opus-4-6"
    max_attempts: int = 3
    max_student_questions: int = 3
    student_temperature: float = 0.7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_tools() -> list:
    content = TOOLS_JSON_PATH.read_text(encoding="utf-8").strip() if TOOLS_JSON_PATH.exists() else "[]"
    return json.loads(content) if content else []


def _append_to_journal(entry: str) -> None:
    journal = _read_file(JOURNAL_PATH)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    new_entry = f"\n\n---\n**{timestamp}**\n\n{entry}"
    JOURNAL_PATH.write_text(journal + new_entry, encoding="utf-8")


def _log_interaction(data: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


async def _broadcast_status(message: str) -> None:
    """Send a status update to all connected WebSocket clients."""
    dead = []
    for ws in _ws_connections:
        try:
            await ws.send_text(json.dumps({"type": "status", "message": message}))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_connections.remove(ws)


# ---------------------------------------------------------------------------
# Master chat helpers
# ---------------------------------------------------------------------------


def _save_master_chat_history(history: list) -> None:
    MASTER_CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MASTER_CHAT_HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _read_recent_logs(last_n: int = 20) -> list:
    """Return a trimmed summary of the last N log entries (to avoid token bloat)."""
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    recent = lines[-last_n:]
    entries = []
    for line in recent:
        try:
            entry = json.loads(line)
            entries.append(
                {
                    "timestamp": entry.get("timestamp", ""),
                    "question": entry.get("question", ""),
                    "student_answer": str(entry.get("student_answer", ""))[:300],
                    "had_errors": entry.get("had_errors", False),
                    "verdict": entry.get("evaluation", {}).get("verdict", ""),
                }
            )
        except json.JSONDecodeError:
            pass
    return entries


def _build_master_context_block() -> str:
    """Build the current system state block injected into the master's system prompt."""
    system_prompt = _read_file(SYSTEM_PROMPT_PATH)
    tools = _read_tools()
    journal = _read_file(JOURNAL_PATH)
    recent_logs = _read_recent_logs(last_n=20)
    return (
        "<current_system_state>\n"
        f"## Student's System Prompt:\n{system_prompt}\n\n"
        f"## Student's Available Tools:\n{json.dumps(tools, indent=2)}\n\n"
        f"## Teaching Journal:\n{journal}\n\n"
        f"## Recent Student Conversations (last 20 interactions):\n"
        f"{json.dumps(recent_logs, indent=2)}\n"
        "</current_system_state>"
    )


def _execute_master_action(action: dict) -> str:
    """Execute a single action from the master's teacher's lounge response."""
    action_type = action.get("type")
    payload = action.get("payload", {})
    try:
        if action_type == "edit_system_prompt":
            SYSTEM_PROMPT_PATH.write_text(payload.get("new_prompt", ""), encoding="utf-8")
            return "System prompt updated"
        elif action_type == "add_tool":
            add_tool(
                name=payload["name"],
                description=payload["description"],
                parameters=payload["parameters"],
                implementation=payload["implementation"],
            )
            return f"Tool '{payload['name']}' added"
        elif action_type == "remove_tool":
            remove_tool(payload["name"])
            return f"Tool '{payload['name']}' removed"
        elif action_type == "edit_code":
            fp = payload.get("file_path", "").strip()
            if not fp:
                return "Action failed: edit_code requires a non-empty file_path"
            file_path = BASE_DIR / fp
            if file_path == BASE_DIR or file_path.is_dir():
                return f"Action failed: '{fp}' resolves to a directory, not a file"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(payload.get("new_content", ""), encoding="utf-8")
            return f"File '{fp}' updated"
        else:
            return f"Unknown action type: {action_type}"
    except Exception as exc:
        return f"Action failed: {exc}"


# ---------------------------------------------------------------------------
# Journal compression
# ---------------------------------------------------------------------------

_CROSS_SESSION_BOILERPLATE = "(No sessions compressed yet.)"

# Patterns that indicate a boilerplate-only journal (nothing real was written)
_BOILERPLATE_JOURNAL_PATTERNS = [
    r"\(No entries yet\.",
    r"\(New session started",
]


def _journal_has_real_content(journal_text: str) -> bool:
    """Return True only if the journal has actual teaching entries beyond the header boilerplate."""
    # Find the end of the last boilerplate line
    last_pos = 0
    for pattern in _BOILERPLATE_JOURNAL_PATTERNS:
        m = re.search(pattern, journal_text)
        if m:
            # Advance past the entire line containing the match
            line_end = journal_text.find("\n", m.end())
            last_pos = max(last_pos, line_end if line_end != -1 else m.end())
    # If there's non-whitespace content after all boilerplate lines, it's real
    return bool(journal_text[last_pos:].strip())


def _archive_journal() -> Optional[str]:
    """
    Compress the current teaching journal:
    1. Ask the master to summarize it.
    2. Save the full journal to master_config/sessions/session_<timestamp>.md.
    3. Append the summary to master_config/cross_session_journal.md.
    4. Reset teaching_journal.md to a fresh state.

    Returns the summary text, or None if the journal had no meaningful content.
    """
    journal_text = _read_file(JOURNAL_PATH)

    # Skip if there's nothing worth compressing
    if not journal_text.strip() or not _journal_has_real_content(journal_text):
        return None

    # Ask the master to summarize
    cfg = _read_teacher_config()
    summary = compress_journal(journal_text, model=cfg["master_model"])

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")

    # Archive full session journal
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_file = SESSIONS_DIR / f"session_{timestamp}.md"
    session_file.write_text(journal_text, encoding="utf-8")

    # Append summary to cross-session journal (strip first-run boilerplate if present)
    cross = _read_file(CROSS_SESSION_JOURNAL_PATH)
    cross = cross.replace(_CROSS_SESSION_BOILERPLATE, "").strip()
    cross_entry = f"\n\n---\n## Session {timestamp}\n\n{summary}\n"
    CROSS_SESSION_JOURNAL_PATH.write_text(cross + cross_entry, encoding="utf-8")

    # Reset current journal with a note pointing to the archive
    JOURNAL_PATH.write_text(
        FRESH_JOURNAL_TEXT.replace(
            "(No entries yet. The journey begins.)",
            f"(New session started {timestamp}. Previous session archived → sessions/session_{timestamp}.md)",
        ),
        encoding="utf-8",
    )

    return summary


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    question = request.message
    cfg = _read_teacher_config()
    max_retries = int(cfg.get("max_attempts", 3))
    actions_taken: list[dict] = []

    for attempt in range(1, max_retries + 1):
        # 1. Ask the student
        await _broadcast_status(
            f"Asking student... (attempt {attempt}/{max_retries})"
        )
        try:
            student_result = ask_student(
                question, _conversation_history,
                temperature=float(cfg.get("student_temperature", 0.7)),
            )
        except Exception as exc:
            await _broadcast_status(f"Student error: {exc}")
            return ChatResponse(
                answer=f"Student LLM error: {exc}",
                attempts=attempt,
                verdict="ERROR",
            )

        # 2. Send to master for evaluation
        await _broadcast_status("Master evaluating student's answer...")
        try:
            evaluation = evaluate(
                question, student_result, attempt,
                model=cfg.get("master_model", "claude-opus-4-6"),
                max_student_questions=int(cfg.get("max_student_questions", 3)),
            )
        except Exception as exc:
            await _broadcast_status(f"Master error: {exc}")
            return ChatResponse(
                answer=f"Master evaluation error: {exc}",
                attempts=attempt,
                verdict="ERROR",
            )

        # 3. Log everything
        _log_interaction(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attempt": attempt,
                "question": question,
                "student_answer": student_result["answer"],
                "had_errors": student_result.get("had_errors", False),
                "execution_trace": student_result.get("execution_trace", []),
                "student_dialogue": evaluation.get("student_dialogue", []),
                "evaluation": evaluation,
            }
        )

        # 4. Record journal entry if master left one
        journal_entry = evaluation.get("teaching_journal_entry")

        # 5. Process master's actions
        for action in evaluation.get("actions", []):
            action_type = action.get("type")
            payload = action.get("payload", {})
            actions_taken.append({"type": action_type, "payload": payload})

            if action_type == "pass_answer":
                if journal_entry:
                    _append_to_journal(journal_entry)
                # Update conversation history
                _conversation_history.append({"role": "user", "content": request.message})
                _conversation_history.append(
                    {"role": "assistant", "content": payload.get("final_answer", student_result["answer"])}
                )
                await _broadcast_status("Answer approved. Delivering to user.")
                return ChatResponse(
                    answer=payload.get("final_answer", student_result["answer"]),
                    attempts=attempt,
                    verdict="PASS",
                    scores=evaluation.get("quality_scores"),
                    reasoning=evaluation.get("reasoning"),
                    actions_taken=actions_taken,
                    execution_trace=student_result.get("execution_trace", []),
                    student_dialogue=evaluation.get("student_dialogue", []),
                )

            elif action_type == "retry_with_hint":
                hint = payload.get("hint", "")
                question = f"{request.message}\n\n[Additional context: {hint}]"
                await _broadcast_status(f"Master added a hint. Retrying (attempt {attempt + 1})...")

            elif action_type == "edit_system_prompt":
                new_prompt = payload.get("new_prompt", "")
                SYSTEM_PROMPT_PATH.write_text(new_prompt, encoding="utf-8")
                await _broadcast_status("Master edited the system prompt. Retrying...")

            elif action_type == "add_tool":
                try:
                    add_tool(
                        name=payload["name"],
                        description=payload["description"],
                        parameters=payload["parameters"],
                        implementation=payload["implementation"],
                    )
                    await _broadcast_status(f"Master added tool '{payload['name']}'. Retrying...")
                except Exception as exc:
                    await _broadcast_status(f"Failed to add tool: {exc}")

            elif action_type == "remove_tool":
                try:
                    remove_tool(payload["name"])
                    await _broadcast_status(f"Master removed tool '{payload['name']}'.")
                except Exception as exc:
                    await _broadcast_status(f"Failed to remove tool: {exc}")

            elif action_type == "edit_code":
                fp = payload.get("file_path", "").strip()
                file_path = BASE_DIR / fp if fp else None
                new_content = payload.get("new_content", "")
                try:
                    if not fp or file_path == BASE_DIR or (file_path and file_path.is_dir()):
                        await _broadcast_status(f"edit_code skipped: invalid file_path '{fp}'")
                    else:
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_text(new_content, encoding="utf-8")
                        await _broadcast_status(f"Master edited code file: {fp}")
                except Exception as exc:
                    await _broadcast_status(f"Failed to edit code: {exc}")

            elif action_type == "fail_with_explanation":
                if journal_entry:
                    _append_to_journal(journal_entry)
                await _broadcast_status("Max retries reached. Master providing fallback answer.")
                return ChatResponse(
                    answer=payload.get("master_answer", "I was unable to get a reliable answer."),
                    explanation=payload.get("explanation", ""),
                    attempts=attempt,
                    verdict="FAIL_FINAL",
                    scores=evaluation.get("quality_scores"),
                    reasoning=evaluation.get("reasoning"),
                    actions_taken=actions_taken,
                    execution_trace=student_result.get("execution_trace", []),
                    student_dialogue=evaluation.get("student_dialogue", []),
                )

        if journal_entry:
            _append_to_journal(journal_entry)

    # Safety net — should not be reached
    return ChatResponse(
        answer="I wasn't able to get a reliable answer after all retries.",
        attempts=max_retries,
        verdict="ERROR",
        actions_taken=actions_taken,
    )


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------


@app.get("/system-prompt")
async def get_system_prompt():
    return {"prompt": _read_file(SYSTEM_PROMPT_PATH)}


@app.put("/system-prompt")
async def update_system_prompt(body: SystemPromptUpdate):
    SYSTEM_PROMPT_PATH.write_text(body.prompt, encoding="utf-8")
    return {"status": "ok"}


@app.get("/teacher-config")
async def get_teacher_config():
    return _read_teacher_config()


@app.put("/teacher-config")
async def update_teacher_config(config: TeacherConfig):
    TEACHER_CONFIG_PATH.write_text(
        json.dumps(config.model_dump(), indent=2), encoding="utf-8"
    )
    return {"status": "ok"}


@app.get("/tools")
async def get_tools():
    return {"tools": _read_tools()}


@app.get("/journal")
async def get_journal():
    return {"journal": _read_file(JOURNAL_PATH)}


@app.get("/cross-session-journal")
async def get_cross_session_journal():
    return {"journal": _read_file(CROSS_SESSION_JOURNAL_PATH)}


@app.get("/sessions")
async def list_sessions():
    if not SESSIONS_DIR.exists():
        return {"sessions": []}
    files = sorted(SESSIONS_DIR.glob("session_*.md"), reverse=True)
    return {"sessions": [f.name for f in files]}


@app.post("/compress-journal")
async def compress_journal_endpoint():
    await _broadcast_status("Compressing journal...")
    try:
        summary = _archive_journal()
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    if summary is None:
        return {"status": "skipped", "reason": "Journal has no meaningful content to compress"}
    # Note: no final broadcast here — the frontend fetch response handles the done state
    return {"status": "ok", "summary": summary}


@app.get("/logs")
async def get_logs(limit: int = 50):
    if not LOG_PATH.exists():
        return {"logs": []}
    lines = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    recent = lines[-limit:] if len(lines) > limit else lines
    entries = []
    for line in recent:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return {"logs": entries}


@app.post("/reset")
async def reset():
    """Compress the journal, then restore initial state for a fresh experiment."""
    # Compress current session journal before wiping
    await _broadcast_status("Compressing journal before reset...")
    try:
        _archive_journal()
    except Exception:
        pass  # Don't let compression failure block the reset

    SYSTEM_PROMPT_PATH.write_text(INITIAL_SYSTEM_PROMPT, encoding="utf-8")
    TOOLS_JSON_PATH.write_text("[]", encoding="utf-8")
    # Remove all tool Python files
    tools_dir = BASE_DIR / "student_config" / "tools"
    for f in tools_dir.glob("*.py"):
        if f.name != "example_tool.py":
            f.unlink()
    # Clear log
    LOG_PATH.write_text("", encoding="utf-8")
    # Journal was already reset by _archive_journal(); ensure it's fresh if archive was skipped
    if _journal_has_real_content(_read_file(JOURNAL_PATH)):
        JOURNAL_PATH.write_text(FRESH_JOURNAL_TEXT, encoding="utf-8")
    # Clear cross-session journal and all session archives
    CROSS_SESSION_JOURNAL_PATH.write_text(FRESH_CROSS_SESSION_TEXT, encoding="utf-8")
    if SESSIONS_DIR.exists():
        for f in SESSIONS_DIR.glob("*.md"):
            f.unlink()
    # Clear conversation history
    _conversation_history.clear()
    # Clear master chat history
    global _master_chat_history
    _master_chat_history = []
    _save_master_chat_history(_master_chat_history)
    return {"status": "reset complete"}


@app.post("/clear-history")
async def clear_history():
    """Wipe all logs and history without touching student config or compressing journals."""
    global _master_chat_history

    # Clear conversation log
    LOG_PATH.write_text("", encoding="utf-8")

    # Reset teaching journal to fresh state
    JOURNAL_PATH.write_text(FRESH_JOURNAL_TEXT, encoding="utf-8")

    # Reset cross-session journal to fresh state
    CROSS_SESSION_JOURNAL_PATH.write_text(FRESH_CROSS_SESSION_TEXT, encoding="utf-8")

    # Remove all archived session files
    if SESSIONS_DIR.exists():
        for f in SESSIONS_DIR.glob("*.md"):
            f.unlink()

    # Clear in-memory conversation history
    _conversation_history.clear()

    # Clear master chat history (in-memory + disk)
    _master_chat_history = []
    _save_master_chat_history(_master_chat_history)

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Teacher's Lounge — direct chat with the master
# ---------------------------------------------------------------------------


class MasterChatResponse(BaseModel):
    response: str
    actions_taken: list = []
    correction: Optional[str] = None
    correction_actions: list = []


@app.get("/master-chat/history")
async def get_master_chat_history():
    return {"history": _master_chat_history}


@app.post("/master-chat", response_model=MasterChatResponse)
async def master_chat(request: ChatRequest):
    global _master_chat_history

    # Add user turn
    _master_chat_history.append({"role": "user", "content": request.message})

    # Build fresh context block and call the master
    context_block = _build_master_context_block()
    cfg = _read_teacher_config()
    try:
        assistant_message = direct_chat(
            _master_chat_history, context_block,
            model=cfg.get("master_model", "claude-opus-4-6"),
        )
    except Exception as exc:
        _master_chat_history.pop()  # roll back the user message
        return MasterChatResponse(response=f"Master error: {exc}")

    # Record assistant turn
    _master_chat_history.append({"role": "assistant", "content": assistant_message})
    _save_master_chat_history(_master_chat_history)

    # Extract and execute any embedded action blocks
    actions_executed = []
    for action in try_extract_actions(assistant_message):
        result = _execute_master_action(action)
        actions_executed.append({"type": action.get("type"), "result": result})
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        _append_to_journal(f"[Teacher's Lounge — {timestamp}] {result}")

    # If any actions failed, feed the failures back so the master can correct itself
    failures = [a for a in actions_executed if (a.get("result") or "").startswith("Action failed")]
    correction_message = None
    correction_actions_executed = []
    if failures:
        failure_lines = "\n".join(f"- {a['type']}: {a['result']}" for a in failures)
        feedback = (
            f"[System] The following action(s) failed to execute:\n{failure_lines}\n\n"
            "Please review the action payload and retry with the correct parameters, "
            "or let the operator know what additional information you need."
        )
        _master_chat_history.append({"role": "user", "content": feedback})
        try:
            correction_message = direct_chat(
                _master_chat_history, context_block,
                model=cfg.get("master_model", "claude-opus-4-6"),
            )
            _master_chat_history.append({"role": "assistant", "content": correction_message})
            for action in try_extract_actions(correction_message):
                result = _execute_master_action(action)
                correction_actions_executed.append({"type": action.get("type"), "result": result})
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                _append_to_journal(f"[Teacher's Lounge correction — {timestamp}] {result}")
        except Exception:
            pass  # correction is best-effort; don't fail the whole request
        _save_master_chat_history(_master_chat_history)

    await _broadcast_status(
        "Teacher's Lounge: master responded"
        + (f" + {len(actions_executed)} action(s)" if actions_executed else "")
    )
    return MasterChatResponse(
        response=assistant_message,
        actions_taken=actions_executed,
        correction=correction_message,
        correction_actions=correction_actions_executed,
    )


@app.post("/master-chat/reset")
async def reset_master_chat():
    global _master_chat_history
    _master_chat_history = []
    _save_master_chat_history(_master_chat_history)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# WebSocket for status streaming
# ---------------------------------------------------------------------------


@app.websocket("/chat-stream")
async def chat_stream(websocket: WebSocket):
    await websocket.accept()
    _ws_connections.append(websocket)
    try:
        while True:
            # Keep connection alive; actual messages are broadcast from /chat
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_connections:
            _ws_connections.remove(websocket)


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------


@app.get("/")
async def serve_frontend():
    if FRONTEND_PATH.exists():
        return FileResponse(str(FRONTEND_PATH))
    return HTMLResponse("<h1>Frontend not found. Build frontend/index.html first.</h1>")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

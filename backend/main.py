"""FastAPI orchestrator — the master-student teaching loop."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure backend package is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(Path(__file__).parent.parent / ".env")

from backend.master import (
    DEFAULT_MASTER_BASE_PROMPT,
    _EVALUATION_TAIL,
    _LOUNGE_TAIL,
    compress_journal,
    direct_chat,
    evaluate,
    try_extract_actions,
)
from backend.student import ask_student
from backend.tool_executor import add_tool, remove_tool
import backend.db as db

BASE_DIR = Path(__file__).parent.parent
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"  # never changes

# ---------------------------------------------------------------------------
# Module-level mutable path globals — updated by _load_active_classroom()
# Initial values point to legacy flat layout (used only before migration)
# ---------------------------------------------------------------------------
_active_classroom_id: str = ""
SYSTEM_PROMPT_PATH: Path = BASE_DIR / "student_config" / "system_prompt.md"
TOOLS_JSON_PATH: Path = BASE_DIR / "student_config" / "tools.json"
TOOLS_DIR: Path = BASE_DIR / "student_config" / "tools"
JOURNAL_PATH: Path = BASE_DIR / "master_config" / "teaching_journal.md"
LOG_PATH: Path = BASE_DIR / "logs" / "conversation_log.jsonl"
MASTER_CHAT_HISTORY_PATH: Path = BASE_DIR / "master_config" / "master_chat_history.json"
TEACHER_CONFIG_PATH: Path = BASE_DIR / "master_config" / "teacher_config.json"
MASTER_PROMPT_PATH: Path = BASE_DIR / "master_config" / "master_prompt.md"

_DEFAULT_TEACHER_CONFIG = {
    "master_model": "claude-opus-4-6",
    "max_attempts": 3,
    "max_student_questions": 3,
    "student_temperature": 0.7,
    "student_name": "Student",
    "master_name": "Master",
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

# Current DB session ID
_current_session_id: str = ""

# Master direct-chat history (persisted to disk)
_master_chat_history: list = []


# ---------------------------------------------------------------------------
# Classroom helpers
# ---------------------------------------------------------------------------


def _classroom_dir(classroom_id: str) -> Path:
    return BASE_DIR / "classrooms" / classroom_id


def _load_master_chat_history() -> list:
    if MASTER_CHAT_HISTORY_PATH.exists():
        try:
            return json.loads(MASTER_CHAT_HISTORY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _load_active_classroom(classroom_id: str) -> None:
    """Update all path globals to point at the given classroom's directories."""
    global _active_classroom_id
    global SYSTEM_PROMPT_PATH, TOOLS_JSON_PATH, TOOLS_DIR
    global JOURNAL_PATH, LOG_PATH, MASTER_CHAT_HISTORY_PATH, TEACHER_CONFIG_PATH
    global MASTER_PROMPT_PATH, _master_chat_history

    _active_classroom_id = classroom_id
    c_dir = _classroom_dir(classroom_id)

    SYSTEM_PROMPT_PATH = c_dir / "student_config" / "system_prompt.md"
    TOOLS_JSON_PATH = c_dir / "student_config" / "tools.json"
    TOOLS_DIR = c_dir / "student_config" / "tools"
    JOURNAL_PATH = c_dir / "master_config" / "teaching_journal.md"
    LOG_PATH = c_dir / "logs" / "conversation_log.jsonl"
    MASTER_CHAT_HISTORY_PATH = c_dir / "master_config" / "master_chat_history.json"
    TEACHER_CONFIG_PATH = c_dir / "master_config" / "teacher_config.json"
    MASTER_PROMPT_PATH = c_dir / "master_config" / "master_prompt.md"

    _master_chat_history = _load_master_chat_history()


def _build_master_eval_prompt() -> str:
    base = _read_file(MASTER_PROMPT_PATH) or DEFAULT_MASTER_BASE_PROMPT
    return base + "\n\n" + _EVALUATION_TAIL


def _build_master_lounge_prompt() -> str:
    base = _read_file(MASTER_PROMPT_PATH) or DEFAULT_MASTER_BASE_PROMPT
    return base + "\n\n" + _LOUNGE_TAIL


def _create_classroom_skeleton(classroom_id: str) -> None:
    """Create directory structure and default files for a new classroom."""
    c_dir = _classroom_dir(classroom_id)
    (c_dir / "student_config" / "tools").mkdir(parents=True, exist_ok=True)
    (c_dir / "master_config").mkdir(parents=True, exist_ok=True)
    (c_dir / "logs").mkdir(parents=True, exist_ok=True)

    sp_path = c_dir / "student_config" / "system_prompt.md"
    if not sp_path.exists():
        sp_path.write_text(INITIAL_SYSTEM_PROMPT, encoding="utf-8")

    tools_path = c_dir / "student_config" / "tools.json"
    if not tools_path.exists():
        tools_path.write_text("[]", encoding="utf-8")

    mp_path = c_dir / "master_config" / "master_prompt.md"
    if not mp_path.exists():
        mp_path.write_text(DEFAULT_MASTER_BASE_PROMPT, encoding="utf-8")

    tc_path = c_dir / "master_config" / "teacher_config.json"
    if not tc_path.exists():
        tc_path.write_text(json.dumps(_DEFAULT_TEACHER_CONFIG, indent=2), encoding="utf-8")

    journal_path = c_dir / "master_config" / "teaching_journal.md"
    if not journal_path.exists():
        journal_path.write_text(FRESH_JOURNAL_TEXT, encoding="utf-8")

    history_path = c_dir / "master_config" / "master_chat_history.json"
    if not history_path.exists():
        history_path.write_text("[]", encoding="utf-8")

    log_path = c_dir / "logs" / "conversation_log.jsonl"
    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")


def _migrate_if_needed() -> None:
    """
    First-run migration: copy flat student_config/master_config/logs into
    classrooms/default/, write master_prompt.md, insert DB row, activate.
    On subsequent runs, just loads the active classroom from DB.
    """
    default_dir = _classroom_dir("default")

    if not default_dir.exists():
        # First time — build the default classroom from the old flat layout
        default_dir.mkdir(parents=True, exist_ok=True)

        old_student = BASE_DIR / "student_config"
        old_master = BASE_DIR / "master_config"
        old_logs = BASE_DIR / "logs"

        if old_student.exists():
            shutil.copytree(str(old_student), str(default_dir / "student_config"), dirs_exist_ok=True)
        else:
            (default_dir / "student_config" / "tools").mkdir(parents=True, exist_ok=True)
            (default_dir / "student_config" / "system_prompt.md").write_text(
                INITIAL_SYSTEM_PROMPT, encoding="utf-8"
            )
            (default_dir / "student_config" / "tools.json").write_text("[]", encoding="utf-8")

        if old_master.exists():
            shutil.copytree(str(old_master), str(default_dir / "master_config"), dirs_exist_ok=True)
        else:
            (default_dir / "master_config").mkdir(parents=True, exist_ok=True)
            (default_dir / "master_config" / "teacher_config.json").write_text(
                json.dumps(_DEFAULT_TEACHER_CONFIG, indent=2), encoding="utf-8"
            )
            (default_dir / "master_config" / "teaching_journal.md").write_text(
                FRESH_JOURNAL_TEXT, encoding="utf-8"
            )
            (default_dir / "master_config" / "master_chat_history.json").write_text(
                "[]", encoding="utf-8"
            )

        if old_logs.exists():
            shutil.copytree(str(old_logs), str(default_dir / "logs"), dirs_exist_ok=True)
        else:
            (default_dir / "logs").mkdir(parents=True, exist_ok=True)
            (default_dir / "logs" / "conversation_log.jsonl").write_text("", encoding="utf-8")

        # Write master_prompt.md if not already copied from old master_config
        mp_path = default_dir / "master_config" / "master_prompt.md"
        if not mp_path.exists():
            mp_path.write_text(DEFAULT_MASTER_BASE_PROMPT, encoding="utf-8")

        # Insert default classroom into DB (is_active=1 set by set_active_classroom below)
        db.create_classroom("Default Classroom", classroom_id="default")
        db.set_active_classroom("default")

    # Load whichever classroom is active (covers both first-run and restarts)
    active_id = db.get_active_classroom_id() or "default"
    _load_active_classroom(active_id)


def _start_new_session() -> None:
    """Create a new DB session and update the global session ID."""
    global _current_session_id
    cfg = _read_teacher_config()
    system_prompt = _read_file(SYSTEM_PROMPT_PATH)
    _current_session_id = db.create_session(cfg, system_prompt, classroom_id=_active_classroom_id)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup():
    db.init_db()
    _migrate_if_needed()  # sets all path globals + _active_classroom_id
    _start_new_session()


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
    student_name: str = "Student"
    master_name: str = "Master"


class CreateClassroomRequest(BaseModel):
    name: str
    copy_from: Optional[str] = None


class MasterPromptUpdate(BaseModel):
    prompt: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_tools() -> list:
    content = TOOLS_JSON_PATH.read_text(encoding="utf-8").strip() if TOOLS_JSON_PATH.exists() else "[]"
    return json.loads(content) if content else []


def _append_to_journal(entry: str, entry_type: str = "evaluation") -> None:
    journal = _read_file(JOURNAL_PATH)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    new_entry = f"\n\n---\n**{timestamp}**\n\n{entry}"
    JOURNAL_PATH.write_text(journal + new_entry, encoding="utf-8")
    # Also log to DB
    if _current_session_id:
        db.log_journal_entry(_current_session_id, entry_type, entry)


def _log_interaction(data: dict) -> None:
    # Write to JSONL (compatibility fallback)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")
    # Write to DB
    if _current_session_id:
        cfg = _read_teacher_config()
        db.log_interaction(_current_session_id, data, cfg)


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


def _format_recent_sessions(sessions: list) -> str:
    if not sessions:
        return "(No previous sessions.)"
    lines = []
    for s in sessions:
        ended = s.get("ended_at") or "ongoing"
        ended_short = ended[:10] if ended != "ongoing" else "ongoing"
        summary = s.get("summary") or "(no summary)"
        lines.append(
            f"- **{s['id']}** ({s.get('student_name','Student')} / {s.get('master_name','Master')}) "
            f"started: {s.get('created_at','')[:10]}, ended: {ended_short}\n"
            f"  Summary: {summary}"
        )
    return "\n".join(lines)


def _build_master_context_block() -> str:
    """Build the current system state block injected into the master's system prompt."""
    system_prompt = _read_file(SYSTEM_PROMPT_PATH)
    tools = _read_tools()
    journal = _read_file(JOURNAL_PATH)
    recent_sessions = db.get_recent_sessions(limit=10)
    recent_interactions = db.get_recent_interactions(last_n=20)
    sessions_md = _format_recent_sessions(recent_sessions)
    return (
        "<current_system_state>\n"
        f"## Active Classroom: {_active_classroom_id} (files at classrooms/{_active_classroom_id}/)\n\n"
        f"## Student's System Prompt:\n{system_prompt}\n\n"
        f"## Student's Available Tools:\n{json.dumps(tools, indent=2)}\n\n"
        f"## Teaching Journal (current session):\n{journal}\n\n"
        f"## Past Sessions (last 10):\n{sessions_md}\n\n"
        f"## Recent Interactions (last 20):\n{json.dumps(recent_interactions, indent=2)}\n\n"
        "Note: Full interaction history is in SQLite DB at `data/master_student.db` "
        "(tables: sessions, interactions, journal_entries)\n"
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
                tools_dir=TOOLS_DIR,
                tools_json=TOOLS_JSON_PATH,
            )
            return f"Tool '{payload['name']}' added"
        elif action_type == "remove_tool":
            remove_tool(payload["name"], tools_dir=TOOLS_DIR, tools_json=TOOLS_JSON_PATH)
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
# Journal compression / session archiving
# ---------------------------------------------------------------------------


def _has_journal_content(journal_text: str) -> bool:
    """Return True if the journal has actual teaching entries beyond the header boilerplate."""
    skip_prefixes = ("#", "*", "---")
    boilerplate_starts = ("(No entries yet.", "(New session started")
    for line in journal_text.splitlines():
        s = line.strip()
        if not s:
            continue
        if any(s.startswith(p) for p in skip_prefixes):
            continue
        if any(s.startswith(b) for b in boilerplate_starts):
            continue
        return True
    return False


def _archive_journal() -> Optional[str]:
    """
    Compress the current teaching journal:
    1. Ask the master to summarize it.
    2. Save the full journal text as a 'session_archive' journal_entry in DB.
    3. Reset teaching_journal.md to a fresh state.

    Returns the summary text, or None if the journal had no meaningful content.
    Does NOT call db.end_session() — callers are responsible for ending the session.
    """
    journal_text = _read_file(JOURNAL_PATH)

    if not journal_text.strip() or not _has_journal_content(journal_text):
        return None

    cfg = _read_teacher_config()
    summary = compress_journal(journal_text, model=cfg["master_model"])

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")

    # Archive full session journal text to DB
    if _current_session_id:
        db.log_journal_entry(_current_session_id, "session_archive", journal_text)

    # Reset current journal with a note
    JOURNAL_PATH.write_text(
        FRESH_JOURNAL_TEXT.replace(
            "(No entries yet. The journey begins.)",
            f"(New session started {timestamp}. Previous session archived to DB.)",
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
                question,
                _conversation_history,
                temperature=float(cfg.get("student_temperature", 0.7)),
                system_prompt=_read_file(SYSTEM_PROMPT_PATH),
                tools=_read_tools(),
                tools_dir=TOOLS_DIR,
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
                question,
                student_result,
                attempt,
                model=cfg.get("master_model", "claude-opus-4-6"),
                max_student_questions=int(cfg.get("max_student_questions", 3)),
                master_system_prompt=_build_master_eval_prompt(),
                student_system_prompt=_read_file(SYSTEM_PROMPT_PATH),
                tools_content=_read_tools(),
                journal=_read_file(JOURNAL_PATH),
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
                    _append_to_journal(journal_entry, entry_type="evaluation")
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
                        tools_dir=TOOLS_DIR,
                        tools_json=TOOLS_JSON_PATH,
                    )
                    await _broadcast_status(f"Master added tool '{payload['name']}'. Retrying...")
                except Exception as exc:
                    await _broadcast_status(f"Failed to add tool: {exc}")

            elif action_type == "remove_tool":
                try:
                    remove_tool(payload["name"], tools_dir=TOOLS_DIR, tools_json=TOOLS_JSON_PATH)
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
                    _append_to_journal(journal_entry, entry_type="evaluation")
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
            _append_to_journal(journal_entry, entry_type="evaluation")

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


@app.post("/compress-journal")
async def compress_journal_endpoint():
    await _broadcast_status("Compressing journal...")
    try:
        summary = _archive_journal()
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    if summary is None:
        return {"status": "skipped", "reason": "Journal has no meaningful content to compress"}
    # End the current session in DB and start a new one
    db.end_session(_current_session_id, summary)
    _start_new_session()
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
    global _master_chat_history

    # Compress current session journal before wiping
    await _broadcast_status("Compressing journal before reset...")
    try:
        summary = _archive_journal()
    except Exception:
        summary = None

    # End current DB session
    db.end_session(_current_session_id, summary)

    SYSTEM_PROMPT_PATH.write_text(INITIAL_SYSTEM_PROMPT, encoding="utf-8")
    TOOLS_JSON_PATH.write_text("[]", encoding="utf-8")
    # Remove all tool Python files in the active classroom
    for f in TOOLS_DIR.glob("*.py"):
        if f.name != "example_tool.py":
            f.unlink()
    # Clear JSONL log
    LOG_PATH.write_text("", encoding="utf-8")
    # Ensure journal is fresh (archive may have already reset it)
    if _has_journal_content(_read_file(JOURNAL_PATH)):
        JOURNAL_PATH.write_text(FRESH_JOURNAL_TEXT, encoding="utf-8")
    # Clear conversation history
    _conversation_history.clear()
    # Clear master chat history
    _master_chat_history = []
    _save_master_chat_history(_master_chat_history)

    # Start new DB session (with reset system prompt)
    _start_new_session()

    return {"status": "reset complete"}


@app.post("/new-chat")
async def new_chat():
    """Start a new chat session: compress journal, clear conversation. Student config unchanged."""
    global _master_chat_history

    # Archive current session journal so master retains cross-session memory
    await _broadcast_status("Archiving session journal...")
    try:
        summary = _archive_journal()
    except Exception:
        summary = None

    # End current DB session and start a new one
    db.end_session(_current_session_id, summary)
    _start_new_session()

    # Clear JSONL conversation log
    LOG_PATH.write_text("", encoding="utf-8")

    # Clear in-memory conversation history
    _conversation_history.clear()

    # Clear master chat history (new session, fresh Teacher's Lounge)
    _master_chat_history = []
    _save_master_chat_history(_master_chat_history)

    return {"status": "ok"}


@app.post("/clear-history")
async def clear_history():
    """Wipe all logs and history without touching student config or compressing journals."""
    global _master_chat_history

    # Clear JSONL log
    LOG_PATH.write_text("", encoding="utf-8")

    # Reset teaching journal to fresh state
    JOURNAL_PATH.write_text(FRESH_JOURNAL_TEXT, encoding="utf-8")

    # Clear in-memory conversation history
    _conversation_history.clear()

    # Clear master chat history (in-memory + disk)
    _master_chat_history = []
    _save_master_chat_history(_master_chat_history)

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# DB query endpoints
# ---------------------------------------------------------------------------


@app.get("/db/sessions")
async def db_list_sessions():
    """List all sessions with metadata and interaction counts."""
    return {"sessions": db.get_all_sessions()}


@app.get("/db/sessions/{session_id}/interactions")
async def db_session_interactions(session_id: str):
    """Return all interactions for a specific session."""
    return {"interactions": db.get_session_interactions(session_id)}


@app.get("/db/export")
async def db_export():
    """Export all interactions as JSONL (for fine-tuning)."""
    with tempfile.NamedTemporaryFile(
        suffix=".jsonl", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)
    count = db.export_jsonl(tmp_path)
    return FileResponse(
        path=str(tmp_path),
        filename="master_student_interactions.jsonl",
        media_type="application/x-ndjson",
        headers={"X-Row-Count": str(count)},
    )


# ---------------------------------------------------------------------------
# Open DB in external viewer
# ---------------------------------------------------------------------------


@app.post("/open-db")
async def open_db():
    """Open the SQLite database file in the system's associated viewer (macOS: DB Browser)."""
    db_path = db.DB_PATH
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database file not found — start the server first to create it.")
    try:
        subprocess.Popen(["open", str(db_path)])
        return {"status": "ok", "path": str(db_path)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to open database: {exc}")


# ---------------------------------------------------------------------------
# Classroom management endpoints
# ---------------------------------------------------------------------------


@app.get("/classrooms")
async def list_classrooms():
    return {"classrooms": db.get_classrooms(), "active_id": _active_classroom_id}


@app.post("/classrooms")
async def create_classroom_endpoint(request: CreateClassroomRequest):
    classroom_id = db.create_classroom(request.name)
    c_dir = _classroom_dir(classroom_id)

    if request.copy_from:
        src_dir = _classroom_dir(request.copy_from)
        if src_dir.exists():
            shutil.copytree(str(src_dir), str(c_dir))
        else:
            _create_classroom_skeleton(classroom_id)
    else:
        _create_classroom_skeleton(classroom_id)

    return {"id": classroom_id, "name": request.name}


@app.post("/classrooms/{classroom_id}/activate")
async def activate_classroom(classroom_id: str):
    global _master_chat_history

    if classroom_id == _active_classroom_id:
        return {"status": "already active"}

    # Verify classroom exists
    classrooms = db.get_classrooms()
    if not any(c["id"] == classroom_id for c in classrooms):
        raise HTTPException(status_code=404, detail=f"Classroom '{classroom_id}' not found")

    # End current session without compressing
    db.end_session(_current_session_id, None)

    db.set_active_classroom(classroom_id)
    _load_active_classroom(classroom_id)
    _conversation_history.clear()
    _start_new_session()

    return {"status": "activated", "classroom_id": classroom_id}


@app.delete("/classrooms/{classroom_id}")
async def delete_classroom(classroom_id: str):
    if classroom_id == _active_classroom_id:
        raise HTTPException(status_code=400, detail="Cannot delete the active classroom")
    if classroom_id == "default":
        raise HTTPException(status_code=400, detail="Cannot delete the default classroom")

    c_dir = _classroom_dir(classroom_id)
    if c_dir.exists():
        shutil.rmtree(str(c_dir))
    db.delete_classroom(classroom_id)

    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Master prompt endpoints
# ---------------------------------------------------------------------------


@app.get("/master-prompt")
async def get_master_prompt():
    return {"prompt": _read_file(MASTER_PROMPT_PATH)}


@app.put("/master-prompt")
async def update_master_prompt(body: MasterPromptUpdate):
    MASTER_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MASTER_PROMPT_PATH.write_text(body.prompt, encoding="utf-8")
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
            _master_chat_history,
            context_block,
            model=cfg.get("master_model", "claude-opus-4-6"),
            master_system_prompt=_build_master_lounge_prompt(),
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
        _append_to_journal(f"[Teacher's Lounge — {timestamp}] {result}", entry_type="teacher_lounge")

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
                _master_chat_history,
                context_block,
                model=cfg.get("master_model", "claude-opus-4-6"),
                master_system_prompt=_build_master_lounge_prompt(),
            )
            _master_chat_history.append({"role": "assistant", "content": correction_message})
            for action in try_extract_actions(correction_message):
                result = _execute_master_action(action)
                correction_actions_executed.append({"type": action.get("type"), "result": result})
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                _append_to_journal(
                    f"[Teacher's Lounge correction — {timestamp}] {result}",
                    entry_type="teacher_lounge",
                )
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

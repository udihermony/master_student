"""SQLite database module for Master-Student LLM logging."""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "data" / "master_student.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS classrooms (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                ended_at TEXT,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                master_id TEXT NOT NULL,
                master_name TEXT NOT NULL,
                master_model TEXT NOT NULL,
                student_temperature REAL,
                max_attempts INTEGER,
                max_student_questions INTEGER,
                summary TEXT
            );

            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                student_answer TEXT,
                attempt INTEGER,
                verdict TEXT,
                had_errors INTEGER,
                quality_scores TEXT,
                reasoning TEXT,
                actions_taken TEXT,
                execution_trace TEXT,
                student_dialogue TEXT,
                master_model TEXT,
                student_temperature REAL
            );

            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                timestamp TEXT NOT NULL,
                entry_type TEXT NOT NULL,
                content TEXT NOT NULL
            );
        """)
        conn.commit()
        # Add classroom_id column to sessions if it doesn't exist yet
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN classroom_id TEXT REFERENCES classrooms(id)")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
    finally:
        conn.close()


def _make_session_id() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("sess_%Y%m%d_%H%M%S")


def _student_id(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode()).hexdigest()[:8]


def create_session(cfg: dict, system_prompt: str, classroom_id: Optional[str] = None) -> str:
    """Create a new session row and return the session_id."""
    session_id = _make_session_id()
    created_at = datetime.now(timezone.utc).isoformat()
    sid = _student_id(system_prompt)
    student_name = cfg.get("student_name", "Student")
    master_model = cfg.get("master_model", "claude-opus-4-6")
    master_id = master_model
    master_name = cfg.get("master_name", "Master")
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO sessions
                (id, created_at, student_id, student_name, master_id, master_name,
                 master_model, student_temperature, max_attempts, max_student_questions,
                 classroom_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                created_at,
                sid,
                student_name,
                master_id,
                master_name,
                master_model,
                cfg.get("student_temperature"),
                cfg.get("max_attempts"),
                cfg.get("max_student_questions"),
                classroom_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return session_id


def end_session(session_id: str, summary: Optional[str] = None) -> None:
    """Set ended_at and optionally write a summary for a session."""
    ended_at = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE sessions SET ended_at = ?, summary = ? WHERE id = ?",
            (ended_at, summary, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def log_interaction(session_id: str, data: dict, cfg: dict) -> None:
    """Insert one interaction row."""
    conn = _get_conn()
    try:
        evaluation = data.get("evaluation", {})
        conn.execute(
            """
            INSERT INTO interactions
                (session_id, timestamp, question, student_answer, attempt, verdict,
                 had_errors, quality_scores, reasoning, actions_taken,
                 execution_trace, student_dialogue, master_model, student_temperature)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                data.get("question", ""),
                data.get("student_answer", ""),
                data.get("attempt"),
                evaluation.get("verdict"),
                1 if data.get("had_errors") else 0,
                json.dumps(evaluation.get("quality_scores")),
                evaluation.get("reasoning"),
                json.dumps(evaluation.get("actions", [])),
                json.dumps(data.get("execution_trace", [])),
                json.dumps(data.get("student_dialogue", [])),
                cfg.get("master_model"),
                cfg.get("student_temperature"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def log_journal_entry(session_id: str, entry_type: str, content: str) -> None:
    """Insert a journal entry row."""
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO journal_entries (session_id, timestamp, entry_type, content) VALUES (?, ?, ?, ?)",
            (session_id, datetime.now(timezone.utc).isoformat(), entry_type, content),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_sessions(limit: int = 10) -> list:
    """Return the most recent sessions as a list of dicts."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, created_at, ended_at, student_name, master_name,
                   master_model, summary
            FROM sessions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_sessions() -> list:
    """Return all sessions with interaction counts."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT s.id, s.created_at, s.ended_at, s.student_name, s.master_name,
                   s.master_model, s.student_id, s.master_id, s.summary,
                   s.student_temperature, s.max_attempts, s.max_student_questions,
                   s.classroom_id,
                   COUNT(i.id) as interaction_count
            FROM sessions s
            LEFT JOIN interactions i ON i.session_id = s.id
            GROUP BY s.id
            ORDER BY s.created_at DESC
            """,
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_session_interactions(session_id: str) -> list:
    """Return all interactions for a session, with JSON fields deserialized."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM interactions WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            for field in ("quality_scores", "actions_taken", "execution_trace", "student_dialogue"):
                if d.get(field):
                    try:
                        d[field] = json.loads(d[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            result.append(d)
        return result
    finally:
        conn.close()


def get_recent_interactions(last_n: int = 20) -> list:
    """Return trimmed summary of the last N interactions for master context."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT timestamp, question, student_answer, had_errors, verdict
            FROM interactions
            ORDER BY id DESC
            LIMIT ?
            """,
            (last_n,),
        ).fetchall()
        return [
            {
                "timestamp": r["timestamp"],
                "question": r["question"],
                "student_answer": str(r["student_answer"] or "")[:300],
                "had_errors": bool(r["had_errors"]),
                "verdict": r["verdict"] or "",
            }
            for r in reversed(rows)
        ]
    finally:
        conn.close()


def get_session_journal(session_id: str) -> str:
    """Return full journal entries for a session as a concatenated string."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT timestamp, entry_type, content FROM journal_entries WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        parts = [f"[{r['timestamp']}] [{r['entry_type']}]\n{r['content']}" for r in rows]
        return "\n\n---\n\n".join(parts)
    finally:
        conn.close()


def export_jsonl(path: Path) -> int:
    """Export all interactions as JSONL to path. Returns count of rows exported."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT i.*, s.student_name, s.master_name
            FROM interactions i
            JOIN sessions s ON s.id = i.session_id
            ORDER BY i.id
            """,
        ).fetchall()
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                d = dict(r)
                for field in ("quality_scores", "actions_taken", "execution_trace", "student_dialogue"):
                    if d.get(field):
                        try:
                            d[field] = json.loads(d[field])
                        except (json.JSONDecodeError, TypeError):
                            pass
                f.write(json.dumps(d) + "\n")
                count += 1
        return count
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Classroom management
# ---------------------------------------------------------------------------


def create_classroom(name: str, classroom_id: Optional[str] = None) -> str:
    """Insert a classroom row and return the classroom_id."""
    if classroom_id is None:
        classroom_id = datetime.now(timezone.utc).strftime("classroom_%Y%m%d_%H%M%S")
    created_at = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO classrooms (id, name, created_at, is_active) VALUES (?, ?, ?, 0)",
            (classroom_id, name, created_at),
        )
        conn.commit()
    finally:
        conn.close()
    return classroom_id


def get_classrooms() -> list:
    """Return all classrooms with interaction counts."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT c.id, c.name, c.created_at, c.is_active,
                   COUNT(i.id) as interaction_count
            FROM classrooms c
            LEFT JOIN sessions s ON s.classroom_id = c.id
            LEFT JOIN interactions i ON i.session_id = s.id
            GROUP BY c.id
            ORDER BY c.created_at ASC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def set_active_classroom(classroom_id: str) -> None:
    """Set one classroom as active, deactivating all others."""
    conn = _get_conn()
    try:
        conn.execute("UPDATE classrooms SET is_active = 0")
        conn.execute("UPDATE classrooms SET is_active = 1 WHERE id = ?", (classroom_id,))
        conn.commit()
    finally:
        conn.close()


def get_active_classroom_id() -> Optional[str]:
    """Return the id of the currently active classroom, or None."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT id FROM classrooms WHERE is_active = 1 LIMIT 1"
        ).fetchone()
        return row["id"] if row else None
    finally:
        conn.close()


def delete_classroom(classroom_id: str) -> None:
    """Delete a classroom row by id."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM classrooms WHERE id = ?", (classroom_id,))
        conn.commit()
    finally:
        conn.close()

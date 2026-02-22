# Master-Student LLM Experiment

An autonomous master-student teaching system. A "student" LLM (local model via LM Studio) answers user questions. Before any answer reaches the user, a "master" LLM (Claude Opus 4.6) evaluates it — and if it isn't good enough, takes corrective action: editing the student's system prompt, adding tools, or retrying with hints.

## Quick Start

### Prerequisites
- Python 3.9+
- [LM Studio](https://lmstudio.ai/) running with a model loaded (default: `http://localhost:1234`)
- An Anthropic API key

### Setup

```bash
pip install -r backend/requirements.txt
echo "ANTHROPIC_API_KEY=your_key_here" > .env
echo "LM_STUDIO_URL=http://localhost:1234/v1" >> .env
```

### Run

```bash
# From the master_student/ directory
PYTHONPATH=. python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080` in your browser.

On first start the server automatically migrates any existing `student_config/`, `master_config/`, and `logs/` into `classrooms/default/` and sets it as the active classroom.

---

## Architecture

```
User (browser)
  ↓ question
FastAPI Orchestrator (backend/main.py)
  ↓ question + system_prompt + tools
Student LLM (LM Studio, local)
  ↓ draft answer
Master LLM (Claude Opus 4.6, Anthropic API)
  ↓ PASS / FAIL + actions
Orchestrator
  → PASS: deliver answer to user
  → FAIL: execute actions, retry (max 3)
```

---

## Features

### Student Chat
The main loop. Ask the student anything; the master silently evaluates every response before it reaches you. Each answer shows a "Behind the scenes" panel with quality scores, master reasoning, execution trace, and any master↔student dialogue.

### Teacher's Lounge
A direct chat with the master teacher. Discuss the student's progress, give directives, or ask the master to make changes on the spot. The master has full context (system prompt, tools, journal, recent interactions) injected each turn and can embed `action` blocks directly in its replies to apply changes immediately.

### Classrooms
Multiple isolated workspaces, each with its own student config, master prompt, teacher config, journal, chat history, and logs. Switch between classrooms from the header dropdown, create new ones (optionally copying from the current), or delete non-active ones. Different master/student setups can run in parallel across classrooms.

### Master Prompt Editor
The master's "character" is a single editable prompt per classroom (`master_config/master_prompt.md`), accessible via the collapsible **Master Prompt** section in the right sidebar. The technical JSON/action-block contracts are appended automatically at call time — you only edit the personality and teaching philosophy.

### Teaching Journal
The master logs observations and interventions after each interaction. The journal is visible in the right sidebar and can be compressed + archived at any time via the **Compress & Archive** button, which saves a bullet-point summary to the DB and resets the active journal for the new session.

### Session Database
All sessions, interactions, and journal entries are logged to a local SQLite database (`data/master_student.db`). Click **Open DB** in the header to open it directly in [DB Browser for SQLite](https://sqlitebrowser.org/). Use `GET /db/export` to download all interactions as JSONL for fine-tuning.

---

## Master Actions

| Action | When used |
|--------|-----------|
| `pass_answer` | Answer is good enough — deliver to user |
| `retry_with_hint` | One-off mistake, no permanent change needed |
| `edit_system_prompt` | Pattern of errors better instructions would fix |
| `add_tool` | Student lacks a needed capability |
| `remove_tool` | A tool is causing problems |
| `ask_student` | Probe the student's reasoning mid-evaluation (up to 3×) |
| `edit_code` | The system itself needs a fix |
| `fail_with_explanation` | Max retries exhausted — master provides fallback answer |

---

## API Reference

### Chat
| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send a message through the student→master loop |
| WS | `/chat-stream` | WebSocket for real-time status updates |

### Student Config
| Method | Path | Description |
|--------|------|-------------|
| GET | `/system-prompt` | Get active classroom's student system prompt |
| PUT | `/system-prompt` | Update student system prompt |
| GET | `/tools` | Get tool definitions |

### Master Config
| Method | Path | Description |
|--------|------|-------------|
| GET | `/master-prompt` | Get active classroom's master prompt |
| PUT | `/master-prompt` | Update master prompt |
| GET | `/teacher-config` | Get teacher config (models, temperature, etc.) |
| PUT | `/teacher-config` | Update teacher config |

### Teacher's Lounge
| Method | Path | Description |
|--------|------|-------------|
| POST | `/master-chat` | Send a message to the master directly |
| GET | `/master-chat/history` | Get conversation history |
| POST | `/master-chat/reset` | Clear Teacher's Lounge conversation |

### Session Management
| Method | Path | Description |
|--------|------|-------------|
| GET | `/journal` | Get current teaching journal |
| POST | `/compress-journal` | Summarise + archive current journal, start new session |
| GET | `/logs` | Get recent conversation logs |
| POST | `/new-chat` | Archive journal, clear conversation, keep student config |
| POST | `/clear-history` | Wipe logs and journal without touching config |
| POST | `/reset` | Full reset: compress journal, restore initial state |

### Classrooms
| Method | Path | Description |
|--------|------|-------------|
| GET | `/classrooms` | List all classrooms with active id |
| POST | `/classrooms` | Create a new classroom (`name`, optional `copy_from`) |
| POST | `/classrooms/{id}/activate` | Switch active classroom |
| DELETE | `/classrooms/{id}` | Delete a non-active classroom |

### Database
| Method | Path | Description |
|--------|------|-------------|
| GET | `/db/sessions` | List all sessions with metadata |
| GET | `/db/sessions/{id}/interactions` | Get interactions for a session |
| GET | `/db/export` | Download all interactions as JSONL |
| POST | `/open-db` | Open `data/master_student.db` in the system viewer |

---

## File Structure

```
master_student/
├── backend/
│   ├── main.py           # FastAPI orchestrator — path globals, migration, all endpoints
│   ├── master.py         # Claude Opus 4.6 — evaluate(), direct_chat(), compress_journal()
│   ├── student.py        # LM Studio client — ask_student(), ask_student_direct()
│   ├── tool_executor.py  # Dynamic Python tool loader — execute_tool(), add_tool(), remove_tool()
│   ├── db.py             # SQLite logging — sessions, interactions, journal_entries, classrooms
│   └── requirements.txt
├── classrooms/           # Created automatically on first run (gitignored)
│   └── {id}/
│       ├── student_config/
│       │   ├── system_prompt.md
│       │   ├── tools.json
│       │   └── tools/          # Tool implementations (.py with run(**kwargs))
│       └── master_config/
│           ├── master_prompt.md
│           ├── teacher_config.json
│           ├── teaching_journal.md
│           └── master_chat_history.json
├── data/
│   └── master_student.db       # SQLite (gitignored)
└── frontend/
    └── index.html              # Single-file vanilla JS UI
```

---

## Adding Custom Tools

Tools are Python files with a `run(**kwargs)` function. The master can create them automatically, or you can add them manually:

```python
# classrooms/default/student_config/tools/my_tool.py
def run(query: str = "") -> str:
    return f"Result for: {query}"
```

Register it in `tools.json`:
```json
[{
  "type": "function",
  "function": {
    "name": "my_tool",
    "description": "What this tool does",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "The query"}
      },
      "required": ["query"]
    }
  }
}]
```

Or just tell the master in the Teacher's Lounge: *"Add a tool that can do X"* and it will write and register the tool itself.

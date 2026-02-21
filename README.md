# Master-Student LLM Experiment

An autonomous master-student teaching system. A "student" LLM (local model via LM Studio) answers user questions. Before any answer reaches the user, a "master" LLM (Claude Opus 4.6) evaluates it — and if it isn't good enough, takes corrective action: editing the student's system prompt, adding tools, or retrying with hints.

## Quick Start

### Prerequisites
- Python 3.9+
- [LM Studio](https://lmstudio.ai/) running with a model loaded (default: `http://localhost:1234`)
- An Anthropic API key

### Setup

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Set your API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
echo "LM_STUDIO_URL=http://localhost:1234/v1" >> .env
```

### Run

```bash
# From the master_student/ directory
PYTHONPATH=. python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080` in your browser.

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
  → PASS: deliver answer
  → FAIL: execute actions, retry (max 3)
```

## Master Actions

| Action | When used |
|--------|-----------|
| `pass_answer` | Answer is good enough |
| `retry_with_hint` | One-off mistake, no permanent change |
| `edit_system_prompt` | Pattern of errors better instructions would fix |
| `add_tool` | Student lacks a needed capability |
| `remove_tool` | A tool is causing problems |
| `edit_code` | The system itself needs a fix |
| `fail_with_explanation` | Max retries exhausted |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send a message, get back the evaluated answer |
| GET | `/system-prompt` | Get current student system prompt |
| PUT | `/system-prompt` | Manually update system prompt |
| GET | `/tools` | Get current tool definitions |
| GET | `/journal` | Get teaching journal |
| GET | `/logs` | Get recent conversation logs |
| POST | `/reset` | Restore initial state |
| WS | `/chat-stream` | WebSocket for real-time status updates |

## File Structure

```
master_student/
├── backend/
│   ├── main.py           # FastAPI orchestrator
│   ├── student.py        # LM Studio client
│   ├── master.py         # Claude Opus 4.6 evaluator
│   ├── tool_executor.py  # Dynamic Python tool loader
│   └── requirements.txt
├── student_config/
│   ├── system_prompt.md  # Editable by master
│   ├── tools.json        # Tool definitions
│   └── tools/            # Tool implementations (.py files with run())
├── master_config/
│   ├── evaluation_rubric.md
│   └── teaching_journal.md
├── frontend/
│   └── index.html        # Single-file UI
└── logs/
    └── conversation_log.jsonl
```

## Adding Custom Tools

Tools are Python files in `student_config/tools/` with a `run(**kwargs)` function:

```python
# student_config/tools/my_tool.py
def run(query: str = "") -> str:
    return f"Result for: {query}"
```

Add the tool definition to `student_config/tools.json`:
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

Or let the master add tools automatically when it decides the student needs them.

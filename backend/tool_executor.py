"""Dynamic tool executor — loads and runs tools from student_config/tools/."""

import importlib.util
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
TOOLS_DIR = BASE_DIR / "student_config" / "tools"
TOOLS_JSON = BASE_DIR / "student_config" / "tools.json"


def execute_tool(tool_name: str, arguments: dict) -> str:
    """
    Dynamically load and execute a tool from student_config/tools/.

    Raises specific exceptions so the caller can log structured error info:
        FileNotFoundError  — tool file doesn't exist
        SyntaxError        — tool code has a syntax error
        ImportError        — tool requires a missing package
        AttributeError     — tool file has no run() function
        Exception          — any runtime error during execution
    """
    tool_path = TOOLS_DIR / f"{tool_name}.py"

    if not tool_path.exists():
        raise FileNotFoundError(f"Tool '{tool_name}' not found at {tool_path}")

    # Load the module — SyntaxError and ImportError surface here
    try:
        spec = importlib.util.spec_from_file_location(tool_name, str(tool_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except SyntaxError as e:
        raise SyntaxError(f"Tool '{tool_name}' has a syntax error: {e}") from e
    except ImportError as e:
        raise ImportError(f"Tool '{tool_name}' requires a missing package: {e}") from e

    if not hasattr(module, "run"):
        raise AttributeError(f"Tool '{tool_name}' has no run() function")

    result = module.run(**arguments)
    return json.dumps(result) if not isinstance(result, str) else result


def add_tool(name: str, description: str, parameters: dict, implementation: str) -> None:
    """Add a new tool: write the Python file and update tools.json."""
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    tool_path = TOOLS_DIR / f"{name}.py"
    tool_path.write_text(implementation, encoding="utf-8")

    tools = _read_tools()
    tools = [t for t in tools if t.get("function", {}).get("name") != name]
    tools.append(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
    )
    TOOLS_JSON.write_text(json.dumps(tools, indent=2), encoding="utf-8")


def remove_tool(name: str) -> None:
    """Remove a tool by name: delete the Python file and update tools.json."""
    tool_path = TOOLS_DIR / f"{name}.py"
    if tool_path.exists():
        tool_path.unlink()

    tools = _read_tools()
    tools = [t for t in tools if t.get("function", {}).get("name") != name]
    TOOLS_JSON.write_text(json.dumps(tools, indent=2), encoding="utf-8")


def _read_tools() -> list:
    content = TOOLS_JSON.read_text(encoding="utf-8").strip() if TOOLS_JSON.exists() else "[]"
    return json.loads(content) if content else []


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing tool executor with example_tool...")
    try:
        result = execute_tool("example_tool", {"input": "hello world"})
        print(f"Result: {result}")
    except FileNotFoundError as e:
        print(f"Note: {e}")

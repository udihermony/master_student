"""Example tool — echoes the input back in uppercase.

Every tool must have a run(**kwargs) function that returns a str or JSON-serializable value.
"""


def run(input: str = "") -> str:
    """Echo input back in uppercase."""
    return input.upper()

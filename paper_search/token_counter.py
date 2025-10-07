import threading
from typing import Any

# Thread-safe global token counter
_total_tokens = 0
_lock = threading.Lock()


def add_tokens(n: int) -> None:
    global _total_tokens
    try:
        n_int = int(n) if n is not None else 0
    except Exception:
        return
    if n_int < 0:
        return
    with _lock:
        _total_tokens += n_int


def add_usage(usage: Any) -> None:
    """
    Extract total token count from an OpenAI-like response.usage object and add it.
    Supports both attribute-style and dict-style access.
    """
    if usage is None:
        return
    total = None
    try:
        # Try attribute access first
        total = getattr(usage, "total_tokens", None)
    except Exception:
        total = None
    if total is None and isinstance(usage, dict):
        total = usage.get("total_tokens")
    if total is not None:
        add_tokens(total)


def get_total() -> int:
    with _lock:
        return _total_tokens


def write_total_to_results(path: str = r"token_cost.txt") -> None:
    total = get_total()
    try:
        with open(path, "a+", encoding="utf-8") as f:
            f.write(f"\nLLM total tokens: {total}\n")
    except Exception:
        # Swallow I/O errors to avoid breaking main logic
        pass
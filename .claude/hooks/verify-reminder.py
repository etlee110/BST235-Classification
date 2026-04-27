#!/usr/bin/env python3
"""
Verification Reminder Hook

Non-blocking reminder that fires on Write/Edit to Python/notebook files
to remind about running the code before marking a task complete.

Hook Event: PostToolUse (matcher: "Write|Edit")
Returns: Exit code 0 (non-blocking, reminder visible but doesn't stop work)

Skips:
- Configuration files (.json, .yaml, .toml, etc.)
- Documentation files (.md, .txt, README)
- Test files prefixed with test_
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
NC = "\033[0m"

VERIFY_EXTENSIONS = {
    ".py": "run a forward pass or quick smoke-test to verify",
    ".ipynb": "re-run affected cells to verify output",
}

SKIP_EXTENSIONS = [
    ".md", ".txt", ".rst",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".lock", ".env", ".gitignore",
    ".png", ".jpg", ".svg", ".pdf",
    ".pth", ".tar", ".npy", ".npz",
]

SKIP_DIRS = [
    "/output/", "/quality_reports/", "/.claude/",
    "/templates/", "/node_modules/", "/__pycache__/",
]


def get_session_dir() -> Path:
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", "")
    if not project_dir:
        return Path.home() / ".claude" / "sessions" / "default"
    import hashlib
    project_hash = hashlib.md5(project_dir.encode()).hexdigest()[:8]
    session_dir = Path.home() / ".claude" / "sessions" / project_hash
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def should_skip(file_path: str) -> bool:
    path = Path(file_path)
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return True
    for skip_dir in SKIP_DIRS:
        if skip_dir in file_path:
            return True
    name = path.name.lower()
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    return False


def needs_verification(file_path: str) -> tuple[bool, str]:
    suffix = Path(file_path).suffix.lower()
    if suffix in VERIFY_EXTENSIONS:
        return True, VERIFY_EXTENSIONS[suffix]
    return False, ""


def was_recently_reminded(file_path: str) -> bool:
    cache_file = get_session_dir() / "verify-reminder-cache.json"
    try:
        cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    except (json.JSONDecodeError, IOError):
        cache = {}

    last_reminder = cache.get(file_path, 0)
    now = time.time()
    cache[file_path] = now
    cache = {k: v for k, v in cache.items() if now - v < 300}

    try:
        cache_file.write_text(json.dumps(cache))
    except IOError:
        pass

    return (now - last_reminder) < 300


def format_reminder(file_path: str, action: str) -> str:
    filename = Path(file_path).name
    return f"\n{CYAN}📋 Verify:{NC} {filename}\n   → {GREEN}{action}{NC} before marking task complete\n"


def main() -> int:
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, IOError):
        return 0

    file_path = hook_input.get("tool_input", {}).get("file_path", "")
    if not file_path or should_skip(file_path):
        return 0

    needs_verify, action = needs_verification(file_path)
    if not needs_verify or was_recently_reminded(file_path):
        return 0

    print(format_reminder(file_path, action))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)

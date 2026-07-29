from __future__ import annotations

import re
import stat
from pathlib import Path

EMAIL_RE = re.compile(r"(?<![\w.-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
SECRET_RE = re.compile(
    r"(?i)(?:sk-or-v1-[A-Za-z0-9_-]{20,}|(?:api[_-]?key|token|secret|password)\s*[:=]\s*[\"']?[^\s\"']{8,})"
)
ABS_PATH_RE = re.compile(r"(?<![\w])(?:/Users|/home|/workspace|[A-Za-z]:\\\\)[^\s\"'`]+")


def redact_text(text: str) -> str:
    value = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    value = SECRET_RE.sub("[REDACTED_SECRET]", value)
    value = ABS_PATH_RE.sub("[REDACTED_ABSOLUTE_PATH]", value)
    return value


def secure_key_file(path: Path, *, fix: bool = False) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"OpenRouter key file not found: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        if fix:
            path.chmod(0o600)
        else:
            raise PermissionError(
                f"OpenRouter key file must not be group/world accessible: {path} mode={mode:o}"
            )


def read_api_key(path: Path) -> str:
    secure_key_file(path)
    key = path.read_text(encoding="utf-8").strip()
    if len(key) < 20:
        raise ValueError("OpenRouter key file does not contain a plausible key")
    return key


def contains_sensitive_text(text: str) -> bool:
    return bool(EMAIL_RE.search(text) or SECRET_RE.search(text) or ABS_PATH_RE.search(text))


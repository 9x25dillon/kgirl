"""Keep credentials out of the chunk index, which stores file text and can be copied to the router.

`secret_reason(path, text)` names the first rule a file trips, or returns None. A hit drops the whole
file. Callers report only `repo:path` and the rule name, never the matched text.

The walker indexes only code, shell and Markdown, so most hits come from the token rules: a key pasted
into a script, a README or a test. The name rules are a backstop in case the indexed types grow.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

_PLACEHOLDER = re.compile(r"your|example|placeholder|changeme|dummy|redacted|xxxx|\*\*\*|<|\$\{|\{\{", re.I)

_NAME_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ssh private key", re.compile(r"^id_(rsa|dsa|ecdsa|ed25519)$")),
    ("key file", re.compile(r"\.(pem|key|p12|pfx|jks|keystore|ppk|kdbx)$", re.I)),
    ("credentials file", re.compile(r"^(credentials|secrets?|service[-_]account[^/]*)\.(json|ya?ml|toml|ini|txt)$",
                                    re.I)),
    ("auth config", re.compile(r"^(\.npmrc|\.pypirc|\.netrc|\.htpasswd|\.git-credentials|\.dockercfg)$")),
    ("wallet file", re.compile(r"(wallet[^/]*\.(json|dat)|^UTC--.+)$", re.I)),
)

_TOKEN_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private key block", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})")),
    ("GitLab token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}")),
    ("Slack token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("Anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("OpenAI key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}")),
    ("Stripe live key", re.compile(r"\b[rs]k_live_[0-9A-Za-z]{20,}")),
)

# The secret is group 1; a placeholder value ("your-password", "${TOKEN}") doesn't count. Unquoted values must
# also mix letters and digits, so identifiers such as `EKeys::Gamepad_FaceButton_Bottom` pass.
_VALUE_RULES: tuple[tuple[str, re.Pattern[str], bool], ...] = (
    ("credentials in URL", re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/:@'\"]+:([^\s/@'\"]{8,})@"), False),
    ("hard-coded secret", re.compile(
        r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd)"
        r"\b['\"]?\s*[:=]\s*['\"]([A-Za-z0-9/+_.=-]{16,})['\"]"), False),
    ("unquoted secret", re.compile(
        r"(?i)(?:api[_-]?key|secret|token|password|passwd|\.key)\s*[:=]\s*([A-Za-z0-9/+_.=-]{32,})"), True),
)
_LETTER, _DIGIT = re.compile(r"[A-Za-z]"), re.compile(r"[0-9]")


def secret_reason(path: str, text: str) -> str | None:
    name = PurePosixPath(path).name
    for rule, pattern in _NAME_RULES:
        if pattern.search(name):
            return rule
    for rule, pattern in _TOKEN_RULES:
        if pattern.search(text):
            return rule
    for rule, pattern, mixed in _VALUE_RULES:
        for m in pattern.finditer(text):
            value = m.group(1)
            if _PLACEHOLDER.search(value) or (mixed and not (_LETTER.search(value) and _DIGIT.search(value))):
                continue
            return rule
    return None

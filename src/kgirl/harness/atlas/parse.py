"""Structural extraction: source text -> symbols, imports, references.

The Atlas stores architecture, not text. For every file we keep:

* symbols  - modules/classes/functions/methods with signature, one-paragraph
             doc, line span and a normalized body hash (clone detection)
* imports  - raw import targets (resolved later, across repositories)
* refs     - identifiers the file calls or references (symbol-level blast radius)

Python goes through `ast` (exact). Other languages use line-anchored regexes:
approximate, but deterministic and dependency-free. A Python file that does not
parse falls back to the regex extractor, so broken files still show up.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from ..util import first_paragraph, sha1

PARSER_VERSION = "1"  # bump when extraction changes; forces re-parse of unchanged files

LANG_BY_EXT = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript",
    ".jl": "julia", ".go": "go", ".rs": "rust",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp",
    ".java": "java", ".kt": "kotlin", ".swift": "swift", ".zig": "zig",
    ".sh": "shell", ".bash": "shell",
    ".md": "markdown", ".gd": "gdscript",
}


@dataclass(frozen=True)
class Symbol:
    kind: str            # module | class | function | method | struct | trait | section ...
    name: str
    qualname: str
    signature: str
    doc: str
    line: int
    end_line: int
    body_hash: str = ""  # normalized body hash; "" when too small to be meaningful


@dataclass(frozen=True)
class Import:
    target: str          # "pkg.mod", "./util", "Base.Threads", "file.jl"
    names: tuple[str, ...]
    level: int           # python relative level; 0 = absolute
    line: int
    kind: str = "import"  # import | include


@dataclass
class FileFacts:
    lang: str
    loc: int
    doc: str = ""
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[Import] = field(default_factory=list)
    refs: set[str] = field(default_factory=set)
    is_entrypoint: bool = False
    parse_error: str = ""


def detect_language(path: str, head: str = "") -> str | None:
    ext = PurePosixPath(path).suffix.lower()
    if ext in LANG_BY_EXT:
        return LANG_BY_EXT[ext]
    if not ext and head.startswith("#!"):
        first = head.split("\n", 1)[0]
        if "python" in first:
            return "python"
        if re.search(r"\b(ba)?sh\b", first):
            return "shell"
        if "node" in first:
            return "javascript"
    return None


def extract(path: str, text: str, lang: str) -> FileFacts:
    loc = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    if lang == "python":
        try:
            return _python(text, loc)
        except (SyntaxError, ValueError, RecursionError) as exc:
            facts = _regex(text, loc, "python")
            facts.parse_error = f"{type(exc).__name__}: {exc}"[:200]
            return facts
    if lang == "markdown":
        return _markdown(text, loc)
    return _regex(text, loc, lang)


# --------------------------------------------------------------------------- python


def _normalized_hash(node: ast.AST) -> str:
    """Hash a function/class body ignoring names of locals, docstrings, formatting.

    Two copies of the same function pasted into different repos hash equal even
    when reformatted, which is what cross-repo clone detection needs.
    """
    body = getattr(node, "body", [])
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if sum(1 for _ in ast.walk(ast.Module(body=body, type_ignores=[]))) < 12:
        return ""
    dumped = "\n".join(ast.dump(stmt, annotate_fields=False, include_attributes=False) for stmt in body)
    return sha1(dumped)[:16]


def _sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        args = ast.unparse(node.args)
    except Exception:  # pragma: no cover - unparse is total on parsed trees
        args = "..."
    ret = ""
    if node.returns is not None:
        try:
            ret = " -> " + ast.unparse(node.returns)
        except Exception:  # pragma: no cover
            ret = ""
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    return f"{prefix}{node.name}({args}){ret}"


def _python(text: str, loc: int) -> FileFacts:
    tree = ast.parse(text)
    facts = FileFacts(lang="python", loc=loc, doc=first_paragraph(ast.get_docstring(tree)))

    def visit(body, prefix: str, in_class: bool):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = f"{prefix}{node.name}"
                facts.symbols.append(Symbol(
                    "method" if in_class else "function", node.name, q, _sig(node),
                    first_paragraph(ast.get_docstring(node)), node.lineno,
                    getattr(node, "end_lineno", node.lineno) or node.lineno, _normalized_hash(node)))
                visit(node.body, q + ".", False)
            elif isinstance(node, ast.ClassDef):
                q = f"{prefix}{node.name}"
                bases = ", ".join(_safe_unparse(b) for b in node.bases)
                facts.symbols.append(Symbol(
                    "class", node.name, q, f"class {node.name}({bases})" if bases else f"class {node.name}",
                    first_paragraph(ast.get_docstring(node)), node.lineno,
                    getattr(node, "end_lineno", node.lineno) or node.lineno, _normalized_hash(node)))
                visit(node.body, q + ".", True)
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)) and not prefix:
                # conditional top-level definitions (try: import X except: def fallback)
                for sub in ("body", "orelse", "finalbody", "handlers"):
                    chunk = getattr(node, sub, None)
                    if chunk:
                        visit([n for h in chunk for n in (h.body if isinstance(h, ast.ExceptHandler) else [h])],
                              prefix, in_class)

    visit(tree.body, "", False)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                facts.imports.append(Import(alias.name, (), 0, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            facts.imports.append(Import(node.module or "", tuple(a.name for a in node.names),
                                        node.level or 0, node.lineno))
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                facts.refs.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                facts.refs.add(fn.attr)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            facts.refs.add(node.id)
        elif isinstance(node, ast.If) and _is_main_guard(node.test):
            facts.is_entrypoint = True
    return facts


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover
        return "?"


def _is_main_guard(test: ast.AST) -> bool:
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id == "__name__" and any(
                isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators))


# --------------------------------------------------------------------------- regex languages

_DEF_PATTERNS: dict[str, list[tuple[str, re.Pattern]]] = {
    "python": [
        ("class", re.compile(r"^(\s*)class\s+([A-Za-z_]\w*)\s*(\([^)]*\))?\s*:")),
        ("function", re.compile(r"^(\s*)(?:async\s+)?def\s+([A-Za-z_]\w*)\s*(\(.*)")),
    ],
    "javascript": [
        ("class", re.compile(r"^(\s*)(?:export\s+(?:default\s+)?)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)(.*)")),
        ("function", re.compile(r"^(\s*)(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*(\(.*)")),
        ("function", re.compile(r"^(\s*)(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s+)?(\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>")),
        ("interface", re.compile(r"^(\s*)(?:export\s+)?(?:interface|type|enum)\s+([A-Za-z_$][\w$]*)(.*)")),
    ],
    "julia": [
        ("module", re.compile(r"^(\s*)(?:bare)?module\s+([A-Za-z_]\w*)()")),
        ("struct", re.compile(r"^(\s*)(?:mutable\s+)?struct\s+([A-Za-z_]\w*)(.*)")),
        ("function", re.compile(r"^(\s*)function\s+([A-Za-z_][\w!.]*)\s*(\(.*)")),
        ("function", re.compile(r"^()([A-Za-z_][\w!]*)\s*(\([^=]*\))\s*=(?!=)")),
    ],
    "go": [
        ("function", re.compile(r"^()func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*(\(.*)")),
        ("struct", re.compile(r"^()type\s+([A-Za-z_]\w*)\s+(struct|interface)")),
    ],
    "rust": [
        ("function", re.compile(r"^(\s*)(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_]\w*)\s*(.*)")),
        ("struct", re.compile(r"^(\s*)(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|union)\s+([A-Za-z_]\w*)(.*)")),
        ("impl", re.compile(r"^(\s*)impl(?:<[^>]*>)?\s+([A-Za-z_][\w:]*)(.*)")),
    ],
    "c": [("function", re.compile(r"^()(?:[A-Za-z_][\w\s\*]*?)\b([A-Za-z_]\w*)\s*(\([^;{]*\))\s*\{?\s*$"))],
    "cpp": [
        ("class", re.compile(r"^(\s*)(?:class|struct)\s+([A-Za-z_]\w*)(.*)")),
        ("function", re.compile(r"^()(?:[A-Za-z_][\w\s\*&:<>]*?)\b([A-Za-z_][\w:]*)\s*(\([^;{]*\))\s*(?:const)?\s*\{?\s*$")),
    ],
    "java": [
        ("class", re.compile(r"^(\s*)(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?(?:class|interface|enum|record)\s+([A-Za-z_]\w*)(.*)")),
        ("function", re.compile(r"^(\s+)(?:public|private|protected|static|final|\s)+[\w<>\[\],\s]+\s+([a-z_]\w*)\s*(\([^;]*\))\s*\{?")),
    ],
    "kotlin": [
        ("class", re.compile(r"^(\s*)(?:data\s+|sealed\s+|open\s+)?(?:class|interface|object)\s+([A-Za-z_]\w*)(.*)")),
        ("function", re.compile(r"^(\s*)(?:\w+\s+)*fun\s+(?:<[^>]*>\s*)?([A-Za-z_][\w.]*)\s*(\(.*)")),
    ],
    "swift": [
        ("class", re.compile(r"^(\s*)(?:\w+\s+)*(?:class|struct|protocol|enum|actor)\s+([A-Za-z_]\w*)(.*)")),
        ("function", re.compile(r"^(\s*)(?:\w+\s+)*func\s+([A-Za-z_]\w*)\s*(.*)")),
    ],
    "zig": [("function", re.compile(r"^(\s*)(?:pub\s+)?fn\s+([A-Za-z_]\w*)\s*(\(.*)"))],
    "gdscript": [
        ("class", re.compile(r"^()class_name\s+([A-Za-z_]\w*)()")),
        ("function", re.compile(r"^(\s*)func\s+([A-Za-z_]\w*)\s*(\(.*)")),
    ],
    "shell": [("function", re.compile(r"^()(?:function\s+)?([A-Za-z_][\w-]*)\s*\(\)\s*\{?"))],
}

_IMPORT_PATTERNS: dict[str, list[re.Pattern]] = {
    "python": [re.compile(r"^\s*from\s+(\.*[\w.]*)\s+import\s+(.+)"), re.compile(r"^\s*import\s+([\w.]+)")],
    "javascript": [re.compile(r"""^\s*import\s+(?:[^'"]*?\s+from\s+)?['"]([^'"]+)['"]"""),
                   re.compile(r"""^\s*export\s+[^'"]*?\s+from\s+['"]([^'"]+)['"]"""),
                   re.compile(r"""require\(\s*['"]([^'"]+)['"]\s*\)""")],
    "julia": [re.compile(r"^\s*(?:using|import)\s+([\w.]+(?:\s*,\s*[\w.]+)*)"),
              re.compile(r"""^\s*include\(\s*["']([^"']+)["']\s*\)""")],
    "go": [re.compile(r'^\s*(?:import\s+)?(?:\w+\s+)?"([\w./-]+)"\s*$')],
    "rust": [re.compile(r"^\s*(?:pub\s+)?use\s+([\w:]+)"), re.compile(r"^\s*(?:pub\s+)?mod\s+(\w+)\s*;")],
    "c": [re.compile(r'^\s*#\s*include\s+[<"]([^>"]+)[>"]')],
    "cpp": [re.compile(r'^\s*#\s*include\s+[<"]([^>"]+)[>"]')],
    "java": [re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)")],
    "kotlin": [re.compile(r"^\s*import\s+([\w.]+)")],
    "swift": [re.compile(r"^\s*import\s+(\w+)")],
    "zig": [re.compile(r"""@import\(\s*"([^"]+)"\s*\)""")],
    "gdscript": [re.compile(r"""(?:preload|load)\(\s*"([^"]+)"\s*\)""")],
    "shell": [re.compile(r"^\s*(?:source|\.)\s+([\w./-]+)")],
}

_FAMILY = {"typescript": "javascript"}

_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_KEYWORDS = frozenset("if for while switch return function catch elif def class print super self new "
                      "sizeof typeof await yield match with and or not in is lambda fn func end".split())


def _regex(text: str, loc: int, lang: str) -> FileFacts:
    facts = FileFacts(lang=lang, loc=loc)
    lines = text.split("\n")
    family = _FAMILY.get(lang, lang)
    defs = _DEF_PATTERNS.get(family, [])
    imps = _IMPORT_PATTERNS.get(family, [])
    stack: list[tuple[int, str]] = []  # (indent, qualname) for python-ish nesting
    for i, line in enumerate(lines, 1):
        if len(line) > 2000:
            continue
        for kind, pat in defs:
            m = pat.match(line)
            if not m:
                continue
            indent, name = len(m.group(1)), m.group(2)
            if name in _KEYWORDS:
                break
            tail = (m.group(3) or "").strip() if m.lastindex and m.lastindex >= 3 else ""
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1] + "." if stack and lang == "python" else ""
            k = "method" if kind == "function" and parent and lang == "python" else kind
            sig = f"{line.strip()[:160]}" if not tail else f"{name}{tail[:140]}"
            facts.symbols.append(Symbol(k, name, parent + name, sig, _leading_comment(lines, i - 1), i, i))
            if lang == "python":
                stack.append((indent, parent + name))
            break
        for pat in imps:
            for m in pat.finditer(line):
                raw = m.group(1).strip()
                if lang == "python" and pat.pattern.startswith("^\\s*from"):
                    level = len(raw) - len(raw.lstrip("."))
                    names = tuple(n.strip().split(" as ")[0] for n in m.group(2).strip("() \\").split(",") if n.strip())
                    facts.imports.append(Import(raw.lstrip("."), names, level, i))
                elif lang == "julia" and "include" in pat.pattern:
                    facts.imports.append(Import(raw, (), 0, i, "include"))
                elif lang == "julia":
                    for mod in raw.split(","):
                        facts.imports.append(Import(mod.strip(), (), 0, i))
                else:
                    facts.imports.append(Import(raw, (), 0, i))
        for m in _CALL.finditer(line):
            if m.group(1) not in _KEYWORDS:
                facts.refs.add(m.group(1))
    if lang == "python":
        facts.is_entrypoint = "__main__" in text
    return facts


def _leading_comment(lines: list[str], idx: int) -> str:
    """Collect `//`, `#` or `///` comment lines right above a definition."""
    out: list[str] = []
    j = idx - 1
    while j >= 0 and len(out) < 6:
        s = lines[j].strip()
        if s.startswith(("//", "#", "--", "*", "/*")) and not s.startswith(("#!", "#include", "#[")):
            out.append(s.lstrip("/#-* ").strip())
            j -= 1
        else:
            break
    return first_paragraph(" ".join(reversed(out)), 200)


# --------------------------------------------------------------------------- markdown

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")


def _markdown(text: str, loc: int) -> FileFacts:
    facts = FileFacts(lang="markdown", loc=loc)
    lines = text.split("\n")
    in_code = False
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        m = None if in_code else _HEADING.match(line)
        if not m:
            continue
        body: list[str] = []
        for nxt in lines[i:i + 12]:
            if _HEADING.match(nxt) or nxt.lstrip().startswith("```"):
                break
            body.append(nxt)
        title = m.group(2).strip()
        facts.symbols.append(Symbol("section", title[:120], title[:120], "#" * len(m.group(1)) + " " + title[:120],
                                    first_paragraph("\n".join(body).strip(), 240), i, i))
    if facts.symbols:
        facts.doc = facts.symbols[0].doc or facts.symbols[0].name
    return facts

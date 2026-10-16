#!/usr/bin/env python3
"""PreToolUse guard for the TEMPER subagent (.claude/agents/temper.md). Standard library only.

TEMPER may read anything and write only under <project>/runs/temper/. While TEMPER runs, this hook sees
every tool call (matcher "*") and refuses what falls outside that:

    Read, Grep, Glob, LS    pass
    Write                   only to a path whose real location is inside runs/temper/
    Bash                    parsed with shlex, segment by segment; refused when a segment
                              - changes git state (only read-only git subcommands pass)
                              - reaches the network, installs software or escalates (curl, ssh, gh, pip, sudo, ...)
                              - runs inline code: python -c, bash -c, heredocs, $(...), backticks, awk, sed, eval
                              - writes outside runs/temper/ through >, >>, tee, cp, mv, rm, mkdir, touch, ln,
                                or an --out / --output / --export / --latex / --plot style argument
                              - runs a kgirl bench without an explicit output path under runs/temper/
                              - changes harness state (index, task, soup add/curate, skills forge) outside an
                                isolated home under runs/temper/, or promotes memory, or serves MCP
    anything else           refused (fail closed)

An error inside the guard refuses the call too, and the agent file sets `onFailure: block`, so a crash, a
timeout or a missing interpreter blocks instead of letting the call through.

Limits, stated rather than hidden: this stops mistakes and drift; it is not a sandbox. A program TEMPER is
allowed to run (a bench, or a script it wrote under runs/temper/scripts/) can still do whatever its own code
does. Nothing TEMPER proposes lands until the user accepts it.

Protocol (Claude Code hooks): JSON on stdin with tool_name, tool_input and cwd. To refuse, print a
PreToolUse `permissionDecision: deny` object and exit 2 (stderr carries the reason). To pass, exit 0 with no
output: that is "no decision", so the normal permission flow still applies. This hook never approves.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys

ALLOWED_DIR = ("runs", "temper")
READ_ONLY_TOOLS = frozenset({"Read", "Grep", "Glob", "LS"})
DEVICES = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})

NETWORK_INSTALL_ESCALATE = frozenset({
    "curl", "wget", "nc", "ncat", "netcat", "socat", "ssh", "scp", "sftp", "rsync", "ftp", "telnet",
    "gh", "hub", "glab", "aws", "gcloud", "az", "docker", "podman", "kubectl",
    "pip", "pip3", "pipx", "uv", "poetry", "conda", "mamba", "npm", "npx", "pnpm", "yarn", "bun",
    "cargo", "go", "apt", "apt-get", "dnf", "yum", "apk", "brew", "snap", "sudo", "su", "doas",
    "crontab", "at", "systemctl", "service",
})
INLINE_ONLY = frozenset({"awk", "gawk", "mawk", "nawk", "sed", "perl", "ruby", "lua", "php", "tclsh", "expect",
                         "eval", "source", ".", "xargs", "pushd", "popd"})
FILE_REWRITERS = frozenset({"dd", "patch", "tar", "zip", "unzip", "gzip", "gunzip", "bzip2", "xz", "split",
                            "csplit", "shred"})
SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish"})
EVAL_FLAGS = {"node": {"-e", "--eval", "-p", "--print"}, "julia": {"-e", "--eval", "-E", "--print"},
              "Rscript": {"-e"}}
WRAPPERS = frozenset({"time", "timeout", "nice", "nohup", "stdbuf", "env", "command", "exec"})
WRITE_ALL_ARGS = frozenset({"rm", "rmdir", "mkdir", "touch", "unlink", "truncate", "tee", "mv"})
WRITE_SKIP_FIRST = frozenset({"chmod", "chown", "chgrp"})          # first operand is a mode or an owner
WRITE_DEST = frozenset({"cp", "ln", "install"})                    # the last operand (or -t DIR) is written
OUT_FLAGS = ("--out", "--output", "--out-dir", "--output-dir", "--outdir", "--export", "--latex", "--plot",
             "--plots", "--fig", "--figure", "--figures", "--json-out", "--save", "--log-file", "--report")
PYTHON_DENY_MODULES = frozenset({"pip", "ensurepip", "http.server", "smtpd", "ftplib"})

GIT_READ_ONLY = frozenset({"status", "log", "show", "diff", "rev-parse", "ls-files", "ls-tree", "cat-file",
                           "describe", "blame", "grep", "shortlog", "merge-base", "rev-list", "name-rev",
                           "show-ref", "for-each-ref", "count-objects", "version"})
GIT_BRANCH_READ = frozenset({"--show-current", "--list", "-l", "-a", "--all", "-r", "--remotes", "-v", "-vv",
                             "--verbose", "--no-color"})
GIT_OPTS_WITH_VALUE = frozenset({"-C", "--git-dir", "--work-tree", "--namespace"})
OLLAMA_READ = frozenset({"list", "ps", "show", "run"})

# kgirl benches: (module, subcommand) -> output flags; one must be given, under runs/temper/
REQUIRE_OUT = {("kgirl.neutronics", "fit"): ("--out",), ("kgirl.iwme", "plot"): ("-o", "--outdir")}
REQUIRE_OUT.update({("kgirl.tauls", s): ("--out",)
                    for s in ("tokenizer", "data", "train", "memory", "improve", "hpsearch")})
SHORT_OUT = {("kgirl.iwme", "plot"): "-o", ("kgirl.harness", "report"): "-o"}
HARNESS_READ = frozenset({"repos", "search", "card", "blast", "coupling", "route", "trace", "report"})
HARNESS_READ_SUB = {"soup": frozenset({"stats", "ledger"}), "skills": frozenset({"list"})}

DENY_ASSIGN = re.compile(r"^(GIT_[A-Z_]*|PAGER|MANPAGER|LESSOPEN|LESSCLOSE|LD_PRELOAD|LD_LIBRARY_PATH|"
                         r"PYTHONSTARTUP|BASH_ENV|ENV|PROMPT_COMMAND|PATH)=")
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
OPERATOR = re.compile(r"&&|\|\||;;|\|&|&>>|&>|>>|>&|<&|>\||<<<|<<|[;&|()<>\n]")
PUNCT = ";&|()<>\n"
SEPARATORS = frozenset({";", ";;", "&&", "||", "|", "|&", "&", "\n"})
REDIRECT_OUT = frozenset({">", ">>", "&>", "&>>", ">|"})


class Deny(Exception):
    pass


class Ctx:
    def __init__(self, project: str, root: str):
        self.project, self.root = project, root


# ----------------------------------------------------------------------------------------------- paths

def project_dir(payload: dict) -> str:
    return os.path.realpath(os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd())


def allowed_root(project: str) -> str:
    expected = os.path.join(project, *ALLOWED_DIR)
    if os.path.realpath(expected) != expected:
        raise Deny("runs/temper resolves somewhere else (a symlink in the path); refusing all writes")
    return expected


def resolve(path: str, cwd: str) -> str:
    if any(c in path for c in "$`"):
        raise Deny(f"cannot verify a path that uses shell expansion: {path!r}")
    p = os.path.expanduser(path)
    return os.path.realpath(p if os.path.isabs(p) else os.path.join(cwd, p))


def inside(real: str, root: str) -> bool:
    return real == root or real.startswith(root + os.sep)


def check_target(path: str, cwd: str, ctx: Ctx, why: str) -> None:
    if path in DEVICES:
        return
    real = resolve(path, cwd)
    if real in DEVICES or inside(real, ctx.root):
        return
    raise Deny(f"{why} writes to {path!r} ({real}), which is outside runs/temper/")


# ----------------------------------------------------------------------------------------------- tokens

def tokens(command: str) -> list[str]:
    for marker in ("$(", "`", "<(", ">("):
        if marker in command:
            raise Deny(f"command or process substitution ({marker}) is not allowed; run each command on its own")
    lx = shlex.shlex(command, posix=True, punctuation_chars=PUNCT)
    lx.whitespace = " \t\r"
    lx.whitespace_split = True
    out: list[str] = []
    for tok in lx:
        out.extend(OPERATOR.findall(tok) if tok and all(c in PUNCT for c in tok) else [tok])
    return out


def operands(args: list[str]) -> list[str]:
    out, after = [], False
    for a in args:
        if a == "--" and not after:
            after = True
        elif after or not a.startswith("-") or a == "-":
            out.append(a)
    return out


def strip_wrappers(seg: list[str]) -> tuple[dict, list[str]]:
    env: dict = {}

    def take_assign(tok: str) -> None:
        if DENY_ASSIGN.match(tok):
            raise Deny(f"setting {tok.split('=', 1)[0]} can make git, the pager or the shell run other programs")
        k, v = tok.split("=", 1)
        env[k] = v

    i = 0
    while i < len(seg) and ASSIGN.match(seg[i]):
        take_assign(seg[i])
        i += 1
    while i < len(seg) and os.path.basename(seg[i]) in WRAPPERS:
        w = os.path.basename(seg[i])
        i += 1
        while i < len(seg) and (seg[i].startswith("-") or ASSIGN.match(seg[i])):
            if ASSIGN.match(seg[i]):
                take_assign(seg[i])
            i += 1
        if w in ("timeout", "nice") and i < len(seg) and re.match(r"^[0-9.]+[smhd]?$", seg[i]):
            i += 1                                                 # the duration or the niceness
        while i < len(seg) and ASSIGN.match(seg[i]):
            take_assign(seg[i])
            i += 1
    return env, seg[i:]


# ----------------------------------------------------------------------------------------------- programs

def check_git(args: list[str]) -> None:
    i = 0
    while i < len(args) and args[i].startswith("-"):
        a = args[i]
        if a.startswith("-c") or a.startswith("--config-env"):
            raise Deny("git -c / --config-env can run programs through configuration; not allowed")
        i += 2 if a in GIT_OPTS_WITH_VALUE else 1
    if i >= len(args):
        return
    sub, rest = args[i], args[i + 1:]
    if sub == "branch" and all(a in GIT_BRANCH_READ for a in rest):
        return
    if sub not in GIT_READ_ONLY:
        raise Deny(f"git {sub} changes repository state or reaches a remote; TEMPER may only read git")
    if sub == "grep" and any(a.startswith("-O") or a.startswith("--open-files-in-pager") for a in rest):
        raise Deny("git grep -O runs a pager program; not allowed")


def check_python(args: list[str], cwd: str, ctx: Ctx) -> tuple[str | None, list[str]]:
    """Allow `python [opts] script.py ...` and `python [opts] -m module ...`; returns (module, module_args)."""
    k = 0
    while k < len(args):
        a = args[k]
        if a == "-" or (a.startswith("-") and not a.startswith("--") and "c" in a[1:].split("m")[0]):
            raise Deny("python -c / python - runs inline code; write a script under runs/temper/scripts/")
        if a in ("-W", "-X", "--check-hash-based-pycs"):
            k += 2                                                 # options that take a separate value
            continue
        if a == "-m" or (a.startswith("-") and not a.startswith("--") and a.endswith("m")):
            mod = args[k + 1] if k + 1 < len(args) else ""
            if not mod:
                raise Deny("python -m without a module")
            if mod in PYTHON_DENY_MODULES or mod.split(".")[0] in ("pip", "ensurepip"):
                raise Deny(f"python -m {mod} installs software or opens the network")
            return mod, args[k + 2:]
        if not a.startswith("-"):
            if inside(resolve(a, cwd), os.path.join(ctx.project, "src", "kgirl", "harness")):
                raise Deny("run the harness as `python3 -m kgirl.harness ...` so its subcommand can be checked")
            return None, args[k + 1:]
        k += 1
    raise Deny("python without a script or -m reads code from input; run a script file instead")


def check_harness(margs: list[str], env: dict, cwd: str, ctx: Ctx) -> None:
    home, i = None, 0
    while i < len(margs) and margs[i].startswith("-"):
        if margs[i] == "--home" and i + 1 < len(margs):
            home, i = margs[i + 1], i + 2
            continue
        if margs[i].startswith("--home="):
            home = margs[i].split("=", 1)[1]
        i += 1
    sub = margs[i] if i < len(margs) else ""
    rest = margs[i + 1:]
    subsub = (operands(rest)[:1] or [""])[0]
    if sub == "mcp":
        raise Deny("serving the harness over MCP is not a tuning task")
    if sub == "ask" or "--summarize" in rest:
        raise Deny("ask / --summarize send repository text to the scout model; not needed for tuning")
    if sub == "soup" and subsub == "promote":
        raise Deny("promoting memory is the user's decision, never TEMPER's")
    if sub == "task" and "--apply" in rest:
        raise Deny("task --apply writes into a repository; TEMPER only measures")
    if sub == "soup" and subsub == "export-sft":
        out = operands(rest)[1:2]
        if not out:
            raise Deny("soup export-sft needs an output path under runs/temper/")
        check_target(out[0], cwd, ctx, "soup export-sft")
        return
    if sub in HARNESS_READ or subsub in HARNESS_READ_SUB.get(sub, frozenset()):
        return
    h = home or env.get("KGIRL_HOME") or os.environ.get("KGIRL_HOME")
    if not h or not inside(resolve(h, cwd), ctx.root):
        raise Deny(f"harness {sub} {subsub}".rstrip() + " changes state in your real KGIRL_HOME; pass "
                   "--home runs/temper/<id>/home (or KGIRL_HOME=...) to experiment on a copy")
    if sub == "index":
        for op in operands(rest):
            if "://" in op or op.startswith("git@") or (re.match(r"^[\w.-]+/[\w.-]+$", op)
                                                          and not os.path.exists(resolve(op, cwd))):
                raise Deny(f"index {op!r} would clone from the network; index local paths only")
    if sub == "task":
        for j, a in enumerate(rest):
            v = rest[j + 1] if a == "--verify" and j + 1 < len(rest) else (
                a.split("=", 1)[1] if a.startswith("--verify=") else None)
            if v is not None:
                check_bash(v, cwd, ctx)                            # the verifier is a command TEMPER chose


def subcommand(margs: list[str]) -> str:
    """First positional argument, skipping global options and the value of --home."""
    i = 0
    while i < len(margs) and margs[i].startswith("-"):
        i += 2 if margs[i] == "--home" else 1
    return margs[i] if i < len(margs) else ""


def check_module(module: str, margs: list[str], env: dict, cwd: str, ctx: Ctx) -> None:
    if module == "kgirl.harness":
        check_harness(margs, env, cwd, ctx)
    sub = subcommand(margs)
    need = REQUIRE_OUT.get((module, sub))
    if need and not any(a == f or a.startswith(f + "=") for a in margs for f in need):
        raise Deny(f"{module} {sub} writes to a default path outside runs/temper/; pass {need[0]} runs/temper/<id>/...")
    short = SHORT_OUT.get((module, sub))
    if short:
        for j, a in enumerate(margs):
            if a == short and j + 1 < len(margs):
                check_target(margs[j + 1], cwd, ctx, f"{module} {sub} {short}")


def check_out_flags(prog: str, args: list[str], cwd: str, ctx: Ctx) -> None:
    for j, a in enumerate(args):
        for flag in OUT_FLAGS:
            if a == flag and j + 1 < len(args):
                check_target(args[j + 1], cwd, ctx, f"{prog} {flag}")
            elif a.startswith(flag + "="):
                check_target(a.split("=", 1)[1], cwd, ctx, f"{prog} {flag}")
        if prog == "sort" and a == "-o" and j + 1 < len(args):
            check_target(args[j + 1], cwd, ctx, "sort -o")


def check_segment(seg: list[str], cwd: str, ctx: Ctx) -> str:
    """Check one simple command; returns the working directory after it (for cd)."""
    env, seg = strip_wrappers(seg)
    if not seg:
        return cwd
    prog, args = os.path.basename(seg[0]), seg[1:]
    if prog == "cd":
        ops = operands(args)
        return resolve(ops[0], cwd) if ops else os.path.expanduser("~")
    if prog in NETWORK_INSTALL_ESCALATE:
        raise Deny(f"{prog} reaches the network, installs software or escalates privileges; that is the user's call")
    if prog in INLINE_ONLY:
        raise Deny(f"{prog} runs inline code or other commands; write a script under runs/temper/scripts/ and run it")
    if prog in FILE_REWRITERS:
        raise Deny(f"{prog} rewrites files in place; produce outputs with Write under runs/temper/")
    if prog == "git":
        check_git(args)
    elif prog in SHELLS:
        for a in args:                                             # the shell's own options, before the script
            if not a.startswith("-") or a == "--":
                break
            if not a.startswith("--") and any(f in a[1:] for f in "csi"):
                raise Deny(f"{prog} -c/-s/-i runs inline code; run a script file instead")
        if not operands(args):
            raise Deny(f"{prog} without a script reads commands from input; run a script file instead")
    elif re.match(r"^python[0-9.]*$", prog):
        module, margs = check_python(args, cwd, ctx)
        if module:
            check_module(module, margs, env, cwd, ctx)
    elif prog in EVAL_FLAGS:
        if any(a in EVAL_FLAGS[prog] for a in args):
            raise Deny(f"{prog} {sorted(EVAL_FLAGS[prog])[0]} runs inline code; run a script file")
        if not operands(args):
            raise Deny(f"{prog} without a script reads code from input; run a script file instead")
    elif prog == "ollama":
        sub = operands(args)[:1]
        if sub and sub[0] not in OLLAMA_READ:
            raise Deny(f"ollama {sub[0]} downloads, publishes or changes models; that is the user's call")
    elif prog == "find":
        bad = {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf", "-fls"}
        if any(a in bad for a in args):
            raise Deny("find -exec/-delete/-fprint runs commands or writes files; not allowed")
    elif prog in WRITE_ALL_ARGS:
        for op in operands(args):
            check_target(op, cwd, ctx, prog)
    elif prog in WRITE_SKIP_FIRST:
        for op in operands(args)[1:]:
            check_target(op, cwd, ctx, prog)
    elif prog in WRITE_DEST:
        dests = [args[j + 1] for j, a in enumerate(args) if a in ("-t", "--target-directory") and j + 1 < len(args)]
        dests += [a.split("=", 1)[1] for a in args if a.startswith("--target-directory=")]
        for dest in dests or operands(args)[-1:]:
            check_target(dest, cwd, ctx, prog)
    check_out_flags(prog, args, cwd, ctx)
    return cwd


def check_bash(command: str, cwd: str, ctx: Ctx) -> None:
    toks = tokens(command)
    seg: list[str] = []
    stack: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in ("<<", "<<<"):
            raise Deny("heredocs and here-strings feed inline code to programs; write a script file instead")
        if t in REDIRECT_OUT or (t == ">&" and i + 1 < len(toks) and not re.match(r"^(\d+|-)$", toks[i + 1])):
            if i + 1 >= len(toks):
                raise Deny("redirection without a target")
            check_target(toks[i + 1], cwd, ctx, f"redirection {t}")
            i += 2
            continue
        if t in (">&", "<&", "<"):
            i += 2                                                 # an fd duplication or an input file
            continue
        if t in SEPARATORS or t in ("(", ")"):
            cwd = check_segment(seg, cwd, ctx) if seg else cwd
            seg = []
            if t == "(":
                stack.append(cwd)
            elif t == ")":
                cwd = stack.pop() if stack else cwd
            i += 1
            continue
        if re.match(r"^\d+$", t) and i + 1 < len(toks) and toks[i + 1] in REDIRECT_OUT | {">&"}:
            i += 1                                                 # the "2" of 2>file
            continue
        seg.append(t)
        i += 1
    if seg:
        check_segment(seg, cwd, ctx)


# ----------------------------------------------------------------------------------------------- main

def decide(payload: dict) -> None:
    tool = payload.get("tool_name", "")
    tin = payload.get("tool_input") or {}
    if tool in READ_ONLY_TOOLS:
        return
    project = project_dir(payload)
    ctx = Ctx(project, allowed_root(project))
    cwd = os.path.realpath(payload.get("cwd") or project)
    if tool == "Write":
        path = tin.get("file_path")
        if not isinstance(path, str) or not path:
            raise Deny("Write without a file_path")
        check_target(path, cwd, ctx, "Write")
        return
    if tool == "Bash":
        command = tin.get("command")
        if not isinstance(command, str):
            raise Deny("Bash without a command")
        check_bash(command, cwd, ctx)
        return
    raise Deny(f"TEMPER's guard allows only Read, Grep, Glob, LS, Bash and Write; {tool or 'this tool'} is refused")


def refuse(reason: str) -> int:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                             "permissionDecisionReason": f"TEMPER guard: {reason}"}}))
    print(f"TEMPER guard: {reason}", file=sys.stderr)
    return 2


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            return refuse("hook input is not a JSON object")
        decide(payload)
        return 0
    except Deny as d:
        return refuse(str(d))
    except Exception as e:                                         # fail closed
        return refuse(f"guard error ({type(e).__name__}: {e}); refusing to be safe")


if __name__ == "__main__":
    sys.exit(main())

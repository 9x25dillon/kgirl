# Blast-Radius Review for MCP Servers

Over MCP, the caller is a model, and it may be a model that just read a malicious web page, issue or README. So a tool argument is not the user's intent; it is *whatever the model was persuaded to send*. The question this review answers for every tool:

> If the model's input were hostile, what could it reach, and does anything stop it?

The patterns below come from a real review of a dependency-free Python MCP server: the kgirl harness, fixed in 9x25dillon/kgirl#55. Every one of them passed that server's own tests, because tests exercise intended use, not hostile use.

Run the checker first, then walk the checklist at the end:

```bash
python scripts/blast_radius_check.py path/to/server     # exit 1 if anything is found; --json for tooling
```

The checker is a lint. It finds these patterns by their shape, so a clean run is not a proof of safety.

---

## BR003: Model-written state goes live

**The pattern.** A tool stores model input (a note, memory, preference or config) where later sessions read it back as context, with nothing in between. One prompt injection becomes permanent: the stored text is recalled in every later session, and each recall can write it again.

**Real case.** A `remember` tool called `soup.add(text, source="user")`. The new fragment was **active**, so it was recalled immediately. It was also **attributed to the user**, so nothing downstream could tell it came from a model. The pool had a careful curator that staged new memories until independent evidence reinforced them, but this path bypassed it.

**Fix.** Stage model-written state, record the real source, and let a person promote it. Don't reuse a promotion path that counts repetition as evidence: a model can repeat itself.

```python
# Python: store as pending, attribute to the caller, never auto-promote
@mcp.tool()
async def remember(text: str) -> str:
    fid = store.add(text, source="mcp", status="staged")        # not active, not "user"
    return f"staged note #{fid}; it is used after you approve it (mytool approve {fid})"
```

```typescript
// TypeScript
server.registerTool("remember", { inputSchema: { text: z.string().max(2000) } }, async ({ text }) => {
  const id = await store.add({ text, source: "mcp", status: "staged" });
  return { content: [{ type: "text", text: `staged note #${id}; approve it to make it recallable` }] };
});
```

Reads must then serve only approved entries, for example `WHERE status = 'active'`.

---

## BR001: The model chooses a program

**The pattern.** A tool takes a command, script, verifier or shell string from its arguments and runs it. "Prevent command injection" usually means escaping metacharacters. That's the small case. The large case is the model choosing the *program*: `python -c "…"` needs no metacharacters at all.

**Real case.** `jev_swarm_task(verify="…")` split the string and ran it as the verification step of a coding agent, with the user's full environment.

**Fix.** The model picks from a fixed menu that the user configured. It never supplies the command itself.

```python
CHECKS = {"test": ["pytest", "-q"], "lint": ["ruff", "check", "."]}   # or read from user config / env

@mcp.tool()
async def run_check(check: Literal["test", "lint"]) -> str:
    if check not in CHECKS:
        raise ValueError(f"unknown check; choose one of {sorted(CHECKS)}")
    p = subprocess.run(CHECKS[check], cwd=PROJECT_ROOT, env=minimal_env(), capture_output=True, text=True, timeout=300)
    return (p.stdout + p.stderr)[-4000:]
```

```typescript
const CHECKS: Record<string, string[]> = { test: ["npm", "test"], lint: ["npx", "eslint", "."] };
// inputSchema: { check: z.enum(["test", "lint"]) } — and still look it up, never pass it through
const [cmd, ...argv] = CHECKS[check];
execFile(cmd, argv, { cwd: PROJECT_ROOT, env: minimalEnv(), timeout: 300_000 });
```

Never use `shell=True` (or `exec` with a string) on anything built from arguments. A fixed program with argument values (`["git", "log", ref]`) is acceptable when every value is validated. Watch for values starting with `-`, which git and most CLIs read as options.

---

## BR002: Children inherit every secret

**The pattern.** `subprocess.run(..., env=os.environ)`, `{**os.environ, …}`, or `spawn(..., { env: process.env })` hands API keys, tokens and cloud credentials to whatever runs. With BR001, that's exfiltration. Even without BR001, it means a test suite or build script that reads environment variables now sees all of them.

**Fix.** Pass only what the child needs, or drop secret-shaped names:

```python
SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.I)
def minimal_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not SECRET.search(k)} | {"PYTHONDONTWRITEBYTECODE": "1"}
```

```typescript
const minimalEnv = () => Object.fromEntries(Object.entries(process.env).filter(([k]) => !/KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL/i.test(k)));
```

---

## BR005: Writes go wherever the model points

**The pattern.** A tool writes, moves or deletes at a path from its arguments. `../../.ssh/authorized_keys` is the classic case. A symlink inside the allowed directory that points outside it is the case people miss.

**Real case.** `apply=true` wrote an agent's diff into any `repo_path` the model named.

**Fix.** Resolve to a real path, require it inside a root the user configured, and require consent for anything that changes a user's working tree.

```python
ROOT = Path(os.environ["MYTOOL_ROOT"]).resolve()
def inside_root(user_path: str) -> Path:
    target = (ROOT / user_path).resolve()          # resolves symlinks too
    if not target.is_relative_to(ROOT):
        raise PermissionError(f"{user_path} is outside {ROOT}")
    return target
```

```typescript
const ROOT = realpathSync(process.env.MYTOOL_ROOT!);
function insideRoot(p: string): string {
  const target = realpathSync(resolve(ROOT, p));    // throws if missing; for new files, realpath the parent
  if (target !== ROOT && !target.startsWith(ROOT + sep)) throw new Error(`${p} is outside ${ROOT}`);
  return target;
}
```

Gate destructive tools behind a setting the user controls, such as `MYTOOL_ALLOW_APPLY=1`, not behind a tool argument. Set `destructiveHint` honestly. Annotations are hints to the client; enforce the rule in code.

---

## BR004: Version echo (hand-rolled servers)

**The pattern.** A server written without an SDK answers `initialize` with whatever `protocolVersion` the client sent. The client then believes the server speaks a protocol revision it has never implemented. The official SDKs negotiate correctly: they echo the client's version only when it's supported, and otherwise answer with their latest.

**Fix.**

```python
SUPPORTED = ("2025-06-18", "2025-03-26", "2024-11-05")
requested = params.get("protocolVersion")
result = {"protocolVersion": requested if requested in SUPPORTED else SUPPORTED[0], ...}
```

If you hand-roll stdio, also copy the SDK's framing: one JSON message per line, nothing but protocol on stdout, logs on stderr, tool failures as `isError` results, unknown methods as `-32601`.

---

## Untrusted text in results (no rule; review by hand)

Tool results that quote third-party content (web pages, issues, other people's repositories, stored notes) are a prompt-injection channel into the model that called the tool.

- Send `instructions` in the `initialize` result, saying that quoted content is data, not instructions.
- Strip control and bidirectional-formatting characters (`‪`–`‮`, `⁦`–`⁩`), and cap string lengths.
- Prefer identifiers and structured fields over raw prose when the task allows it.
- Never put text from a result into a place where it will be *executed* or *trusted*: a shell, a config file, an instruction file such as `AGENTS.md`, or memory (BR003).

## Egress (no rule; review by hand)

Anything the server sends off the machine (to an LLM API, a webhook or a third-party service) leaves the user's control.

- Redact secret-shaped strings before sending. Signatures and docstrings carry them too, for example `def connect(key="sk-live-…")`.
- Strip credentials from URLs before returning or logging them (`https://user:token@host/…`).
- Say in the tool description what leaves the machine.

---

## Checklist

For each tool, answer in one line. "Yes, because …" should name the code that enforces it.

- [ ] **Reach.** What is the worst thing this tool can do with hostile input? Is that worse than what the user's own client can already do? A tool that adds no new capability adds no blast radius.
- [ ] **State (BR003).** Does it store anything later sessions read? Is that stored as pending, attributed to `mcp`, and promoted only by a person?
- [ ] **Programs (BR001, BR002).** Does it run anything? Is the program fixed or allowlisted, without a shell, with a minimal environment, a working directory and a timeout?
- [ ] **Paths (BR005).** Does it write, move or delete? Is the real path inside a configured root? Is it gated by user consent, not by an argument?
- [ ] **Text.** Does it return third-party text? Is that text framed as data, cleaned and capped?
- [ ] **Egress.** Does anything leave the machine? Is it redacted, and documented?
- [ ] **Protocol (BR004).** Hand-rolled transport only: are versions negotiated, framing and stderr logging handled, and errors returned as results?
- [ ] **Hints match code.** Are `readOnlyHint`, `destructiveHint` and `openWorldHint` true to what the code enforces?

If several of these servers will run side by side (for example in a multi-repo harness), model them as a coupling graph and check for persistence loops: model → tool → store → tool → model. A loop that passes the same server twice is invisible to simple cycle enumeration. Strongly connected components find it. See `context-relay/plugins/porter-blast-radius` in 9x25dillon/myAssistant (`check_model`, `as_built`).

---

*Added in the mcp-builder-hardened fork (2026-10-01). See NOTICE.md.*

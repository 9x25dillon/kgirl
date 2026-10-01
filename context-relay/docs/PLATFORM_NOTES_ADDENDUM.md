# Platform Notes: Addendum

Facts the workflow and blast-radius model relies on, checked on **2026-09-30** in this session.
This file supplements the Phase 0 `docs/PLATFORM_NOTES.md` (F1–F15), which is not in this repository. Merge the two at Checkpoint A.
IDs `PN-nn` are referenced from `model/relay-model.json` and the design docs.

Status: **verified** = read in the official docs today; **carried** = taken from the Phase 0 notes, not re-checked here;
**unverified** = an assumption that still needs a source.

| ID | Fact (exact fields) | Source | Status |
|---|---|---|---|
| PN-01 | A hook's `additionalContext`, `systemMessage`, `initialUserMessage` and plain stdout are each **capped at 10,000 characters**. Over the cap, Claude Code saves the output to a file in the session directory and substitutes the path plus a **preview of the first 2,000 characters**. For `SessionStart`, plain-text stdout is added to Claude's context; stdout that starts with `{` and ends with `}` is parsed as JSON. | https://code.claude.com/docs/en/hooks | verified |
| PN-02 | `SessionStart` matcher `source` ∈ `startup`, `resume`, `clear`, `compact`, `fork`. `PreCompact` matcher ∈ `manual`, `auto`. **PreCompact can block**: exit 2 blocks with or without JSON, and so does `"decision": "block"` with `reason`. | https://code.claude.com/docs/en/hooks | verified |
| PN-03 | `SessionEnd` reasons: `clear`, `resume`, `logout`, `prompt_input_exit`, `other`. SessionEnd hooks **share a 1.5-second budget**; a longer per-hook `timeout` raises it, up to 60 s. SessionEnd cannot block; stderr is shown to the user only. | https://code.claude.com/docs/en/hooks | verified |
| PN-04 | All matching hooks **run in parallel**. An identical handler defined in several settings files runs once; a plugin's copy stays separate. | https://code.claude.com/docs/en/hooks | verified |
| PN-05 | `${CLAUDE_PLUGIN_DATA}` = `~/.claude/plugins/data/<id>/`: created on first reference, **kept across plugin updates**, and **deleted when the plugin is uninstalled from the last place it is installed** (except with `--keep-data`). It is exported to hooks, and to MCP and LSP servers, but **not** to commands Claude runs through the Bash tool. `${CLAUDE_PLUGIN_ROOT}` changes on every update, so no state may live there. | https://code.claude.com/docs/en/plugins-reference | verified |
| PN-06 | Node.js is not guaranteed on the user's machine: Claude Code ships as a standalone program (Phase 0 F14). | Phase 0 notes | carried |
| PN-07 | MCP tool output: warning at **10,000 tokens**, default maximum **25,000 tokens**, configurable with `MAX_MCP_OUTPUT_TOKENS`. Over the maximum, the result is saved to a file and replaced by its path. Per-tool override: `_meta["anthropic/maxResultSizeChars"]` in `tools/list`. The model uses 30,000 chars (the warning threshold × 3 chars/token, a conservative estimate, L1). | https://code.claude.com/docs/en/mcp | verified (limits); L1 (char conversion) |
| PN-08 | Native messaging: at most **1 MB per message from the host to the extension** (Phase 0 F13). Re-checking developer.chrome.com was blocked by this environment's network egress policy. | Phase 0 notes | carried |
| PN-09 | Bash tool output is truncated at about 30,000 characters by default. | — | **unverified** (needed for sink `cc.bash-output`) |
| PN-10 | The format of claude.ai's official data export is undocumented (Phase 0 F10). | Phase 0 notes | carried |
| PN-11 | Hook **shell form** (no `args`) runs through `sh -c` on macOS and Linux, Git Bash on Windows, and **PowerShell when Git Bash is not installed**; `shell: "bash" \| "powershell"` overrides. **Exec form** (`args` present) spawns the command directly with no shell; it needs a real executable, so `.cmd` and `.bat` shims fail. Hooks have no controlling terminal and no `/dev/tty`. | https://code.claude.com/docs/en/hooks | verified |
| PN-12 | `CLAUDE_CODE_REMOTE="true"` in remote (web) environments; unset in the local CLI. | https://code.claude.com/docs/en/hooks | verified |
| PN-13 | Default hook timeout: **600 s** for `command`, `http` and `mcp_tool` hooks (30 s for `prompt`, 60 s for `agent`). | https://code.claude.com/docs/en/hooks | verified |
| PN-14 | `plugin.json` `version` pins users to that version until it changes. `userConfig` options are strict objects that require `type`, `title` and `description`. Shell-form hooks **reject** `${user_config.*}`; use exec-form `args` or the env var `CLAUDE_PLUGIN_OPTION_<KEY>`. **claude.ai and Cowork don't install a plugin that has a top-level `bin/` directory.** | https://code.claude.com/docs/en/plugins-reference | verified |
| PN-15 | Hook processes receive `CLAUDE_PROJECT_DIR`, `CLAUDE_PLUGIN_ROOT`, `CLAUDE_PLUGIN_DATA` and `CLAUDE_PLUGIN_OPTION_<KEY>`; `SessionStart` also gets `CLAUDE_ENV_FILE`. On exit 0, stderr goes to the debug log. Common input fields: `session_id`, `transcript_path` (which may lag the in-memory state), `cwd`, `hook_event_name`, `permission_mode`. | https://code.claude.com/docs/en/hooks | verified |

## Consequences already applied in the model

- PN-01 → brief budget of 8,000 chars plus a 600-char wrapper (INV-BUDGET, anti-workflow A5); hook stdout discipline (C-STDOUT-DISCIPLINE, FM-28).
- PN-02 → PreCompact never blocks (C-FAILOPEN, FM-03); `fork` gets lineage handling (C-LINEAGE, W3).
- PN-03 → incremental capture; atomic writes (FM-04).
- PN-04 → concurrent writers are the normal case (C-ATOMIC, FM-05).
- PN-05 → store location decision D-007 (FM-06).
- PN-11 → the missing-Node shim design is open on Windows without Git Bash (FM-23, D-001).
- PN-14 → putting `ctxr` in the plugin's `bin/` would block claude.ai organization distribution. Ship the CLI through npm only.

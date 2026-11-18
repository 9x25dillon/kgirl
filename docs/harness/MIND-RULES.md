# The user's rules in chat

kgirl recalls preferences by similarity to the message. So "never push unless I name it" only reached a prompt when the message happened to look like it. A 7b model then wrote `git push` into a script the user had asked for. Two layers close that gap (`mind/rules.py`):

- **Standing rules.** Every active `preference` phrased as a rule goes into every chat system prompt, right after the persona and before anything recalled. A rule starts with Never, Always, Only, Before, Don't, Do not or No. They're global (`scope *`) or match the conversation's scope, within half the memory budget. The persona says they're binding, even in examples. They're excluded from the similarity-recalled preferences, so they don't appear twice. They aren't credited as "recalled", so feedback on an answer doesn't move them.
- **The rules check.** After the answer, `rules.check` reads its fenced code blocks only, so prose that merely mentions a command isn't flagged. It looks for:
  - `git push` (and force-push), `git pull`, `git commit`
  - `git add .`/`-A`, `git reset --hard`, `git clean -f`, `git branch -D`
  - `rm -rf`
  - `torch.load`/`pickle.load` without `weights_only=True`

  It reads shell form and list form (`["git", "-C", repo, "push"]`). A step the user's message names ("commit it", "push") is allowed.
- **On a hit:** the mind revises once with the rules and the offending snippets, streaming a short "revising" note first. If the revision still breaks a rule, the answer carries a visible ⚠ warning. It never loops. The stored turn records `rules_hits`, `rules_left` and `rules_revised`.

A clean answer costs no extra model call. Tests: `tests/harness/mind/test_rules.py`, offline, with the real console answer as a fixture. The measurement plan with a preregistered bar is kept outside the repo.

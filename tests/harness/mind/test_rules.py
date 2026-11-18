"""Offline tests for mind.rules and the Mind's rules check: standing rules in every prompt, one revision."""

import unittest

import tests.harness._fixtures  # noqa: F401  (sys.path)
from kgirl.harness.mind.rules import check, git_calls, is_rule, standing_rules
from kgirl.harness.util import estimate_tokens
from tests.harness.mind.test_mind import MindCase

# Trimmed from the answer kgirl's console gave on 2026-11-17 to "build a cross product skill ... that recovers
# records and commits any stale coding cli work". The user named commit, not pull, add-all or push.
REQUEST = ("build a cross product skill for antigraviry claude code codex and qwen code that recovers records "
           "and commits any stale coding cli work from the users desired platform.")
BAD_ANSWER = '''Here is the skill:

```python
def recover_records(repo_path):
    subprocess.run(["git", "pull"], cwd=repo_path, check=True)

def commit_changes(repo_path):
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "Recovered stale coding work"], cwd=repo_path, check=True)
    subprocess.run(["git", "push"], cwd=repo_path, check=True)
```
'''
CLEAN_ANSWER = '''Here is the skill, which never pushes:

```python
def commit_changes(repo_path, paths):
    subprocess.run(["git", "add", "--", *paths], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "rescue"], cwd=repo_path, check=True)
```
Push it yourself with `git push` once you have read the diff.
'''

RULES = ["Never commit, push or merge unless the user names it. Pushing to a public repo is publication.",
         "Never pull into an existing kgirl checkout, because several hold unique uncommitted work.",
         "When you touch file paths, use pathlib.Path, never os.path.",       # a style preference, not a rule
         "Report results honestly: say so when tests fail."]                 # not phrased as a rule


class CheckTests(unittest.TestCase):
    def names(self, answer, request=REQUEST):
        return sorted(v.name for v in check(answer, request))

    def test_the_console_answer_is_caught_but_the_named_commit_is_allowed(self):
        self.assertEqual(self.names(BAD_ANSWER), ["add-all", "pull", "push"])

    def test_clean_answer_and_prose_mentions_pass(self):
        self.assertEqual(self.names(CLEAN_ANSWER), [])
        self.assertEqual(self.names("Never run `git push --force` here; I left it out."), [])

    def test_shell_form_with_options_and_named_steps(self):
        shell = "```sh\ngit -C ~/work/x push origin main\ngit reset --hard HEAD\nrm -rf build\n```"
        self.assertEqual(self.names(shell, "tidy the repo"), ["push", "reset-hard", "rm-rf"])
        self.assertEqual(self.names(shell, "push it, reset --hard and delete build"), [])
        self.assertEqual(self.names("```sh\ngit push --force\n```", "push it"), ["force-push"])

    def test_pickle_loads(self):
        self.assertEqual(self.names("```python\nm = torch.load(p)\n```", "load the model"), ["pickle-load"])
        self.assertEqual(self.names("```python\nm = torch.load(p, weights_only=True)\n```", "load the model"), [])

    def test_git_calls_parse_both_forms(self):
        self.assertEqual(git_calls('run(["git", "-C", "/r", "commit", "-m", "x"])')[0][:2], ["commit", "-m"])
        self.assertEqual(git_calls("cd r && git status")[0][0], "status")

    def test_is_rule(self):
        self.assertEqual([is_rule(r) for r in RULES], [True, True, False, False])


class StandingRulesTests(MindCase):
    def test_rules_are_global_or_scoped_and_budgeted(self):
        self.mind()
        soup = self.a.soup
        ids = [soup.add(t, kind="preference", status="active") for t in RULES]
        scoped = soup.add("Never run the GPU above 95 C.", kind="preference", status="active", scope="gpu-unlock")
        soup.add("Never touch staged things.", kind="preference", status="staged")
        self.assertEqual([f.id for f in standing_rules(soup)], ids[:2])
        self.assertEqual([f.id for f in standing_rules(soup, "gpu-unlock")], ids[:2] + [scoped])
        one = estimate_tokens(RULES[0]) + 2           # room for exactly the first rule
        self.assertEqual([f.id for f in standing_rules(soup, budget_tokens=one)], ids[:1])


class MindRulesTests(MindCase):
    def setup_rules(self, mind):
        for t in RULES[:2]:
            self.a.soup.add(t, kind="preference", status="active")
        return mind

    def test_rules_are_in_every_prompt_and_a_violation_is_revised_once(self):
        mind = self.setup_rules(self.mind(BAD_ANSWER, CLEAN_ANSWER))
        cid = mind.store.new("t")
        streamed, events = [], []
        reply = mind.chat(cid, REQUEST, on_text=streamed.append, on_event=events.append)
        self.assertEqual(reply.text, CLEAN_ANSWER.strip())
        system, msgs = self.chat.calls[1]
        self.assertIn("RULES THE USER SET", system)
        self.assertIn("Never pull into an existing kgirl checkout", system)
        self.assertTrue(msgs[-1].content.startswith("RULES CHECK"))
        self.assertEqual(msgs[-2].content.strip(), BAD_ANSWER.strip())
        self.assertTrue(any(e.startswith("rules check: ") for e in events))
        self.assertIn("revising", "".join(streamed))
        stored = [t for t in mind.store.window(cid, 10_000) if t.role == "assistant"][-1]
        self.assertEqual(stored.content, CLEAN_ANSWER.strip())

    def test_a_revision_that_still_breaks_a_rule_carries_a_warning(self):
        mind = self.setup_rules(self.mind(BAD_ANSWER, BAD_ANSWER))
        reply = mind.chat(mind.store.new("t"), REQUEST)
        self.assertIn("kgirl's rules check", reply.text)
        self.assertIn("git push", reply.text)
        self.assertEqual(len(self.chat.calls), 2)          # one revision, never a loop

    def test_a_clean_answer_costs_no_extra_call(self):
        mind = self.setup_rules(self.mind(CLEAN_ANSWER))
        reply = mind.chat(mind.store.new("t"), REQUEST)
        self.assertEqual(reply.text, CLEAN_ANSWER.strip())
        self.assertEqual(len(self.chat.calls), 1)


if __name__ == "__main__":
    unittest.main()

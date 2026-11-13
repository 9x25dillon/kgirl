import unittest

from kgirl.harness.atlas.secrets import secret_reason

# Token-shaped values are assembled at run time, so this file holds nothing a scanner (or this filter, when
# kgirl indexes itself) would take for a real key.


class SecretRuleTests(unittest.TestCase):
    def test_known_token_shapes_name_their_rule(self):
        cases = {
            "GitHub token": 'TOKEN = "' + "gh" + "p_" + "A1b2" * 9 + '"',
            "AWS access key": "key = " + "AK" + "IA" + "ABCDEFGHIJKLMNOP",
            "Hugging Face token": "hf" + "_" + "x" * 34,
            "Anthropic key": "sk" + "-ant-" + "api03-" + "y" * 40,
            "OpenAI key": "sk" + "-proj-" + "z" * 40,
            "private key block": "-----BEGIN " + "OPENSSH PRIVATE KEY-----\nabc\n",
            "Slack token": "xo" + "xb-" + "1234567890-abcdef",
            "Google API key": "AI" + "za" + "B" * 35,
        }
        for rule, text in cases.items():
            with self.subTest(rule=rule):
                self.assertEqual(secret_reason("app/config.py", text), rule)

    def test_credentials_in_urls_and_assignments(self):
        self.assertEqual(secret_reason("README.md", "git clone https://bot:" + "Q7w" * 5 + "@github.com/o/r.git"),
                         "credentials in URL")
        self.assertEqual(secret_reason("db.py", 'password = "' + "Zx9" * 6 + '"'), "hard-coded secret")
        self.assertEqual(secret_reason("run.sh", 'export API_KEY="' + "k3y" * 6 + '"'), "hard-coded secret")
        self.assertEqual(secret_reason("README.md", "translate.service.vendor.key" + "=" + "a1B2c3" * 12),
                         "unquoted secret")

    def test_unquoted_identifiers_without_digits_pass(self):
        for text in ("if (Pressed(EKeys::Gamepad_FaceButton_Bottom)) Accept();",
                     "API keys (ANTHROPIC_API_KEY/OPENAI_API_KEY) rotate at their consoles",
                     "token = make_session_token_for_current_user_now"):
            with self.subTest(text=text):
                self.assertIsNone(secret_reason("x.cpp", text))

    def test_placeholders_and_ordinary_code_pass(self):
        for text in ('password = "your-password-here-please"', "postgresql://user:${DB_PASSWORD}@localhost/db",
                     "https://user:changeme-changeme@example.com", 'api_key = os.environ["API_KEY"]',
                     "def tokenize(text):\n    return text.split()\n", "see https://github.com/o/r for the key"):
            with self.subTest(text=text):
                self.assertIsNone(secret_reason("x.py", text))

    def test_key_and_credential_files_by_name(self):
        for path, rule in (("deploy/id_ed25519", "ssh private key"), ("certs/server.pem", "key file"),
                           ("config/credentials.json", "credentials file"), (".npmrc", "auth config"),
                           ("keys/UTC--2024-01-01T00-00-00Z--abc", "wallet file")):
            with self.subTest(path=path):
                self.assertEqual(secret_reason(path, "harmless"), rule)
        for path in ("deploy/id_ed25519.pub", "docs/keys.md", "src/secrets_scan.py", "credentials.py"):
            with self.subTest(path=path):
                self.assertIsNone(secret_reason(path, "harmless"))


if __name__ == "__main__":
    unittest.main()

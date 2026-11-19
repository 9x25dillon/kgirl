"""Offline checks for the read-only doctor report."""

import json
import os
import subprocess
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

import tests.harness._fixtures  # noqa: F401
from kgirl.harness import doctor


class Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class DoctorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        chunks = self.home / "chunks" / "all"
        chunks.mkdir(parents=True)
        self.index = chunks / "all-repos.db"
        self.index.write_text("index", encoding="utf-8")
        self.marker = chunks / "router-synced"
        self.marker.write_text("", encoding="utf-8")
        os.utime(self.index, (100, 100))
        os.utime(self.marker, (200, 200))
        self.code = self.home / "code"
        self.code.mkdir()
        self.venv_python = self.home / "venv" / "bin" / "python3"
        self.venv_python.parent.mkdir(parents=True)
        self.venv_python.write_text("", encoding="utf-8")
        self.env = mock.patch.dict(os.environ, {"KGIRL_HOME": str(self.home), "KGIRL_CODE": str(self.code),
                                                "KGIRL_VENV": str(self.home / "venv"), "OLLAMA_HOST": "",
                                                "KGIRL_SMITH_MODEL": "qwen2.5-coder:7b",
                                                "KGIRL_JEV_MODEL": "qwen2.5-coder:1.5b",
                                                "KGIRL_BUILD_MODEL": "qwen2.5-coder:7b",
                                                "KGIRL_CHAT_MODEL": "qwen2.5-coder:7b"}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)

    @staticmethod
    def opener(url, timeout):
        body = {"models": [{"name": "qwen2.5-coder:7b"}, {"name": "qwen2.5-coder:1.5b"}]}
        return Response(json.dumps(body).encode())

    @staticmethod
    def runner(argv, **kwargs):
        if argv[0] == "nvidia-smi":
            return subprocess.CompletedProcess(argv, 0, "72\n", "")
        if argv[0] == "git":
            return subprocess.CompletedProcess(argv, 0, "abc1234\n", "")
        if argv[:4] == ["codex", "mcp", "get", "kgirl"]:
            return subprocess.CompletedProcess(argv, 0, "kgirl\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def test_healthy_report_and_router_marker(self):
        checks = doctor.collect(self.home, runner=self.runner, opener=self.opener, now=300)
        self.assertTrue(all(c.status == "OK" for c in checks), checks)
        self.assertEqual({c.name for c in checks}, {"ollama", "router models", "GPU temperature", "chunk index",
                                                   "router sync", "KGIRL_CODE", "kgirl MCP", "pytest"})

    def test_non_loopback_ollama_is_refused_without_a_request(self):
        with mock.patch.dict(os.environ, {"OLLAMA_HOST": "http://example.com:11434"}):
            checks = doctor.collect(self.home, runner=self.runner, opener=mock.Mock(), now=300)
        ollama = next(c for c in checks if c.name == "ollama")
        self.assertEqual(ollama.status, "FAIL")
        self.assertIn("non-loopback", ollama.detail)

    def test_hot_gpu_and_stale_router_are_warnings(self):
        os.utime(self.marker, (50, 50))
        def hot(argv, **kwargs):
            if argv[0] == "nvidia-smi":
                return subprocess.CompletedProcess(argv, 0, "95\n", "")
            return self.runner(argv, **kwargs)
        checks = doctor.collect(self.home, runner=hot, opener=self.opener, now=300)
        self.assertEqual(next(c for c in checks if c.name == "GPU temperature").status, "WARN")
        self.assertEqual(next(c for c in checks if c.name == "router sync").status, "WARN")

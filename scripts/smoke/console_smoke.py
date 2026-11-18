#!/usr/bin/env python3
"""Check the local console's HTTP contract with scratch state and a fake chat backend."""

from __future__ import annotations

import argparse
import http.client
import importlib
import json
import socket
import threading
import time
from contextlib import ExitStack
from http.server import ThreadingHTTPServer
from html.parser import HTMLParser
from types import SimpleNamespace
from unittest.mock import patch

from _support import Stages, isolated_home, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true", help="also require the Helix/Holo UI lane routes")
    args = parser.parse_args()
    stages, state = Stages(), SimpleNamespace()
    attempts = []
    with isolated_home() as root, ExitStack() as cleanup:
        def setup():
            from kgirl.harness.assistant import Assistant
            from kgirl.harness.console.core import Console
            from kgirl.harness.console.server import Handler
            from kgirl.harness.hermes.router import CHAT, BUILD, Router
            from kgirl.harness.llm.scripted import ScriptedBackend
            from kgirl.harness.mind.builder import Builder, BuildResult
            from kgirl.harness.mind.writer import Draft, Writer

            class Fake(ScriptedBackend):
                def stream(self, system, messages, **kwargs):
                    text = self.complete(system, messages, **kwargs).text
                    yield text[:8]
                    yield text[8:]

            fake = Fake(lambda system, messages: "Offline console reply.")
            state.a = Assistant(home=root / "home", router=Router({CHAT: [fake], BUILD: [fake]}),
                                trace=False, life=False)
            cleanup.callback(state.a.close)
            state.workspace = root / "project"
            state.workspace.mkdir()
            (state.workspace / "sample.py").write_text("def smoke_add(a, b):\n    return a + b\n", encoding="utf-8")
            state.a.atlas.ingest(state.workspace, "smoke-project", "")
            state.c = Console(state.a, workspace_root=state.workspace)
            cleanup.callback(state.c.close)
            state.c.ops().background = False
            state.c.ops().reflector.every = 999
            # HTTP/job plumbing is under test here; model computation is covered by mind_smoke.py.
            built = BuildResult(spec="add", code="def add(a, b):\n    return a + b\n", tests="",
                                chosen=0, group_size=2, passed=1, credible=1, candidates=[], suites=[], exemplars=[])
            cleanup.enter_context(patch.object(Builder, "build", return_value=built))
            cleanup.enter_context(patch.object(Writer, "write", return_value=Draft(
                "Smoke draft", {}, [("Example", "Two plus three is five.")])))
            evolve = importlib.import_module("kgirl.harness.mind.evolve")
            cleanup.enter_context(patch.object(evolve, "evolve", return_value=evolve.EvolveReport()))
            handler = type("SmokeHandler", (Handler,), {"console": state.c})
            state.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            state.httpd.daemon_threads = True
            cleanup.callback(state.httpd.server_close)
            thread = threading.Thread(target=state.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
            thread.start()
            cleanup.callback(thread.join, 2)
            cleanup.callback(state.httpd.shutdown)
            state.port = state.httpd.server_address[1]
            state.origin = f"http://127.0.0.1:{state.port}"
            connect = socket.socket.connect
            connect_ex = socket.socket.connect_ex
            resolve = socket.getaddrinfo

            def guarded(fn):
                def call(sock, address):
                    if not isinstance(address, tuple) or address[:2] != ("127.0.0.1", state.port):
                        attempts.append("unexpected connection")
                        raise RuntimeError("console smoke allows only its own loopback server")
                    return fn(sock, address)
                return call

            def local_resolution(host, port, *rest, **kw):
                if host != "127.0.0.1" or int(port) != state.port:
                    attempts.append("unexpected address lookup")
                    raise RuntimeError("console smoke blocks non-fixture address lookups")
                return resolve(host, port, *rest, **kw)

            cleanup.enter_context(patch.object(socket.socket, "connect", guarded(connect)))
            cleanup.enter_context(patch.object(socket.socket, "connect_ex", guarded(connect_ex)))
            cleanup.enter_context(patch.object(socket, "getaddrinfo", local_resolution))
            return "real HTTP server on an ephemeral loopback port; scratch stores only"

        def request(method, path, body=None, *, expected=200, headers=None, authenticated=True, raw=False):
            connection = http.client.HTTPConnection("127.0.0.1", state.port, timeout=3)
            supplied = {"Origin": state.origin, "Content-Type": "application/json"}
            token = getattr(state, "token", "")
            if authenticated and token:
                supplied["X-Kgirl-Token"] = token
            supplied.update(headers or {})
            try:
                connection.request(method, path, body=json.dumps(body) if body is not None else None,
                                   headers=supplied)
                response = connection.getresponse()
                data = response.read()
                require(response.status == expected, f"{method} {path}: expected {expected}, got {response.status}")
                if raw:
                    return data
                value = json.loads(data)
                if expected >= 400:
                    require(isinstance(value, dict) and bool(value.get("error")), "error response needs a reason")
                return value
            finally:
                connection.close()

        def bootstrap():
            class TokenMeta(HTMLParser):
                token = ""

                def handle_starttag(self, tag, attrs):
                    values = dict(attrs)
                    if tag == "meta" and values.get("name") == "kgirl-token":
                        self.token = values.get("content") or ""

            page = TokenMeta()
            page.feed(request("GET", "/", authenticated=False, raw=True).decode("utf-8"))
            require(len(page.token) >= 20 and page.token != "__KGIRL_TOKEN__",
                    "launch token missing from HTML; integrate the console security lane")
            state.token = page.token
            return "launch token read from served HTML (value never printed)"

        def job(path, body, *, expected_status="done"):
            jid = request("POST", path, body)["job"]
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                value = request("GET", f"/api/jobs/{jid}")
                if value["status"] != "running":
                    require(value["status"] == expected_status,
                            f"{path} job: expected {expected_status}, got {value['status']}: {value.get('error')}")
                    return value["result"] if expected_status == "done" else value["error"]
                time.sleep(0.02)
            raise TimeoutError(f"{path} job did not finish in five seconds")

        def reads():
            for path, kind in (("/api/state", dict), ("/api/jobs", list), ("/api/conversations", list),
                               ("/api/tools", list), ("/api/approvals", list), ("/api/workspace", dict),
                               ("/api/soup", list), ("/api/soup/ledger", list), ("/api/repos", list),
                               ("/api/search?q=smoke_add", dict), ("/api/skills", list), ("/api/vitals", dict),
                               ("/api/plan", dict), ("/api/matrix", dict), ("/api/trace", list),
                               ("/api/mcp/tools", list)):
                require(isinstance(request("GET", path), kind), f"wrong response shape for {path}")
            source = request("GET", "/api/source?repo=smoke-project&path=sample.py")
            require("def smoke_add" in source["text"], "indexed source missing")
            for path in ("/", "/static/app.js", "/static/style.css"):
                require(bool(request("GET", path, raw=True)), f"empty static asset: {path}")
            return "all baseline GET routes and static assets responded"

        def conversation():
            state.cid = request("POST", "/api/conversations", {"title": "Smoke conversation"})["id"]
            mind = state.c.mind(state.cid)
            mind.background, mind.reflector.every = False, 999
            result = job("/api/chat", {"conv_id": state.cid, "text": "Hello."})
            require(result["text"] == "Offline console reply.", "chat result mismatch")
            detail = request("GET", f"/api/conversations/{state.cid}")
            require(len(detail["turns"]) == 2, "chat transcript missing")
            request("POST", "/api/feedback", {"conv_id": state.cid, "verdict": "positive"})
            request("POST", "/api/tool", {"name": "search_code", "args": {"query": "smoke_add"}})
            request("POST", "/api/apply", {})
            other = request("POST", "/api/conversations", {"title": "Disposable"})["id"]
            require(request("DELETE", f"/api/conversations/{other}")["ok"], "conversation delete failed")
            request("GET", f"/api/conversations/{other}", expected=400)
            return "create, streamed chat job, feedback, tools and delete"

        def memory():
            fid = request("POST", "/api/soup/remember", {"text": "The smoke fixture uses addition."})["id"]
            request("POST", "/api/soup/credit", {"ids": [fid], "success": True})
            request("POST", "/api/soup/status", {"id": fid, "status": "staged"})
            rows = request("GET", "/api/soup?status=staged")
            require(any(r["id"] == fid for r in rows), "memory transition missing")
            request("POST", "/api/skills/forge", {})
            request("POST", "/api/skills/export", {"dir": str(root / "exported-skills")})
            return "memory writes and skill export confined to scratch state"

        def workspace():
            require(not request("POST", "/api/workspace/auto", {"on": False})["auto_edit"], "auto-edit enabled")
            allowed = request("POST", "/api/workspace/allow", {"command": "python3 --version"})
            require("python3 --version" in allowed["allowed"], "workspace allowlist missing")
            require(request("POST", "/api/workspace/verify", {"command": ""})["verify"] is None,
                    "verification command was not cleared")
            request("POST", "/api/workspace/undo", {})
            return "workspace settings and empty undo; no command executed"

        def approvals():
            state.c.approvals.timeout = 2
            answer = []
            thread = threading.Thread(target=lambda: answer.append(state.c.approvals.ask("Smoke?", "No edit")))
            thread.start()
            try:
                deadline = time.monotonic() + 1
                pending = []
                while time.monotonic() < deadline and not pending:
                    pending = request("GET", "/api/approvals")
                    if not pending:
                        time.sleep(0.01)
                require(bool(pending), "approval did not appear")
                request("POST", "/api/approve", {"id": pending[0]["id"], "yes": False})
            finally:
                thread.join(3)
            require(not thread.is_alive() and answer == [False], "approval denial did not reach its caller")
            return "HTTP denial unblocks a waiting approval"

        def jobs():
            built = job("/api/build", {"spec": "add two integers", "candidates": 2})
            require(bool(built["code"]), "build job lost result")
            request("POST", "/api/build/accept", {"note": "Synthetic smoke fixture only"})
            require("Smoke draft" in job("/api/write", {"brief": "Explain addition"})["markdown"], "writer job failed")
            job("/api/evolve", {})
            indexed = job("/api/index", {"spec": str(state.workspace), "refresh": False})
            require(not indexed["failures"], "local fixture index failed")
            reply = request("POST", "/api/mcp/call", {"name": "atlas_search", "arguments": {"query": "smoke_add"}})
            require(not reply.get("isError"), "MCP search failed")
            return "job plumbing; build/write/evolve computations are fixed stubs, local indexing is real"

        def validation():
            for path, body in (("/api/chat", {}), ("/api/feedback", {"conv_id": state.cid, "verdict": "unknown"}),
                               ("/api/tool", {"name": "missing"}), ("/api/approve", {"id": "missing"}),
                               ("/api/task", {}), ("/api/build", {}), ("/api/write", {}), ("/api/index", {}),
                               ("/api/skills/export", {}), ("/api/matrix/search", {"query": "addition"}),
                               ("/api/soup/remember", {}), ("/api/not-real", {})):
                request("POST", path, body, expected=400)
            request("GET", "/api/jobs/abcdef", expected=400)
            request("GET", "/api/search", expected=400)
            request("GET", "/api/not-real", expected=400)
            request("DELETE", "/api/not-real", expected=404)
            return "bad inputs and unavailable model-dependent matrix path fail explicitly"

        def guards():
            for method, path in (("GET", "/api/state"), ("POST", "/api/conversations"),
                                 ("DELETE", f"/api/conversations/{state.cid}")):
                body = {} if method == "POST" else None
                request(method, path, body, headers={"Host": "outside.invalid"}, expected=403)
                request(method, path, body, headers={"Origin": "https://outside.invalid"}, expected=403)
                request(method, path, body, authenticated=False, expected=403)
                request(method, path, body, headers={"X-Kgirl-Token": "invalid-smoke-token"}, expected=403)
            request("POST", "/api/conversations", {}, headers={"Content-Type": "text/plain"}, expected=415)
            for path in ("/static/../core.py", "/static/%2e%2e/core.py", "/static/../../mind/mind.py"):
                request("GET", path, expected=404)
            require(state.c.store.get(state.cid) is not None, "rejected DELETE still changed state")
            return "Host, Origin, launch token, JSON content type and traversal guards"

        def sse():
            connection = http.client.HTTPConnection("127.0.0.1", state.port, timeout=3)
            response = None
            try:
                token = getattr(state, "token", "")
                connection.request("GET", f"/api/events?token={token}", headers={"Origin": state.origin})
                response = connection.getresponse()
                require(response.status == 200 and response.getheader("Content-Type") == "text/event-stream",
                        "SSE handshake failed")

                def event():
                    for _ in range(30):
                        line = response.readline()
                        require(bool(line), "SSE stream closed before an event")
                        if line.startswith(b"data: "):
                            return json.loads(line[6:])
                    raise AssertionError("SSE stream produced no data event")

                require(event()["type"] == "hello", "SSE hello missing")
                state.a.bus.publish("smoke.ping", {"marker": "fixture"}, "smoke")
                value = event()
                require(value["type"] == "bus" and value["topic"] == "smoke.ping", "bus event did not reach SSE")
            finally:
                if connection.sock is not None:
                    try:
                        connection.sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                if response is not None:
                    response.close()
                connection.close()
                state.c.hub.emit("smoke_closed")
            return "hello plus an actual bus event streamed over HTTP"

        stages.check("server", setup)
        stages.check("token-bootstrap", bootstrap, needs=("server",))
        for name, action in (("reads", reads), ("conversation", conversation), ("memory", memory),
                             ("workspace", workspace), ("approvals", approvals), ("jobs", jobs)):
            stages.check(name, action, needs=("server",))
        stages.check("validation", validation, needs=("conversation",))
        stages.check("guards", guards, needs=("conversation",))
        stages.check("sse", sse, needs=("server",))
        if args.extended:
            def extended():
                for path in ("/api/helix/state", "/api/helix/history", "/api/helix/verify",
                             "/api/holo/status", "/api/holo/owners?digest=" + "0" * 64):
                    request("GET", path)
                require(not request("GET", "/api/lanes")["available"], "lane view read a live Porter database")
                request("POST", "/api/helix/run", {}, expected=400)
                request("POST", "/api/helix/run", {"task": "smoke", "critic_host": "https://outside.invalid"},
                        expected=400)
                request("POST", "/api/holo/query", {}, expected=400)
                # Scratch state has no chunk vectors: the job must report the missing input, not succeed.
                error = job("/api/holo/build", {}, expected_status="error")
                require("FileNotFoundError" in error and "chunk index" in error,
                        "missing holo input did not produce its expected error")
                return "Helix/Holo reads, missing-input errors, critic-host guard and isolated lane view"
            stages.check("helix-holo", extended, needs=("server",))
        stages.check("offline", lambda: require(not attempts, "a non-fixture network operation was attempted"))
        return stages.status()


if __name__ == "__main__":
    raise SystemExit(main())

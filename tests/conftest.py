"""A fake model server that speaks both Ollama's /api/chat and OpenAI's /v1/chat/completions.

`responder(model, messages, n)` decides the reply; `n` counts how many times this exact request has been
seen, so a test can make a repeat call answer differently (noise) or the same.
"""

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class FakeModels:
    def __init__(self):
        self.responder = lambda model, messages, n: "Answer: 0"
        self.seen = Counter()
        self.requests = []
        self.fail_with = None  # an HTTP status to return instead of a reply

    def reply(self, model, messages):
        key = json.dumps([model, messages], sort_keys=True)
        self.seen[key] += 1
        return self.responder(model, messages, self.seen[key])


def make_handler(fake: FakeModels):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/api/tags":
                self._send(200, {"models": [{"name": "fake:1b", "digest": "abc123"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
            if fake.fail_with:
                return self._send(fake.fail_with, {"error": "nope"})
            text = fake.reply(body["model"], body["messages"])
            if self.path == "/api/chat":
                self._send(200, {"message": {"role": "assistant", "content": text}, "done_reason": "stop",
                                 "prompt_eval_count": 10, "eval_count": len(text.split()),
                                 "prompt_eval_duration": 1_000_000, "eval_duration": 2_000_000})
            elif self.path == "/v1/chat/completions":
                self._send(200, {"choices": [{"message": {"role": "assistant", "content": text},
                                              "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 10, "completion_tokens": len(text.split())}})
            else:
                self._send(404, {"error": "not found"})
    return H


@pytest.fixture
def fake():
    f = FakeModels()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(f))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    f.url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield f
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = str(tmp_path / "home")
    monkeypatch.setenv("EVALPAIR_HOME", h)
    for k in ("EVALPAIR_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    return h


def write_suite(root: str, name: str, spec: str, items: list[dict]) -> str:
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "suite.toml"), "w") as f:
        f.write(spec)
    with open(os.path.join(d, "items.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return d


@pytest.fixture
def anyio_backend():
    return "asyncio"

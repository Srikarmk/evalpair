"""Model backends and the call cache.

Two backends: Ollama's native /api/chat, and any OpenAI-compatible /v1/chat/completions server (vLLM,
llama.cpp's server, LM Studio, or a hosted API). Every reply is cached in SQLite under a hash of
(backend, base URL, model, messages, options, tag), so re-running a suite or re-scoring it costs nothing.
`tag` is not sent to the model; it gives a repeat pass its own cache entry, so the repeat is a real second
call and the difference between the two passes is the noise floor.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request

THINK_RE = re.compile(r"<think>.*?</think>", re.S)
DEFAULT_URLS = {"ollama": "http://localhost:11434", "openai": "http://localhost:8000/v1"}


class BackendError(RuntimeError):
    pass


class CacheMiss(RuntimeError):
    pass


class Cache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.db.execute("create table if not exists calls (key text primary key, model text, response text)")
        self.db.commit()

    def get(self, key: str) -> dict | None:
        with self.lock:
            row = self.db.execute("select response from calls where key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, model: str, resp: dict) -> None:
        with self.lock:
            self.db.execute("insert or replace into calls values (?, ?, ?)", (key, model, json.dumps(resp)))
            self.db.commit()

    def __len__(self) -> int:
        with self.lock:
            return self.db.execute("select count(*) from calls").fetchone()[0]

    def close(self) -> None:
        with self.lock:
            self.db.close()


def _post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class Backend:
    """One model on one server. `chat` returns {"text", "prompt_tokens", "output_tokens", "seconds",
    "done_reason", "cached"}."""

    def __init__(self, kind: str, model: str, cache: Cache, base_url: str | None = None,
                 api_key: str | None = None, options: dict | None = None, think: bool | None = None,
                 cache_only: bool = False, timeout: float = 600, retries: int = 4):
        if kind not in DEFAULT_URLS:
            raise BackendError(f"backend must be one of {sorted(DEFAULT_URLS)}")
        self.kind, self.model, self.cache = kind, model, cache
        self.base_url = (base_url or DEFAULT_URLS[kind]).rstrip("/")
        self.api_key = api_key
        self.options = {"temperature": 0, "seed": 7, **(options or {})}
        self.think, self.cache_only, self.timeout, self.retries = think, cache_only, timeout, retries
        self.stats = {"hits": 0, "calls": 0}
        self.lock = threading.Lock()

    def key(self, messages: list[dict], max_tokens: int, tag: str) -> str:
        blob = json.dumps({"backend": self.kind, "url": self.base_url, "model": self.model, "messages": messages,
                           "options": self.options, "max_tokens": max_tokens, "think": self.think, "tag": tag},
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def body(self, messages: list[dict], max_tokens: int) -> tuple[str, dict, dict]:
        o = self.options
        if self.kind == "ollama":
            body = {"model": self.model, "messages": messages, "stream": False,
                    "options": {**o, "num_predict": max_tokens}}
            if self.think is not None:
                body["think"] = self.think
            return self.base_url + "/api/chat", body, {}
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": o.get("temperature", 0)}
        if "seed" in o:
            body["seed"] = o["seed"]
        if "top_p" in o:
            body["top_p"] = o["top_p"]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return self.base_url + "/chat/completions", body, headers

    def parse(self, data: dict, wall: float) -> dict:
        if self.kind == "ollama":
            raw = data["message"]["content"]
            return {"text": THINK_RE.sub("", raw).strip(), "raw": raw,
                    "done_reason": data.get("done_reason", ""),
                    "prompt_tokens": data.get("prompt_eval_count", 0), "output_tokens": data.get("eval_count", 0),
                    # server-side compute time, which excludes time spent queued behind other clients
                    "seconds": (data.get("prompt_eval_duration", 0) + data.get("eval_duration", 0)) / 1e9 or wall}
        choice = data["choices"][0]
        raw = choice["message"].get("content") or ""
        usage = data.get("usage") or {}
        return {"text": THINK_RE.sub("", raw).strip(), "raw": raw,
                "done_reason": {"length": "length"}.get(choice.get("finish_reason"), choice.get("finish_reason") or ""),
                "prompt_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0),
                "seconds": wall}

    def chat(self, messages: list[dict], max_tokens: int, tag: str = "") -> dict:
        key = self.key(messages, max_tokens, tag)
        hit = self.cache.get(key)
        if hit is not None:
            with self.lock:
                self.stats["hits"] += 1
            return {**hit, "cached": True}
        if self.cache_only:
            raise CacheMiss(key)
        url, body, headers = self.body(messages, max_tokens)
        for attempt in range(self.retries + 1):
            try:
                t0 = time.time()
                data = _post(url, body, headers, self.timeout)
                wall = time.time() - t0
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                # a request the server rejects won't succeed on retry; rate limits and 5xx might
                if e.code < 500 and e.code != 429 or attempt == self.retries:
                    raise BackendError(f"{self.kind} {self.model}: HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == self.retries:
                    raise BackendError(f"{self.kind} at {self.base_url} unreachable: {e}") from None
            time.sleep(min(2 ** attempt, 30))
        try:
            resp = self.parse(data, wall)
        except (KeyError, IndexError, TypeError) as e:
            raise BackendError(f"{self.kind} {self.model}: unexpected reply shape ({e!r})") from None
        self.cache.put(key, self.model, resp)
        with self.lock:
            self.stats["calls"] += 1
        return {**resp, "cached": False}

    def describe(self) -> dict:
        """What identifies the model, for the run record. For Ollama, the model digest."""
        out = {"backend": self.kind, "model": self.model, "base_url": self.base_url}
        if self.kind == "ollama":
            try:
                with urllib.request.urlopen(self.base_url + "/api/tags", timeout=30) as r:
                    tags = json.load(r)
                m = next((m for m in tags.get("models", []) if m.get("name") in (self.model, self.model + ":latest")), {})
                out["digest"] = m.get("digest", "")
            except (urllib.error.URLError, TimeoutError, ValueError):
                out["digest"] = ""
        return out

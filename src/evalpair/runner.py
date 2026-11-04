"""Run a suite against one model and save the run.

Items are taken in file order (bundled suites are already a seeded random draw), so `limit=50` is the
same 50 items every time and two limited runs still pair up. With `repeat`, every item is sent a second
time as a separate call; how often the answer changes between the two passes is the run's noise floor.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from . import __version__, store
from .backends import Backend
from .suites import Suite

REPEAT_TAG = "repeat"


def _row(suite: Suite, item: dict, pass_no: int, reply: dict) -> dict:
    c = suite.check(reply["text"], item)
    return {"id": item["id"], "pass": pass_no, "group": item.get(suite.group) if suite.group else None,
            "answer": c.answer, "score": c.score, "how": c.how, "text": reply["text"],
            "truncated": reply.get("done_reason") == "length", "prompt_tokens": reply.get("prompt_tokens", 0),
            "output_tokens": reply.get("output_tokens", 0), "seconds": round(reply.get("seconds", 0.0), 3),
            "cached": reply.get("cached", False)}


def run(suite: Suite, backend: Backend, home: str, limit: int | None = None, repeat: bool = False,
        workers: int = 1, label: str | None = None,
        progress: Callable[[int, int], None] | None = None) -> dict:
    items = suite.items[:limit] if limit else list(suite.items)
    jobs = [(it, 1) for it in items] + ([(it, 2) for it in items] if repeat else [])
    started = dt.datetime.now(dt.timezone.utc)
    run_id = store.check_id(label) if label else store.slug(
        f"{suite.name}-{backend.model}-{started.strftime('%Y%m%d-%H%M%S')}")
    t0 = time.time()
    done, lock = [0], threading.Lock()

    def one(job):
        it, p = job
        reply = backend.chat(suite.render(it), suite.max_tokens, REPEAT_TAG if p == 2 else "")
        with lock:
            done[0] += 1
            n = done[0]
        if progress:
            progress(n, len(jobs))
        return _row(suite, it, p, reply)

    if workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            rows = list(ex.map(one, jobs))
    else:
        rows = [one(j) for j in jobs]
    meta = {
        "run": run_id, "suite": suite.name, "suite_hash": suite.hash, "checker": suite.checker,
        **backend.describe(), "options": backend.options, "think": backend.think, "max_tokens": suite.max_tokens,
        "items": len(items), "repeat": repeat, "calls": backend.stats["calls"], "cache_hits": backend.stats["hits"],
        "started": started.isoformat(timespec="seconds"),
        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - t0, 1), "evalpair": __version__,
    }
    store.save(home, meta, rows)
    return meta

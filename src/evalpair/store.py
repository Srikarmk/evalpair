"""Where evalpair keeps things: `$EVALPAIR_HOME`, or ~/.evalpair.

    <home>/cache.sqlite          every model reply, keyed by request
    <home>/runs/<run>/meta.json  what was run: suite and its hash, model, backend, options, counts
    <home>/runs/<run>/rows.jsonl one line per item and pass: the reply, the parsed answer, the score
    <home>/suites/<name>/        your own suites
"""

from __future__ import annotations

import json
import os
import re

RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")


class RunError(ValueError):
    pass


def home(path: str | None = None) -> str:
    return os.path.abspath(os.path.expanduser(path or os.environ.get("EVALPAIR_HOME") or "~/.evalpair"))


def cache_path(h: str) -> str:
    return os.path.join(h, "cache.sqlite")


def runs_dir(h: str) -> str:
    return os.path.join(h, "runs")


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")[:60] or "run"


def check_id(run_id: str) -> str:
    if not RUN_ID.match(run_id or "") or ".." in run_id:
        raise RunError(f"bad run id {run_id!r}: use letters, digits, '.', '_' and '-'")
    return run_id


def run_path(h: str, run_id: str) -> str:
    return os.path.join(runs_dir(h), check_id(run_id))


def save(h: str, meta: dict, rows: list[dict]) -> str:
    d = run_path(h, meta["run"])
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, "rows.jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, os.path.join(d, "rows.jsonl"))
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    return d


def load(h: str, run_id: str) -> tuple[dict, list[dict]]:
    d = run_path(h, run_id)
    if not os.path.isfile(os.path.join(d, "meta.json")):
        known = [m["run"] for m in list_runs(h)][:10]
        raise RunError(f"no run {run_id!r}" + (f"; recent: {', '.join(known)}" if known else ""))
    with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(d, "rows.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return meta, rows


def list_runs(h: str, suite: str | None = None) -> list[dict]:
    """Newest first."""
    out = []
    d = runs_dir(h)
    if not os.path.isdir(d):
        return out
    for name in os.listdir(d):
        p = os.path.join(d, name, "meta.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            if suite is None or m.get("suite") == suite:
                out.append(m)
    return sorted(out, key=lambda m: m.get("finished", ""), reverse=True)

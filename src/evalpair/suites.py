"""Eval suites: a folder with `suite.toml` (prompt, checker, options) and `items.jsonl` (one item per line).

Bundled suites live in the package; your own go in `<home>/suites/<name>/` or are given by path. A suite's
hash covers its spec and items, so two runs are only compared when they answered the same questions.

suite.toml:

    description = "What the suite measures"
    checker = "number"            # number | choice | exact | contains | regex | json_fields
    max_tokens = 1024             # optional; temperature, seed and num_ctx are run options
    system = "..."                # optional
    user = "Problem: {question}"  # {field} is filled from the item; {{ and }} for literal braces
    group = "subject"             # optional item field to stratify the bootstrap by
    [checker_options]             # optional, e.g. pattern = 'Label: (\\w+)' for regex
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass, field
from importlib import resources

from .checkers import CHECKERS, Check

ALLOWED = {"description", "checker", "max_tokens", "system", "user", "group", "checker_options", "license", "source"}


class SuiteError(ValueError):
    pass


@dataclass
class Suite:
    name: str
    path: str
    description: str
    checker: str
    user: str
    system: str | None
    max_tokens: int
    group: str | None
    checker_options: dict
    items: list[dict]
    license: str | None = None
    source: str | None = None
    hash: str = field(default="")

    def render(self, item: dict) -> list[dict]:
        fields = {k: (v if isinstance(v, str) else _show(v)) for k, v in item.items()}
        try:
            msgs = [{"role": "user", "content": self.user.format_map(fields)}]
            if self.system:
                msgs.insert(0, {"role": "system", "content": self.system.format_map(fields)})
        except KeyError as e:
            raise SuiteError(f"suite {self.name}: item {item.get('id')} has no field {e} used in the prompt") from None
        return msgs

    def check(self, text: str, item: dict) -> Check:
        return CHECKERS[self.checker](text, item, self.checker_options)

    def summary(self) -> dict:
        return {"name": self.name, "description": self.description, "checker": self.checker,
                "items": len(self.items), "hash": self.hash[:12], "path": self.path,
                "license": self.license, "source": self.source}


def _show(v) -> str:
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return "\n".join(f"{'ABCDEFGHIJ'[i]}. {x}" for i, x in enumerate(v))
    return json.dumps(v, indent=2, ensure_ascii=False)


def bundled_dir() -> str:
    return str(resources.files("evalpair") / "suites")


def suite_dirs(home: str) -> list[str]:
    return [bundled_dir(), os.path.join(home, "suites")]


def load(path: str, name: str | None = None) -> Suite:
    spec_path = os.path.join(path, "suite.toml")
    items_path = os.path.join(path, "items.jsonl")
    if not os.path.isfile(spec_path) or not os.path.isfile(items_path):
        raise SuiteError(f"{path}: a suite folder needs suite.toml and items.jsonl")
    with open(spec_path, "rb") as f:
        raw = f.read()
    spec = tomllib.loads(raw.decode())
    unknown = set(spec) - ALLOWED
    if unknown:
        raise SuiteError(f"{spec_path}: unknown keys {sorted(unknown)}")
    if spec.get("checker") not in CHECKERS:
        raise SuiteError(f"{spec_path}: checker must be one of {sorted(CHECKERS)}")
    if not spec.get("user"):
        raise SuiteError(f"{spec_path}: needs a 'user' prompt template")
    items, ids = [], set()
    with open(items_path, encoding="utf-8") as f:
        item_bytes = f.read()
    for n, line in enumerate(item_bytes.splitlines(), 1):
        if not line.strip():
            continue
        try:
            it = json.loads(line)
        except json.JSONDecodeError as e:
            raise SuiteError(f"{items_path}:{n}: not JSON ({e.msg})") from None
        if not isinstance(it, dict) or "id" not in it or "gold" not in it:
            raise SuiteError(f"{items_path}:{n}: every item needs an 'id' and a 'gold'")
        it["id"] = str(it["id"])
        if it["id"] in ids:
            raise SuiteError(f"{items_path}:{n}: duplicate id {it['id']}")
        ids.add(it["id"])
        items.append(it)
    if not items:
        raise SuiteError(f"{items_path}: no items")
    h = hashlib.sha256(raw + b"\0" + item_bytes.encode()).hexdigest()
    s = Suite(name=name or os.path.basename(os.path.normpath(path)), path=path,
              description=spec.get("description", ""), checker=spec["checker"], user=spec["user"],
              system=spec.get("system"), max_tokens=int(spec.get("max_tokens", 1024)), group=spec.get("group"),
              checker_options=spec.get("checker_options", {}), items=items, license=spec.get("license"),
              source=spec.get("source"), hash=h)
    s.render(items[0])  # fail now, not mid-run, if the template names a missing field
    return s


def available(home: str) -> dict[str, str]:
    """Suite name -> folder. A suite in your home shadows a bundled one of the same name."""
    out = {}
    for d in suite_dirs(home):
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if os.path.isfile(os.path.join(d, name, "suite.toml")):
                out[name] = os.path.join(d, name)
    return out


def find(ref: str, home: str) -> Suite:
    """A suite by name, or by path to its folder."""
    if os.path.isdir(ref):
        return load(ref)
    known = available(home)
    if ref not in known:
        raise SuiteError(f"no suite named {ref!r}; known: {', '.join(known) or 'none'}")
    return load(known[ref], ref)

"""Command line: evalpair suites | run | runs | show | compare | mcp."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__, analysis, store, suites
from .backends import Backend, BackendError, Cache

API_KEY_ENV = ("EVALPAIR_API_KEY", "OPENAI_API_KEY")


def api_key() -> str | None:
    return next((os.environ[k] for k in API_KEY_ENV if os.environ.get(k)), None)


def make_backend(h: str, kind: str, model: str, base_url: str | None, temperature: float, seed: int | None,
                 num_ctx: int | None, think: bool | None, cache_only: bool = False) -> Backend:
    opts = {"temperature": temperature}
    if seed is not None:
        opts["seed"] = seed
    if num_ctx and kind == "ollama":
        opts["num_ctx"] = num_ctx
    return Backend(kind, model, Cache(store.cache_path(h)), base_url, api_key(), opts, think, cache_only)


def _print(obj, as_json: bool, text_fn) -> None:
    if as_json:
        print(json.dumps(obj, indent=1, ensure_ascii=False))
    else:
        print(text_fn(obj))


def text_summary(s: dict) -> str:
    lines = [f"{s['run']}  ({s['model']} on {s['suite']}, {s['items']} items)",
             f"  score        {s['score']['text']}",
             f"  no answer    {s['no_answer']}    truncated {s['truncated']}    "
             f"mean output {s['mean_output_tokens']} tokens"]
    nf = s.get("noise_floor")
    if nf:
        lines.append(f"  noise floor  {nf['answers_changed']} of {nf['items']} answers changed on a repeat "
                     f"({nf['change_rate']['text']})")
    for g, v in (s.get("by_group") or {}).items():
        lines.append(f"    {g:<28} {100 * v['score']:5.1f}%  ({v['items']})")
    if s["failures"]:
        lines.append("  failures (unparsed and truncated first):")
        for f in s["failures"]:
            end = " ".join(f["reply_end"].split())[-160:]  # newlines and tabs as single spaces
            lines.append(f"    {f['id']}: answer {f['answer']!r} via {f['how']}"
                         f"{', truncated' if f['truncated'] else ''}  …{end}")
    return "\n".join(lines)


def text_compare(c: dict) -> str:
    if not c.get("items"):
        return "\n".join(c["notes"])
    lines = [f"A            {c['a']['run']}: {c['a']['score']['text']}",
             f"B            {c['b']['run']}: {c['b']['score']['text']}",
             f"B − A        {c['difference']['text']}",
             f"changed      {c['answers_changed']['text']} of {c['items']} items "
             f"({c['worse_in_b']} worse in B, {c['better_in_b']} better)"]
    for k, f in (c.get("noise_floor") or {}).items():
        lines.append(f"floor ({k.upper()})    {f['change_rate']['text']}; excess {f['excess_over_floor']['text']}")
    lines += ["", c["verdict"]] + [f"Note: {n}" for n in c["notes"]]
    for kind, label in (("worse_in_b", "worse in B"), ("better_in_b", "better in B")):
        for e in c["examples"][kind]:
            lines.append(f"  {label + ':':<13}{e['id']}  {e['a_answer']!r} → {e['b_answer']!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="evalpair", description=__doc__)
    p.add_argument("--version", action="version", version=f"evalpair {__version__}")
    p.add_argument("--home", help="data folder (default $EVALPAIR_HOME or ~/.evalpair)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("suites", help="list the suites you can run").add_argument("--json", action="store_true")

    r = sub.add_parser("run", help="run a suite against a model")
    r.add_argument("suite", help="a suite name, or a path to a suite folder")
    r.add_argument("--model", "-m", required=True)
    r.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    r.add_argument("--base-url", help="server URL (default: Ollama on localhost, or http://localhost:8000/v1)")
    r.add_argument("--limit", "-n", type=int, help="only the first N items")
    r.add_argument("--repeat", action="store_true", help="send every item twice to measure the noise floor")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--label", help="run id (default: suite-model-timestamp)")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--num-ctx", type=int, help="Ollama context window")
    r.add_argument("--think", choices=["on", "off"], help="Ollama thinking for models that support it")
    r.add_argument("--json", action="store_true")

    ls = sub.add_parser("runs", help="list saved runs, newest first")
    ls.add_argument("--suite")
    ls.add_argument("--json", action="store_true")

    sh = sub.add_parser("show", help="summarise a run, with its failures")
    sh.add_argument("run")
    sh.add_argument("--failures", type=int, default=5)
    sh.add_argument("--json", action="store_true")

    c = sub.add_parser("compare", help="compare two runs item by item")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--examples", type=int, default=5)
    c.add_argument("--json", action="store_true")

    sub.add_parser("mcp", help="run the MCP server on stdio")

    args = p.parse_args(argv)
    h = store.home(args.home)
    try:
        if args.cmd == "suites":
            rows = [suites.find(n, h).summary() for n in suites.available(h)]
            _print(rows, args.json, lambda rs: "\n".join(
                f"{s['name']:<22} {s['items']:>5} items  {s['checker']:<12} {s['description']}" for s in rs))
        elif args.cmd == "run":
            suite = suites.find(args.suite, h)
            think = None if args.think is None else args.think == "on"
            be = make_backend(h, args.backend, args.model, args.base_url, args.temperature, args.seed,
                              args.num_ctx, think)

            tty = sys.stderr.isatty()

            def progress(done, total):
                if tty and (done == total or done % 10 == 0):
                    print(f"\r{done}/{total}", end="", file=sys.stderr, flush=True)
                elif not tty and (done == total or done % 50 == 0):  # a log file: whole lines, fewer of them
                    print(f"{done}/{total}", file=sys.stderr, flush=True)
            from .runner import run
            meta = run(suite, be, h, args.limit, args.repeat, args.workers, args.label, progress)
            if tty:
                print(file=sys.stderr)
            m, rows = store.load(h, meta["run"])
            _print(analysis.summarize(m, rows), args.json, text_summary)
        elif args.cmd == "runs":
            rows = store.list_runs(h, args.suite)
            _print(rows, args.json, lambda rs: "\n".join(
                f"{m['run']:<50} {m['model']:<24} {m['items']:>5} items{'  +repeat' if m['repeat'] else ''}"
                for m in rs) or "no runs yet")
        elif args.cmd == "show":
            m, rows = store.load(h, args.run)
            _print(analysis.summarize(m, rows, args.failures), args.json, text_summary)
        elif args.cmd == "compare":
            ma, ra = store.load(h, args.a)
            mb, rb = store.load(h, args.b)
            _print(analysis.compare(ma, ra, mb, rb, args.examples), args.json, text_compare)
        elif args.cmd == "mcp":
            from .server import mcp
            mcp.run()
    except (suites.SuiteError, store.RunError, BackendError) as e:
        print(f"evalpair: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\ninterrupted; replies so far are cached, so the same command resumes", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()

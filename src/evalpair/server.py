"""MCP server front end (stdio). Four read-only tools over saved runs, and one that runs a suite."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import anyio
import anyio.from_thread
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from . import __version__, analysis, store, suites
from .backends import BackendError
from .cli import make_backend

INSTRUCTIONS = (
    "evalpair runs an eval suite (questions with known answers, a prompt template and a deterministic checker) "
    "against a model on Ollama or any OpenAI-compatible server, caches every call, and compares two runs item by "
    "item. Scores come with 95% bootstrap intervals over items. A comparison says whether a score difference is "
    "distinguishable from zero and whether answers changed more than the noise floor (the same model asked twice, "
    "measured when a run is made with repeat=true). Prefer a comparison's verdict over reading two scores side by "
    "side. Read a run's failures before trusting its score: unparsed or truncated replies are listed first."
)

mcp = MCPServer("evalpair", instructions=INSTRUCTIONS, version=__version__)
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
RUNS = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True)

RunId = Annotated[str, Field(description="A run id from list_runs or run_eval.", min_length=1, max_length=120)]


def home() -> str:
    return store.home()


def _guard(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except (suites.SuiteError, store.RunError, BackendError) as e:
        raise ToolError(str(e)) from None


@mcp.tool(title="List eval suites", annotations=READ_ONLY)
def list_suites() -> dict[str, Any]:
    """Use to see which eval suites can be run: name, item count, checker and what each measures."""
    h = home()
    return {"suites": _guard(lambda: [suites.find(n, h).summary() for n in suites.available(h)]),
            "your_suites_folder": f"{h}/suites"}


@mcp.tool(title="List saved runs", annotations=READ_ONLY)
def list_runs(
    suite: Annotated[str | None, Field(description="Only runs of this suite.")] = None,
    limit: Annotated[int, Field(description="How many, newest first.", ge=1, le=200)] = 20,
) -> dict[str, Any]:
    """Use to find earlier runs to show or compare: run id, suite, model, item count, whether it has a repeat pass."""
    keep = ("run", "suite", "model", "backend", "items", "repeat", "finished", "digest")
    return {"runs": [{k: m.get(k) for k in keep} for m in store.list_runs(home(), suite)[:limit]]}


@mcp.tool(title="Run an eval suite", annotations=RUNS)
async def run_eval(
    suite: Annotated[str, Field(description="A suite name from list_suites, or a path to a suite folder.")],
    model: Annotated[str, Field(description="Model name as the server knows it, e.g. 'llama3.1:8b' on Ollama.")],
    ctx: Context,
    backend: Annotated[Literal["ollama", "openai"], Field(
        description="'ollama' for Ollama's API; 'openai' for any OpenAI-compatible server.")] = "ollama",
    base_url: Annotated[str | None, Field(
        description="Server URL. Defaults: http://localhost:11434 (ollama), http://localhost:8000/v1 (openai).")] = None,
    limit: Annotated[int | None, Field(description="Only the first N items; start small, runs take a while.",
                                       ge=1)] = None,
    repeat: Annotated[bool, Field(description="Send every item twice to measure the noise floor.")] = False,
    temperature: Annotated[float, Field(ge=0, le=2)] = 0.0,
    seed: Annotated[int | None, Field(description="Sampling seed, where the server supports one.")] = 7,
    think: Annotated[bool | None, Field(description="Ollama thinking on or off, for models that support it.")] = None,
    label: Annotated[str | None, Field(description="Run id to save under (letters, digits, . _ -).")] = None,
) -> dict[str, Any]:
    """Use to run a suite against one model and save the run. Returns its score with an interval and its first failures. Calls the model; cached replies are reused."""
    from .runner import run
    h = home()
    s = _guard(suites.find, suite, h)
    be = _guard(make_backend, h, backend, model, base_url, temperature, seed, None, think)
    n = min(limit or len(s.items), len(s.items)) * (2 if repeat else 1)

    def progress(done, total):
        if done == total or done % 5 == 0:
            anyio.from_thread.run(ctx.report_progress, done, total, f"{done}/{total} calls")

    meta = await anyio.to_thread.run_sync(lambda: _guard(run, s, be, h, limit, repeat, 1, label, progress))
    m, rows = store.load(h, meta["run"])
    return {**analysis.summarize(m, rows), "calls_made": meta["calls"], "calls_planned": n}


@mcp.tool(title="Show a run", annotations=READ_ONLY)
def show_run(
    run: RunId,
    failures: Annotated[int, Field(description="How many failed items to include.", ge=0, le=50)] = 5,
) -> dict[str, Any]:
    """Use to see a saved run's score with its interval, the noise floor if it has a repeat, and the end of failed replies."""
    m, rows = _guard(store.load, home(), run)
    return analysis.summarize(m, rows, failures)


@mcp.tool(title="Compare two runs", annotations=READ_ONLY)
def compare_runs(
    a: Annotated[str, Field(description="The baseline run id.", min_length=1, max_length=120)],
    b: Annotated[str, Field(description="The run to compare against it.", min_length=1, max_length=120)],
    examples: Annotated[int, Field(description="Items to show that got worse and better.", ge=0, le=20)] = 5,
) -> dict[str, Any]:
    """Use to compare two runs of the same suite item by item: score difference with a paired interval, flips each way, and whether answers moved more than the noise floor."""
    h = home()
    ma, ra = _guard(store.load, h, a)
    mb, rb = _guard(store.load, h, b)
    return analysis.compare(ma, ra, mb, rb, examples)

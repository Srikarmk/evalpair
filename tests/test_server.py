import json

import pytest
from mcp import Client

from evalpair.server import mcp

pytestmark = pytest.mark.anyio

TOOLS = {"list_suites", "list_runs", "run_eval", "show_run", "compare_runs"}


@pytest.fixture
async def client(home):
    async with Client(mcp, raise_exceptions=True) as c:
        yield c


async def test_tools_and_annotations(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == TOOLS
    for name, t in tools.items():
        assert t.description and t.description.startswith("Use ") and len(t.description) < 300
        assert t.annotations.read_only_hint is (name != "run_eval")
        assert t.annotations.destructive_hint is False
    assert tools["run_eval"].input_schema["required"] == ["suite", "model"]


async def test_list_suites(client):
    res = await client.call_tool("list_suites", {})
    names = {s["name"] for s in res.structured_content["suites"]}
    assert {"gsm8k-160", "mmlu-hs-160", "e2e-extract-150"} <= names


async def test_run_show_compare_through_mcp(client, fake):
    fake.responder = lambda model, messages, n: "Answer: 18" if model == "a" else "Answer: 0"
    ra = await client.call_tool("run_eval", {"suite": "gsm8k-160", "model": "a", "base_url": fake.url,
                                             "limit": 12, "repeat": True, "label": "run-a"})
    assert not ra.is_error
    a = ra.structured_content
    assert a["run"] == "run-a" and a["items"] == 12 and a["calls_made"] == 24 and a["noise_floor"]["answers_changed"] == 0
    rb = await client.call_tool("run_eval", {"suite": "gsm8k-160", "model": "b", "base_url": fake.url,
                                             "limit": 12, "label": "run-b"})
    assert rb.structured_content["score"]["mean"] == 0
    runs = (await client.call_tool("list_runs", {})).structured_content["runs"]
    assert {r["run"] for r in runs} == {"run-a", "run-b"}
    shown = (await client.call_tool("show_run", {"run": "run-b", "failures": 2})).structured_content
    assert len(shown["failures"]) == 2 and shown["failures"][0]["answer"] == "0"
    c = (await client.call_tool("compare_runs", {"a": "run-a", "b": "run-b"})).structured_content
    assert c["items"] == 12 and "verdict" in c
    json.dumps(c)  # everything a client gets is plain JSON


async def test_errors_come_back_as_tool_errors(home, monkeypatch):
    monkeypatch.setattr("evalpair.backends.time.sleep", lambda s: None)  # skip the retry backoff
    async with Client(mcp, raise_exceptions=False) as c:
        res = await c.call_tool("show_run", {"run": "nope"})
        assert res.is_error and "no run" in res.content[0].text
        res = await c.call_tool("run_eval", {"suite": "nope", "model": "m"})
        assert res.is_error and "no suite named" in res.content[0].text
        res = await c.call_tool("run_eval", {"suite": "gsm8k-160", "model": "m", "base_url": "http://127.0.0.1:9",
                                             "limit": 1})
        assert res.is_error and "unreachable" in res.content[0].text

import re

import pytest

from conftest import write_suite
from evalpair import analysis, store, suites
from evalpair.backends import Backend, BackendError, Cache, CacheMiss
from evalpair.runner import run

N = 60


def number_suite(tmp_path):
    items = [{"id": f"q{i:02d}", "question": f"What is {i} + 1?", "gold": str(i + 1), "kind": "odd" if i % 2 else "even"}
             for i in range(N)]
    d = write_suite(str(tmp_path), "add", 'checker = "number"\nmax_tokens = 64\ngroup = "kind"\n'
                    'user = "What is the answer? {question}"\n', items)
    return suites.load(d)


def model_says(right_until: int, noisy: set[int] = frozenset()):
    """Right on questions below `right_until`; on `noisy` questions a repeat of the request answers differently."""
    def respond(model, messages, n):
        i = int(re.search(r"What is (\d+)", messages[-1]["content"]).group(1))
        ans = i + 1 if i < right_until else i + 2
        if i in noisy and n > 1:
            ans += 10
        return f"Working...\nAnswer: {ans}"
    return respond


def backend(fake, home, model="fake:1b", kind="ollama", **kw):
    url = fake.url if kind == "ollama" else fake.url + "/v1"
    return Backend(kind, model, Cache(store.cache_path(home)), url, **kw)


def test_run_scores_saves_and_caches(fake, home, tmp_path):
    s = number_suite(tmp_path)
    fake.responder = model_says(45)
    meta = run(s, backend(fake, home), home, label="first")
    assert meta["calls"] == N and meta["cache_hits"] == 0 and meta["digest"] == "abc123"
    m, rows = store.load(home, "first")
    summ = analysis.summarize(m, rows)
    assert summ["score"]["mean"] == 45 / N and summ["score"]["lo"] < 0.75 < summ["score"]["hi"]
    assert summ["by_group"] == {"even": {"items": 30, "score": 0.7667}, "odd": {"items": 30, "score": 0.7333}}
    assert summ["noise_floor"] is None and len(summ["failures"]) == 5
    body = fake.requests[0]["body"]
    assert body["options"] == {"temperature": 0, "seed": 7, "num_predict": 64} and body["stream"] is False
    again = run(s, backend(fake, home), home, label="second")
    assert again["calls"] == 0 and again["cache_hits"] == N and len(fake.requests) == N


def test_cache_only_and_limit(fake, home, tmp_path):
    s = number_suite(tmp_path)
    fake.responder = model_says(N)
    run(s, backend(fake, home), home, limit=10, label="ten")
    assert store.load(home, "ten")[0]["items"] == 10
    run(s, backend(fake, home, cache_only=True), home, limit=10, label="ten-again")
    with pytest.raises(CacheMiss):
        run(s, backend(fake, home, cache_only=True), home, limit=11)


def test_repeat_measures_the_noise_floor(fake, home, tmp_path):
    s = number_suite(tmp_path)
    fake.responder = model_says(N, noisy={1, 2, 3})
    meta = run(s, backend(fake, home), home, repeat=True, label="rep")
    assert meta["calls"] == 2 * N  # the repeat is a real second call, not a cache hit
    nf = analysis.summarize(*store.load(home, "rep"))["noise_floor"]
    assert nf["items"] == N and nf["answers_changed"] == 3 and nf["change_rate"]["mean"] == 3 / N


def test_compare_finds_a_real_drop_and_names_the_items(fake, home, tmp_path):
    s = number_suite(tmp_path)
    fake.responder = model_says(N, noisy={0})
    run(s, backend(fake, home, model="good"), home, repeat=True, label="a")
    fake.responder = model_says(30)
    run(s, backend(fake, home, model="worse"), home, label="b")
    c = analysis.compare(*store.load(home, "a"), *store.load(home, "b"))
    assert c["items"] == N and c["worse_in_b"] == 30 and c["better_in_b"] == 0
    assert c["difference"]["mean"] == -0.5 and c["difference"]["hi"] < 0
    assert c["verdict"].startswith("B scores lower than A: -50.0 pp")
    assert "the change is beyond noise" in c["verdict"]
    assert [e["id"] for e in c["examples"]["worse_in_b"]] == ["q30", "q31", "q32", "q33", "q34"]
    assert c["notes"] == []


def test_compare_same_model_twice_is_no_difference(fake, home, tmp_path):
    s = number_suite(tmp_path)
    fake.responder = model_says(40, noisy={5, 50})
    run(s, backend(fake, home), home, repeat=True, label="x")
    # a second run whose answers differ only by the model's own noise
    run(s, backend(fake, home, options={"temperature": 0, "seed": 8}), home, label="y")
    c = analysis.compare(*store.load(home, "x"), *store.load(home, "y"))
    assert c["difference"]["lo"] <= 0 <= c["difference"]["hi"]
    assert c["verdict"].startswith("No clear difference in score")
    assert "not clearly beyond noise" in c["verdict"]


def test_compare_warns_about_different_suites(fake, home, tmp_path):
    s = number_suite(tmp_path)
    fake.responder = model_says(N)
    run(s, backend(fake, home), home, limit=20, label="short")
    s2 = suites.load(write_suite(str(tmp_path), "add2", 'checker = "number"\nuser = "Q: {question}"\n',
                                 [{"id": f"q{i:02d}", "question": f"What is {i} + 1?", "gold": str(i + 1)}
                                  for i in range(10, 40)]))
    run(s2, backend(fake, home), home, label="other")
    c = analysis.compare(*store.load(home, "short"), *store.load(home, "other"))
    assert c["items"] == 10 and len(c["notes"]) == 2 and "different versions of the suite" in c["notes"][0]
    assert "Neither run has a repeat pass" in c["verdict"]


def test_openai_backend_sends_bearer_and_max_tokens(fake, home, tmp_path):
    s = number_suite(tmp_path)
    fake.responder = model_says(N)
    run(s, backend(fake, home, kind="openai", model="gpt-x", api_key="sk-test"), home, limit=3, label="oa")
    req = fake.requests[0]
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer sk-test"
    assert req["body"]["max_tokens"] == 64 and req["body"]["seed"] == 7 and "options" not in req["body"]
    assert analysis.summarize(*store.load(home, "oa"))["score"]["mean"] == 1


def test_client_errors_are_not_retried(fake, home, tmp_path):
    s = number_suite(tmp_path)
    fake.fail_with = 404
    with pytest.raises(BackendError, match="HTTP 404"):
        run(s, backend(fake, home), home, limit=1)
    assert len(fake.requests) == 1


def test_unreachable_server(home, tmp_path):
    s = number_suite(tmp_path)
    be = Backend("ollama", "m", Cache(store.cache_path(home)), "http://127.0.0.1:9", retries=0)
    with pytest.raises(BackendError, match="unreachable"):
        run(s, be, home, limit=1)


def test_bad_run_ids(home):
    for bad in ["../x", "", "a/b", ".hidden"]:
        with pytest.raises(store.RunError):
            store.check_id(bad)
    with pytest.raises(store.RunError, match="no run"):
        store.load(home, "missing")

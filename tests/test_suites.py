import pytest

from conftest import write_suite
from evalpair import suites


def test_bundled_suites_load(home):
    names = suites.available(home)
    assert {"gsm8k-160", "mmlu-hs-160", "e2e-extract-150"} <= set(names)
    for n in names:
        s = suites.find(n, home)
        assert s.items and len({it["id"] for it in s.items}) == len(s.items)
        assert len(s.hash) == 64 and s.license and s.source


def test_bundled_prompts_match_the_studies(home):
    """The studies' prompts, word for word, so a bundled run can be checked against their results."""
    m = suites.find("mmlu-hs-160", home)
    msg = m.render(m.items[0])[0]["content"]
    assert msg.startswith("Answer the following multiple-choice question about high school ")
    assert '"Answer: <letter>".\n\nQuestion: ' in msg and "\n\nA. " in msg and "\nD. " in msg
    g = suites.find("gsm8k-160", home)
    assert g.render(g.items[0])[0]["content"].startswith("Solve the following math problem. Reason step by step")
    e = suites.find("e2e-extract-150", home)
    body = e.render(e.items[0])[0]["content"]
    assert body.endswith("}") and 'Record:\n{\n  "name": ' in body


def test_user_suite_by_path_and_by_name(home, tmp_path):
    d = write_suite(str(tmp_path), "capitals", 'checker = "exact"\nsystem = "Be brief."\nuser = "Capital of {country}?"\n',
                    [{"id": 1, "country": "France", "gold": "Paris"}])
    s = suites.find(d, home)
    assert s.render(s.items[0]) == [{"role": "system", "content": "Be brief."},
                                    {"role": "user", "content": "Capital of France?"}]
    assert s.items[0]["id"] == "1"
    write_suite(f"{home}/suites", "capitals", 'checker = "exact"\nuser = "{country}"\n',
                [{"id": "x", "country": "Peru", "gold": "Lima"}])
    assert suites.find("capitals", home).checker == "exact"


@pytest.mark.parametrize("spec,items,msg", [
    ('checker = "vibes"\nuser = "x"\n', [{"id": 1, "gold": 1}], "checker must be one of"),
    ('checker = "exact"\n', [{"id": 1, "gold": 1}], "needs a 'user'"),
    ('checker = "exact"\nuser = "x"\ntemperature = 1\n', [{"id": 1, "gold": 1}], "unknown keys"),
    ('checker = "exact"\nuser = "x"\n', [{"id": 1}], "needs an 'id' and a 'gold'"),
    ('checker = "exact"\nuser = "x"\n', [{"id": 1, "gold": 1}, {"id": 1, "gold": 2}], "duplicate id"),
    ('checker = "exact"\nuser = "{missing}"\n', [{"id": 1, "gold": 1}], "has no field 'missing'"),
])
def test_bad_suites_are_rejected(tmp_path, home, spec, items, msg):
    d = write_suite(str(tmp_path), "bad", spec, items)
    with pytest.raises(suites.SuiteError, match=msg):
        suites.load(d)


def test_unknown_suite_name(home):
    with pytest.raises(suites.SuiteError, match="no suite named"):
        suites.find("nope", home)


def test_hash_changes_with_items(tmp_path):
    a = write_suite(str(tmp_path), "a", 'checker = "exact"\nuser = "{q}"\n', [{"id": 1, "q": "x", "gold": "y"}])
    h1 = suites.load(a).hash
    write_suite(str(tmp_path), "a", 'checker = "exact"\nuser = "{q}"\n', [{"id": 1, "q": "x", "gold": "z"}])
    assert suites.load(a).hash != h1

import json

import pytest

from conftest import write_suite
from evalpair import __version__
from evalpair.cli import main


def test_suites_and_version(home, capsys):
    main(["suites"])
    out = capsys.readouterr().out
    assert "gsm8k-160" in out and "160 items" in out
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.startswith(f"evalpair {__version__}")


def test_run_show_compare(home, fake, capsys):
    fake.responder = lambda model, messages, n: "Answer: A"
    main(["run", "mmlu-hs-160", "-m", "fake:1b", "--base-url", fake.url, "-n", "8", "--label", "one", "--repeat"])
    out = capsys.readouterr().out
    assert out.startswith("one  (fake:1b on mmlu-hs-160, 8 items)") and "noise floor  0 of 8" in out
    main(["run", "mmlu-hs-160", "-m", "fake:1b", "--base-url", fake.url, "-n", "8", "--label", "two",
          "--seed", "9", "--json"])
    s = json.loads(capsys.readouterr().out)
    assert s["run"] == "two" and "by_group" in s
    main(["compare", "one", "two"])
    out = capsys.readouterr().out
    assert "B − A" in out and "No clear difference" in out
    main(["runs"])
    assert "one" in capsys.readouterr().out


def test_errors_exit_2(home, capsys):
    with pytest.raises(SystemExit) as e:
        main(["show", "nope"])
    assert e.value.code == 2 and "no run 'nope'" in capsys.readouterr().err


def test_compare_text_lines_up_and_excerpts_are_one_line(home, fake, capsys, tmp_path):
    items = [{"id": f"q{i}", "question": f"What is {i} + 1?", "gold": str(i + 1)} for i in range(6)]
    d = write_suite(str(tmp_path), "add", 'checker = "number"\nuser = "{question}"\n', items)
    def respond(model, messages, n):
        i = int(messages[-1]["content"].split()[2])
        return f"Step 1:\tadd\n\nAnswer: {i + 1 if model == 'a' else 0}"
    fake.responder = respond
    for m in ("a", "b"):
        main(["run", d, "-m", m, "--base-url", fake.url, "--label", m])
    assert "\t" not in capsys.readouterr().out
    main(["compare", "a", "b"])
    rows = [x for x in capsys.readouterr().out.splitlines() if x.startswith("  worse in B:")]
    assert len(rows) == 5 and all(x.index("q") == len("  better in B: ") for x in rows)

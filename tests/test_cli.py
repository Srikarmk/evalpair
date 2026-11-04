import json

import pytest

from evalpair.cli import main


def test_suites_and_version(home, capsys):
    main(["suites"])
    out = capsys.readouterr().out
    assert "gsm8k-160" in out and "160 items" in out
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.startswith("evalpair 0.1.0")


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

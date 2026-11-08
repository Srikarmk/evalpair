import pytest

from evalpair.checkers import (canon_number, check_choice, check_chrf, chrf, check_contains, check_exact, check_json_fields,
                               check_number, check_regex, parse_choice, parse_json, parse_number)


@pytest.mark.parametrize("text,want,how", [
    ("12 + 6 = 18\nAnswer: 18", "18", "cue"),
    ("**Answer:** $1,200.00", "1200", "cue"),
    ("so the total is \\boxed{42}.", "42", "boxed"),
    ("first 3, then 7, so 10 apples", "10", "last number"),
    ("<think>maybe 5</think>no idea", None, "none"),
    ("Answer: -3.5", "-3.5", "cue"),
])
def test_parse_number(text, want, how):
    assert parse_number(text) == (want, how)


def test_canon_number():
    assert canon_number("18.0") == "18" and canon_number("1,000") == "1000" and canon_number("x") is None
    assert canon_number("nan") is None


@pytest.mark.parametrize("text,want", [
    ("Reasoning...\nAnswer: C", "C"),
    ("The answer is (B).", "B"),
    ("D", "D"),
    ("A. because ...", "A"),
    ("I think the answer: b", None),  # lowercase letters are not read as options
    ("Answer: depression", "B"),
])
def test_parse_choice(text, want):
    a, _ = parse_choice(text, ["anxiety", "depression", "pain", "stimulation"])
    assert a == want


def test_parse_json_strict_and_lenient():
    assert parse_json('{"a": 1}') == ({"a": 1}, True)
    assert parse_json('Here:\n```json\n{"a": "}"}\n```') == ({"a": "}"}, False)
    assert parse_json("no json here") == (None, False)


def test_check_number_and_choice():
    assert check_number("Answer: 18", {"gold": "18"}, {}).score == 1
    assert check_number("Answer: 17", {"gold": "18"}, {}).score == 0
    assert check_choice("Answer: c", {"gold": "C"}, {}).score == 0
    assert check_choice("Answer: C", {"gold": "c"}, {}).score == 1
    assert check_choice("Answer: E", {"gold": "E"}, {"letters": "ABCDE"}).score == 1


def test_check_exact_contains_regex():
    assert check_exact("Thinking\nAnswer: Paris.", {"gold": "paris"}, {}).score == 1
    assert check_exact("Paris", {"gold": ["Paris", "City of Light"]}, {}).score == 1
    assert check_contains("It was the Treaty of Paris, 1783.", {"gold": "treaty of paris"}, {}).score == 1
    assert check_contains("Parisian", {"gold": "paris"}, {}).score == 0
    c = check_regex("Label: spam\nLabel: ham", {"gold": "ham"}, {"pattern": r"Label: (\w+)"})
    assert (c.answer, c.score) == ("ham", 1)
    with pytest.raises(ValueError):
        check_regex("x", {"gold": "x"}, {})


def test_check_json_fields():
    gold = {"name": "The Punter", "food": None, "family_friendly": False, "price": "£20-25"}
    c = check_json_fields('{"name": "the punter", "food": "N/A", "family_friendly": "no", "price": "£20-25"}',
                          {"gold": gold}, {})
    assert c.score == 1 and c.how == "strict json"
    c = check_json_fields('```json\n{"name": "The Punter", "food": "pasta"}\n```', {"gold": gold}, {})
    assert c.score == 0.25 and c.how == "json in text"
    assert check_json_fields("sorry", {"gold": gold}, {}).score == 0
    with pytest.raises(ValueError):
        check_json_fields("{}", {"id": "x", "gold": "flat"}, {})


def test_space_as_thousands_separator_on_the_answer_line():
    assert parse_number("The difference is $409 500 - $400 000 = $9 500.\n\nAnswer: $9 500")[0] == "9500"
    assert parse_number("Answer: 1 250 000")[0] == "1250000"
    assert parse_number("so \\boxed{12 000}")[0] == "12000"
    assert parse_number("Answer: 3 apples and 450 grams")[0] == "3"


def test_chrf_matches_sacrebleu():
    # reference values from sacrebleu 2.x: CHRF().sentence_score(hyp, [ref]).score / 100
    cases = [("Add a chrF checker for free-text replies", "Add a chrF checker to evalpair", 0.555569),
             ("Fix the parser", "Fix the number parser on answer lines", 0.275346),
             ("x", "Rebuild gems, nitro, orb and item box", 0.04),
             ("ab", "abc", 0.636364)]
    for hyp, ref, want in cases:
        assert abs(chrf(hyp, ref) - want) < 1e-6
    assert chrf("same words", "same words") == 1.0
    assert chrf("", "abc") == 0.0


def test_check_chrf_first_line_and_fence():
    item = {"id": "c1", "gold": "README: link the live site\n\nThe body differs."}
    c = check_chrf("```\nREADME: link the live site\n\nSomething else entirely.\n```", item, {"line": "first"})
    assert c.score == 1.0 and c.answer == "README: link the live site"
    assert check_chrf("```markdown\nREADME: link the live site\n\nCut off by the token lim", item,
                      {"line": "first"}).score == 1.0
    assert check_chrf("   ", item, {}).score == 0.0
    best = check_chrf("Fix the parser", {"id": "c2", "gold": ["Unrelated", "Fix the parser"]}, {})
    assert best.score == 1.0

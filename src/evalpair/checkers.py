"""Deterministic answer checkers. Each returns a Check: the parsed answer, a score in [0, 1], and how the
answer was found, so a wrong score can be traced to the parser rather than the model.

The number and choice parsers come from the Measuring LLM Systems studies (prompt-brittleness, format-tax,
groupthink, quant-cliff): an "Answer:" cue first, then fallbacks. A reply with no usable answer scores 0.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

NUM = r"-?\d[\d,]*(?:\.\d+)?"
CUE = r"(?i:final answer|answer|a)"
NULLS = {"", "null", "none", "n a", "na", "unknown", "not given", "not specified", "not provided"}


@dataclass
class Check:
    answer: str | None
    score: float
    how: str


def norm(s: Any) -> str:
    """Casefolded, accents stripped, punctuation (all but £, as in the studies) to spaces, whitespace collapsed."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w£ ]", " ", s)).strip()


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.replace("**", "").replace("\\(", "").replace("\\)", "")


def _join_groups(s: str) -> str:
    """Read "9 500" (a space, no-break or thin space between thousands) as "9500". Used only on an
    answer line or a \\boxed{}, where a space between digit groups is one number, not two."""
    return re.sub(r"(?<=\d)[ \u00a0\u202f](?=\d{3}\b)", "", s)


def canon_number(s: Any) -> str | None:
    s = str(s).replace(",", "").replace("$", "").strip()
    try:
        x = float(s)
    except ValueError:
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    if x == int(x):
        return str(int(x))
    return f"{x:.6g}"


def parse_number(text: str) -> tuple[str | None, str]:
    t = _clean(text)
    cues = list(re.finditer(rf"(?m)^\s*[#>*\- ]*{CUE}\s*[:：]\s*(.*)$", t))
    if cues:
        m = re.search(NUM, _join_groups(cues[-1].group(1).replace("$", "")))
        if m:
            return canon_number(m.group(0)), "cue"
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", t)
    if boxed:
        m = re.search(NUM, _join_groups(boxed[-1]))
        if m:
            return canon_number(m.group(0)), "boxed"
    nums = re.findall(NUM, t.replace("$", ""))
    if nums:
        return canon_number(nums[-1]), "last number"
    return None, "none"


def _letter_from_text(rest: str, options: list[str] | None, letters: str) -> str | None:
    if not options:
        return None
    rest = norm(rest)
    hits = [(len(norm(o)), i) for i, o in enumerate(options) if norm(o) and rest.startswith(norm(o))]
    return letters[max(hits)[1]] if hits else None


def parse_choice(text: str, options: list[str] | None = None, letters: str = "ABCD") -> tuple[str | None, str]:
    t = _clean(text).strip()
    L = f"[{letters}]"
    cues = re.findall(rf"(?m)^\s*[#>*\- ]*{CUE}\s*[:：]\s*\(?({L})\b", t)
    if cues:
        return cues[-1], "cue"
    m = re.search(rf"\b(?i:answer|option|choice) is\s*:?\s*\(?({L})\b", t)
    if m:
        return m.group(1), "answer is"
    m = re.fullmatch(rf"\(?({L})\)?[.:)]?", t)
    if m:
        return m.group(1), "bare letter"
    m = re.match(rf"\(?({L})[.):]\s", t + " ")
    if m:
        return m.group(1), "leading letter"
    m = re.search(rf"\b(?i:answer)\s*[:：]?\s*\(?({L})\b", t)
    if m:
        return m.group(1), "inline cue"
    cues = re.findall(rf"(?m)^\s*[#>*\- ]*{CUE}\s*[:：]\s*(.+)$", t)
    if cues:
        a = _letter_from_text(cues[-1], options, letters)
        if a:
            return a, "option text"
    return None, "none"


def _json_objects(t: str) -> list[str]:
    """Top-level {...} spans, skipping braces inside strings."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                out.append(t[start:i + 1])
    return out


def parse_json(text: str) -> tuple[dict | None, bool]:
    """(object, strict). `strict` is True when the whole reply is exactly one JSON object; otherwise the
    first JSON object anywhere in the reply, so code fences and a sentence around it are tolerated."""
    t = _clean(text).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj, True
    except json.JSONDecodeError:
        pass
    for blob in _json_objects(t):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj, False
    return None, False


def _is_null(x) -> bool:
    return x is None or (isinstance(x, str) and norm(x) in NULLS)


def _as_bool(x):
    if isinstance(x, bool):
        return x
    s = norm(x)
    if s in ("yes", "true", "y"):
        return True
    if s in ("no", "false", "n"):
        return False
    return "other"


def field_ok(pred, gold) -> bool:
    if gold is None:
        return _is_null(pred)
    if _is_null(pred):
        return False
    if isinstance(gold, bool):
        return _as_bool(pred) == gold
    if isinstance(gold, (int, float)):
        return canon_number(pred) == canon_number(gold)
    return norm(pred) == norm(gold)


# ---------------------------------------------------------------- the checkers a suite can name

def check_number(text: str, item: dict, cfg: dict) -> Check:
    a, how = parse_number(text)
    return Check(a, float(a is not None and a == canon_number(item["gold"])), how)


def check_choice(text: str, item: dict, cfg: dict) -> Check:
    letters = cfg.get("letters", "ABCD")
    a, how = parse_choice(text, item.get("options"), letters)
    return Check(a, float(a is not None and a == str(item["gold"]).strip().upper()), how)


def check_exact(text: str, item: dict, cfg: dict) -> Check:
    """The last non-empty line after any "Answer:" cue, compared after normalisation."""
    t = _clean(text).strip()
    cues = re.findall(rf"(?m)^\s*[#>*\- ]*{CUE}\s*[:：]\s*(.+)$", t)
    if cues:
        a, how = cues[-1].strip(), "cue"
    else:
        lines = [x for x in t.splitlines() if x.strip()]
        a, how = (lines[-1].strip(), "last line") if lines else (None, "none")
    golds = item["gold"] if isinstance(item["gold"], list) else [item["gold"]]
    ok = a is not None and norm(a) in {norm(g) for g in golds}
    return Check(a, float(ok), how)


def check_contains(text: str, item: dict, cfg: dict) -> Check:
    golds = item["gold"] if isinstance(item["gold"], list) else [item["gold"]]
    t = norm(_clean(text))
    hit = next((g for g in golds if norm(g) and f" {norm(g)} " in f" {t} "), None)
    return Check(hit, float(hit is not None), "contains" if hit else "none")


def check_regex(text: str, item: dict, cfg: dict) -> Check:
    """The suite's `pattern` (first group, or the whole match), last match wins."""
    pat = cfg.get("pattern")
    if not pat:
        raise ValueError("the regex checker needs a 'pattern' in the suite's [checker] table")
    ms = list(re.finditer(pat, _clean(text), flags=re.M))
    if not ms:
        return Check(None, 0.0, "none")
    m = ms[-1]
    a = (m.group(1) if m.groups() else m.group(0)).strip()
    golds = item["gold"] if isinstance(item["gold"], list) else [item["gold"]]
    return Check(a, float(norm(a) in {norm(g) for g in golds}), "regex")


def check_json_fields(text: str, item: dict, cfg: dict) -> Check:
    """Share of the gold object's fields the reply gets right. A field whose gold is null must be null,
    missing or a null-like string in the reply. Booleans accept yes/no and true/false."""
    gold = item["gold"]
    if not isinstance(gold, dict) or not gold:
        raise ValueError(f"item {item.get('id')}: the json_fields checker needs a non-empty object as gold")
    obj, strict = parse_json(text)
    if obj is None:
        return Check(None, 0.0, "no json")
    ok = {f: field_ok(obj.get(f), g) for f, g in gold.items()}
    answer = json.dumps({f: (None if _is_null(obj.get(f)) else str(obj.get(f))) for f in gold}, sort_keys=True)
    return Check(answer, sum(ok.values()) / len(ok), "strict json" if strict else "json in text")


def _ngrams(s: str, n: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in range(len(s) - n + 1):
        out[s[i:i + n]] = out.get(s[i:i + n], 0) + 1
    return out


def chrf(hyp: str, ref: str, order: int = 6, beta: float = 2.0) -> float:
    """Sentence chrF in [0, 1], computed as sacreBLEU's CHRF(char_order=6, word_order=0, beta=2) does:
    whitespace removed, character n-grams 1..6, precision and recall averaged over the orders both sides
    have, then the F-beta of those averages (recall weighted beta times as much as precision)."""
    hyp, ref = "".join(hyp.split()), "".join(ref.split())
    eps, factor = 1e-16, beta ** 2
    avg_p = avg_r = 0.0
    effective = 0
    for n in range(1, order + 1):
        h, r = _ngrams(hyp, n), _ngrams(ref, n)
        nh, nr = sum(h.values()), sum(r.values())
        match = sum(min(c, r[g]) for g, c in h.items() if g in r)
        avg_p += match / nh if nh else eps
        avg_r += match / nr if nr else eps
        effective += bool(nh and nr)
    if not effective:
        return 0.0
    avg_p, avg_r = avg_p / effective, avg_r / effective
    if not avg_p + avg_r:
        return 0.0
    return (1 + factor) * avg_p * avg_r / (factor * avg_p + avg_r)


def _unfence(t: str) -> str:
    """Drop a code fence wrapped around the whole reply, which small models often add."""
    m = re.fullmatch(r"\s*```[\w-]*\n(.*?)\n?```\s*", t, flags=re.S)
    if m:
        return m.group(1)
    m = re.match(r"\s*```[\w-]*\n", t)  # opened but never closed: the reply ran out of tokens
    return t[m.end():] if m else t


def check_chrf(text: str, item: dict, cfg: dict) -> Check:
    """chrF between the reply and the gold text (the best of several golds), for free text with a reference,
    such as a commit message or a summary. `line = "first"` compares only the first non-empty line of each,
    e.g. a commit message's subject. Case-sensitive, as sacreBLEU's chrF is."""
    t = _unfence(_clean(text)).strip()
    golds = item["gold"] if isinstance(item["gold"], list) else [item["gold"]]
    if cfg.get("line") == "first":
        first = lambda s: next((x.strip() for x in str(s).splitlines() if x.strip()), "")
        t, golds = first(t), [first(g) for g in golds]
    if not t:
        return Check(None, 0.0, "empty")
    return Check(t, max(chrf(t, str(g)) for g in golds), "chrf")


CHECKERS: dict[str, Callable[[str, dict, dict], Check]] = {
    "chrf": check_chrf,
    "number": check_number,
    "choice": check_choice,
    "exact": check_exact,
    "contains": check_contains,
    "regex": check_regex,
    "json_fields": check_json_fields,
}

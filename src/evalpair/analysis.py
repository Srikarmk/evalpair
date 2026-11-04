"""Summaries of one run and comparisons of two, built from saved rows only (no model calls).

A comparison pairs the two runs by item id over the items both answered (first pass). It reports the
score difference with a paired bootstrap interval, how many items flipped each way, how often the answer
changed at all, and, when a run was made with a repeat pass, that run's own answer-change rate: the noise
floor. A change rate between two runs only means something above the floor.
"""

from __future__ import annotations

from .stats import excludes_zero, mean, mean_ci

EXCERPT = 400


def _pp(x: float) -> str:
    return f"{100 * x:+.1f} pp"


def _pc(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ci(ci: dict, pct=_pc) -> dict:
    return {**{k: round(v, 4) for k, v in ci.items() if k != "n"}, "n": ci["n"],
            "text": f"{pct(ci['mean'])} [{pct(ci['lo'])}, {pct(ci['hi'])}]" if ci["n"] else "no items"}


def by_pass(rows: list[dict], p: int) -> dict[str, dict]:
    return {r["id"]: r for r in rows if r["pass"] == p}


def excerpt(text: str, n: int = EXCERPT) -> str:
    text = text or ""
    return text if len(text) <= n else "…" + text[-n:]


def noise_floor(rows: list[dict]) -> dict | None:
    """Share of items whose parsed answer changed between pass 1 and the repeat pass, and the score
    change between them. None if the run had no repeat."""
    a, b = by_pass(rows, 1), by_pass(rows, 2)
    ids = [i for i in a if i in b]
    if not ids:
        return None
    changed = [float(a[i]["answer"] != b[i]["answer"]) for i in ids]
    groups = [a[i]["group"] for i in ids]
    return {"items": len(ids), "answers_changed": int(sum(changed)),
            "change_rate": _ci(mean_ci(changed, groups)),
            "score_change": _ci(mean_ci([b[i]["score"] - a[i]["score"] for i in ids], groups), _pp)}


def summarize(meta: dict, rows: list[dict], failures: int = 5) -> dict:
    first = list(by_pass(rows, 1).values())
    groups = [r["group"] for r in first]
    wrong = [r for r in first if r["score"] < 1]
    out = {
        "run": meta["run"], "suite": meta["suite"], "model": meta["model"], "backend": meta["backend"],
        "items": len(first), "score": _ci(mean_ci([r["score"] for r in first], groups)),
        "no_answer": sum(r["answer"] is None for r in first), "truncated": sum(r["truncated"] for r in first),
        "mean_output_tokens": round(mean(r["output_tokens"] for r in first), 1),
        "noise_floor": noise_floor(rows),
        "calls": meta.get("calls"), "cache_hits": meta.get("cache_hits"), "finished": meta.get("finished"),
    }
    if any(r["group"] is not None for r in first):
        per = {}
        for g in sorted({r["group"] for r in first}, key=str):
            rs = [r for r in first if r["group"] == g]
            per[str(g)] = {"items": len(rs), "score": round(mean(r["score"] for r in rs), 4)}
        out["by_group"] = per
    # unparsed and truncated replies first: they are usually the checker's or the budget's fault, not the model's
    wrong.sort(key=lambda r: (r["answer"] is not None, not r["truncated"], r["id"]))
    out["failures"] = [{"id": r["id"], "answer": r["answer"], "score": r["score"], "how": r["how"],
                        "truncated": r["truncated"], "reply_end": excerpt(r["text"])} for r in wrong[:failures]]
    return out


def compare(meta_a: dict, rows_a: list[dict], meta_b: dict, rows_b: list[dict], examples: int = 5) -> dict:
    a, b = by_pass(rows_a, 1), by_pass(rows_b, 1)
    ids = [i for i in a if i in b]
    notes = []
    if meta_a.get("suite_hash") != meta_b.get("suite_hash"):
        notes.append("The two runs used different versions of the suite (prompt, checker or items differ); "
                     "only items with the same id are paired, and a difference may come from the suite.")
    only = len(a) - len(ids), len(b) - len(ids)
    if any(only):
        notes.append(f"Paired on the {len(ids)} items both runs answered ({only[0]} only in A, {only[1]} only in B).")
    if not ids:
        return {"a": meta_a["run"], "b": meta_b["run"], "items": 0, "notes": notes + ["No items in common."]}
    groups = [a[i]["group"] for i in ids]
    sa, sb = [a[i]["score"] for i in ids], [b[i]["score"] for i in ids]
    diff = mean_ci([y - x for x, y in zip(sa, sb)], groups)
    worse = [i for i in ids if b[i]["score"] < a[i]["score"]]
    better = [i for i in ids if b[i]["score"] > a[i]["score"]]
    changed = [float(a[i]["answer"] != b[i]["answer"]) for i in ids]
    out = {
        "a": {"run": meta_a["run"], "model": meta_a["model"], "score": _ci(mean_ci(sa, groups))},
        "b": {"run": meta_b["run"], "model": meta_b["model"], "score": _ci(mean_ci(sb, groups))},
        "items": len(ids),
        "difference": _ci(diff, _pp),
        "worse_in_b": len(worse), "better_in_b": len(better),
        "answers_changed": _ci(mean_ci(changed, groups)),
    }
    floors = {}
    for name, rows in (("a", rows_a), ("b", rows_b)):
        rep, p1 = by_pass(rows, 2), by_pass(rows, 1)
        common = [i for i in ids if i in rep]
        if common:
            # per item: changed between the runs, minus changed between this run's own two passes
            excess = [float(a[i]["answer"] != b[i]["answer"]) - float(p1[i]["answer"] != rep[i]["answer"])
                      for i in common]
            floors[name] = {"items": len(common),
                            "change_rate": _ci(mean_ci([float(p1[i]["answer"] != rep[i]["answer"]) for i in common],
                                                       [a[i]["group"] for i in common])),
                            "excess_over_floor": _ci(mean_ci(excess, [a[i]["group"] for i in common]), _pp)}
    out["noise_floor"] = floors or None
    if excludes_zero(diff):
        verdict = (f"B scores {'higher' if diff['mean'] > 0 else 'lower'} than A: "
                   f"{_pp(diff['mean'])} [{_pp(diff['lo'])}, {_pp(diff['hi'])}], paired over {len(ids)} items.")
    else:
        verdict = (f"No clear difference in score: {_pp(diff['mean'])} [{_pp(diff['lo'])}, {_pp(diff['hi'])}], "
                   f"an interval that includes zero, over {len(ids)} items.")
    ch = out["answers_changed"]
    verdict += f" Answers changed on {_pc(ch['mean'])} of items ({len(worse)} got worse, {len(better)} better)."
    if floors:
        # beyond noise only if it clears every floor measured, so a quiet run can't hide a noisy one
        beyond = all(f["excess_over_floor"]["lo"] > 0 for f in floors.values())
        rates = " and ".join(f"{_pc(f['change_rate']['mean'])} for {k.upper()}" for k, f in floors.items())
        verdict += (f" The noise floor (the same model asked twice) is {rates}, so "
                    + ("the change is beyond noise." if beyond else "the change is not clearly beyond noise."))
    else:
        verdict += " Neither run has a repeat pass, so there is no noise floor to compare against (run with repeat)."
    out["verdict"] = verdict
    out["notes"] = notes

    def ex_rows(ids_):
        return [{"id": i, "a_answer": a[i]["answer"], "b_answer": b[i]["answer"],
                 "a_score": a[i]["score"], "b_score": b[i]["score"],
                 "a_reply_end": excerpt(a[i]["text"], 240), "b_reply_end": excerpt(b[i]["text"], 240)}
                for i in ids_[:examples]]
    out["examples"] = {"worse_in_b": ex_rows(worse), "better_in_b": ex_rows(better)}
    return out

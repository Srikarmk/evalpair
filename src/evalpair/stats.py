"""Percentile bootstrap over items, never over single calls.

Every run of a suite answers the same items, so a comparison is paired: resample items, and an item's
score under both runs moves together. With a `group` (e.g. MMLU subject), items are resampled within each
group so every resample keeps the suite's mix. Pure Python, seeded, so the same inputs give the same interval.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

BOOT = 2000


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else math.nan


def _percentile(sorted_xs: list[float], q: float) -> float:
    # linear interpolation, as numpy's default
    pos = (len(sorted_xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def mean_ci(values, groups=None, n_boot: int = BOOT, seed: int = 0, level: float = 0.95) -> dict:
    values = [float(v) for v in values]
    n = len(values)
    if n == 0:
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "n": 0}
    groups = list(groups) if groups is not None else [0] * n
    by = defaultdict(list)
    for v, g in zip(values, groups):
        by[g].append(v)
    strata = list(by.values())
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for vs in strata:
            s += sum(rng.choices(vs, k=len(vs)))
        boots.append(s / n)
    boots.sort()
    a = (1 - level) / 2
    return {"mean": sum(values) / n, "lo": _percentile(boots, a), "hi": _percentile(boots, 1 - a), "n": n}


def excludes_zero(ci: dict) -> bool:
    return ci["n"] > 0 and (ci["lo"] > 0 or ci["hi"] < 0)

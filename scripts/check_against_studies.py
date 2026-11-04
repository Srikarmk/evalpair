"""Re-score a study's saved replies with evalpair's checkers and count disagreements with the study's own scores.

    python scripts/check_against_studies.py ../quant-cliff/results/answers
"""

import glob
import json
import os
import sys
from collections import Counter

from evalpair import suites

SUITE = {"gsm8k": "gsm8k-160", "mmlu": "mmlu-hs-160", "drift": "e2e-extract-150"}


def main(folder):
    h = os.path.join(os.path.dirname(__file__), ".no-home")
    S = {t: suites.find(n, h) for t, n in SUITE.items()}
    items = {t: {it["id"]: it for it in s.items} for t, s in S.items()}
    total, diff = Counter(), []
    for f in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        if "_pilot" in f:
            continue
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            it = items.get(r["task"], {}).get(r["id"])
            if it is None or r.get("variant") != "base":
                continue
            mine = S[r["task"]].check(r["output"], it).score
            theirs = (sum(r["fields"].values()) / len(r["fields"]) if r.get("fields") else 0.0) \
                if r["task"] == "drift" else float(bool(r["correct"]))
            total[r["task"]] += 1
            if abs(mine - theirs) > 1e-9:
                diff.append((os.path.basename(f), r["id"], mine, theirs))
    print(f"{sum(total.values())} replies ({dict(total)}), {len(diff)} scored differently")
    for d in diff[:20]:
        print(" ", *d)


if __name__ == "__main__":
    main(sys.argv[1])

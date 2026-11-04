import math

from evalpair.stats import excludes_zero, mean_ci


def test_interval_brackets_the_mean_and_is_seeded():
    xs = [1] * 80 + [0] * 20
    a, b = mean_ci(xs), mean_ci(xs)
    assert a == b and a["mean"] == 0.8 and a["lo"] < 0.8 < a["hi"] and a["n"] == 100
    # the normal approximation: 0.8 ± 1.96 * sqrt(.16 / 100) ≈ [0.72, 0.88]
    assert abs(a["lo"] - 0.72) < 0.02 and abs(a["hi"] - 0.88) < 0.02


def test_constant_values_have_a_zero_width_interval():
    ci = mean_ci([0.0] * 50)
    assert ci["lo"] == ci["hi"] == 0 and not excludes_zero(ci)


def test_strata_keep_their_size():
    # one group all 1s, one all 0s: a stratified resample always has exactly half 1s
    ci = mean_ci([1] * 30 + [0] * 30, ["a"] * 30 + ["b"] * 30)
    assert ci["lo"] == ci["hi"] == 0.5


def test_excludes_zero_and_empty():
    assert excludes_zero(mean_ci([0.1] * 40 + [0.2] * 40))
    assert math.isnan(mean_ci([])["mean"]) and not excludes_zero(mean_ci([]))

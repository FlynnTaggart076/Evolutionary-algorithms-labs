import numpy as np
from portfolio import (Problem, breed, crowding_distance, environmental_selection,
                       exact_front, metrics, nondominated_sort, optimize, pareto_archive)


def small():
    return Problem(np.array([[5., 2, 3, 1], [7, 4, 1, 2], [9, 3, 4, 2], [4, 2, 1, 1]]), 7, 4, ((2, 0),))


def test_strict_dominance_and_crowding():
    f = np.array([[0., 3], [1, 2], [3, 0], [2, 3], [0, 3]])
    fronts, ranks = nondominated_sort(f)
    assert set(fronts[0]) == {0, 1, 2, 4}
    assert ranks[3] == 1
    d = crowding_distance(f, np.array([0, 1, 2]))
    assert np.isinf(d[[0, 2]]).all() and np.isclose(d[1], 2)
    assert np.isfinite(crowding_distance(np.ones((3, 2)), np.arange(3))).all()


def test_repair_and_binary_variation():
    p = small()
    rng = np.random.default_rng(2)
    x = p.repair(rng.random((100, p.n)) < .7, rng)
    assert p.feasible(x).all()
    children = breed(x, np.arange(100), rng, 1., 1. / p.n)
    assert children.dtype == bool
    assert p.feasible(p.repair(children, rng)).all()


def test_archive_against_independent_brute_dominance():
    p = small()
    (x, f), _ = exact_front(p)
    all_x = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(bool)
    all_x = all_x[p.feasible(all_x)]
    all_f = p.evaluate(all_x)
    expected = {tuple(row) for row in all_f if not any(np.all(other <= row) and np.any(other < row) for other in all_f)}
    assert {tuple(row) for row in f} == expected
    assert p.feasible(x).all()
    assert metrics(f, f, p.scales)["igd"] == 0
    assert metrics(f, f, p.scales)["exact_coverage"] == 1


def test_exact_budget_and_seed_replay():
    p = small()
    a, h, n = optimize(p, 42, 123, 12, .9, .25)
    b, j, m = optimize(p, 42, 123, 12, .9, .25)
    assert n == m == 123 and h == j
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert p.feasible(a[0]).all()


def test_elitist_selection_retains_extremes():
    vals = np.array([[0., 4], [1, 3], [2, 2], [3, 1], [4, 0], [5, 5]])
    pop = np.eye(6, dtype=bool)
    _, retained = environmental_selection(pop, vals, 3)
    assert any(np.array_equal(r, vals[0]) for r in retained)
    assert any(np.array_equal(r, vals[4]) for r in retained)
    assert not any(np.array_equal(r, vals[5]) for r in retained)

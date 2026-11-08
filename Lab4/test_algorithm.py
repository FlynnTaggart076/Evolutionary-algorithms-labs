import random

import numpy as np
import pytest

from algorithm import (BudgetExhausted, Evaluator, Parameters, distance_matrix,
                       inversion, local_search, mst_lower_bound, optimize,
                       order_crossover, tour_length, two_opt_neighbor)


def test_known_closed_tour_and_mst():
    points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    assert tour_length(np.arange(4), distance_matrix(points)) == pytest.approx(4)
    assert mst_lower_bound(points) == pytest.approx(3)


def test_permutation_operators():
    rng = random.Random(17)
    for n in (4, 8, 40):
        a, b = np.arange(n), np.arange(n)[::-1]
        for _ in range(30):
            assert sorted(order_crossover(a, b, rng)) == list(range(n))
            assert sorted(inversion(a, rng)) == list(range(n))


def test_two_opt_removes_crossing_and_is_involution():
    points = np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.]])
    route = np.arange(4)
    candidate = two_opt_neighbor(route, 0, 2)
    distances = distance_matrix(points)
    assert tour_length(candidate, distances) < tour_length(route, distances)
    np.testing.assert_array_equal(two_opt_neighbor(candidate, 0, 2), route)
    with pytest.raises(ValueError):
        two_opt_neighbor(route, 0, 3)


def test_budget_includes_local_search_and_cannot_overrun():
    points = np.random.default_rng(3).uniform(size=(8, 2))
    evaluator = Evaluator(distance_matrix(points), budget=7, checkpoint=2)
    route = np.arange(8)
    value = evaluator(route)
    improved, score = local_search(route, value, evaluator, random.Random(4), 100)
    assert evaluator.evaluations == 7
    assert score <= value
    assert sorted(improved) == list(range(8))
    with pytest.raises(BudgetExhausted):
        evaluator(route)


@pytest.mark.parametrize("memetic", [False, True])
def test_fixed_budget_elitism_and_reproducibility(memetic):
    points = np.random.default_rng(8).uniform(size=(9, 2))
    p = Parameters(population=12, budget=157, local_checks=20, trace_step=10)
    a, b = optimize(points, 99, memetic, p), optimize(points, 99, memetic, p)
    assert a["evaluations"] == b["evaluations"] == 157
    assert a["tour"] == b["tour"]
    assert a["best"] == b["best"]
    assert a["trace"] == b["trace"]
    assert sorted(a["tour"]) == list(range(9))
    scores = [v for _, v in a["trace"]]
    assert all(y <= x for x, y in zip(scores, scores[1:]))
    assert a["local_evaluations"] > 0 if memetic else a["local_evaluations"] == 0


def test_invalid_configuration():
    with pytest.raises(ValueError):
        optimize(np.zeros((6, 2)), 1, False, Parameters(population=10, budget=9))


@pytest.mark.parametrize("points", [np.zeros(6), np.zeros((5, 3)), np.full((5, 2), np.nan), np.full((5, 2), np.inf)])
def test_invalid_coordinate_table_rejected_before_search(points):
    with pytest.raises(ValueError, match="coordinate table"):
        optimize(points, 1, False)


@pytest.mark.parametrize("field,value", [("population", 4.5), ("budget", 100.5), ("local_checks", float("nan"))])
def test_noninteger_search_parameters_rejected(field, value):
    from dataclasses import replace
    with pytest.raises(ValueError, match="integer"):
        optimize(np.zeros((5, 2)), 1, False, replace(Parameters(), **{field: value}))


def test_zero_trace_checkpoint_rejected():
    with pytest.raises(ValueError, match="checkpoint"):
        Evaluator(distance_matrix(np.zeros((5, 2))), 5, checkpoint=0)


def test_generator_preserves_nonmultiple_city_count(tmp_path):
    from experiment import generated_points, load_data
    config = {"cities": 13, "data_seed": 77}
    path = tmp_path / "new" / "points.csv"
    points = load_data(config, path)
    assert points.shape == (13, 2)
    np.testing.assert_array_equal(points, generated_points(config))
    np.testing.assert_array_equal(points, load_data(config, path))


def test_duplicate_city_id_rejected(tmp_path):
    from experiment import load_data
    path = tmp_path / "bad.csv"
    path.write_text("city,x,y\n0,0,0\n1,1,0\n1,1,1\n3,0,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identifiers"):
        load_data({"cities": 4, "data_seed": 1}, path)


def test_custom_data_report_uses_actual_model_and_handles_zero_lengths(tmp_path):
    from experiment import make_report
    path = tmp_path / "imported.csv"
    path.write_text("city,x,y\n" + "".join(f"{i},0,0\n" for i in range(5)), encoding="utf-8")
    config = {"cities": 5, "data_seed": 1, "population": 4, "elites": 1, "tournament_size": 2,
              "crossover_probability": .5, "mutation_probability": .1, "local_probability": .1,
              "sparse_local_probability": .01, "local_checks": 5, "budget": 100,
              "runs": 1, "seed_start": 17}
    summary = [{"best": 0., "mean": 0., "median": 0., "std": 0., "worst": 0.,
                "mean_seconds": .01, "mean_local_evaluations": 0.} for _ in range(3)]
    results = [{"algorithm": a, "seed": 17, "best": 0.} for a in ["ga", "memetic", "memetic_sparse"]]
    report = tmp_path / "report.md"
    make_report(config, summary, {"nearest_neighbor": 0., "mst_lower_bound": 0.}, results,
                tmp_path, data_path=path, report_path=report)
    text = report.read_text(encoding="utf-8")
    assert "обслуживания 5 пунктов" in text and "40 пунктов" not in text
    assert "входная таблица imported.csv" in text
    assert "синтетические:" not in text
    assert "0.1 и 0.01" in text and "seed 17" in text
    assert "(./algorithm.png)" in text
    assert "оптимальность доказана" in text


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_distance_stable_for_extreme_finite_scales(scale):
    points = np.array([[0., 0.], [scale, 0.], [scale, scale], [0., scale]])
    distances = distance_matrix(points)
    assert tour_length(np.arange(4), distances) / scale == pytest.approx(4.)
    result = optimize(points, 1, False, Parameters(population=4, budget=12, elites=1))
    assert result["best"] / scale == pytest.approx(4.)


def test_unrepresentable_distance_rejected_explicitly():
    with pytest.raises(ValueError, match="floating-point range"):
        distance_matrix(np.array([[-1e308, 0.], [1e308, 0.], [0., 1.], [1., 0.]]))

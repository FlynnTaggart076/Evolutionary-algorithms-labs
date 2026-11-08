from itertools import product
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from Lab2.core import Problem, Evaluator, crossover, exact_dynamic_programming, repair, solve
from Lab2.main import make_examples, summarize, validate_config


def small_problem():
    return Problem(tuple(f"I{i}" for i in range(6)),
                   np.array([3, 5, 2, 7, 4, 1]), np.array([4, 2, 5, 3, 1, 2]),
                   np.array([7, 8, 6, 10, 5, 3]), 10, 9)


def test_repair_is_feasible_binary_and_idempotent():
    problem = small_problem()
    for bits in product((0, 1), repeat=problem.n):
        raw = np.array(bits)
        fixed = repair(raw, problem)
        assert problem.describe(fixed)["feasible"]
        assert np.all(fixed <= raw)
        assert np.array_equal(repair(fixed, problem), fixed)
        if problem.describe(raw)["feasible"]:
            assert np.array_equal(raw, fixed)


def test_exact_dp_agrees_with_independent_exhaustive_search():
    problem = small_problem()
    brute = max(problem.describe(np.array(bits))["value"]
                for bits in product((0, 1), repeat=problem.n)
                if problem.describe(np.array(bits))["feasible"])
    exact = exact_dynamic_programming(problem)
    assert exact["feasible"]
    assert exact["value"] == brute


def test_crossovers_preserve_binary_genes_and_parents():
    rng = np.random.default_rng(17)
    first, second = np.zeros(20, dtype=bool), np.ones(20, dtype=bool)
    for kind in ("uniform", "two_point"):
        for _ in range(100):
            child = crossover(first, second, kind, rng)
            assert child.shape == first.shape
            assert np.all((child == first) | (child == second))
    assert not first.any() and second.all()


def test_budget_and_seed_reproduce_solution_and_trajectory():
    problem = small_problem()
    config = {"population": 10, "budget": 100, "penalty": 10,
              "tournament": 3, "crossover_probability": 0.9, "mutation_probability": 1/6}
    method = {"name": "test", "algorithm": "ga", "constraint": "repair", "crossover": "uniform"}
    first, second = solve(problem, config, method, 44), solve(problem, config, method, 44)
    assert first["evaluations"] == second["evaluations"] == 100
    assert first["solution"] == second["solution"]
    assert first["history"] == second["history"]
    assert first["feasible_fraction"] == 1
    assert all(a[1] <= b[1] for a, b in zip(first["history"], first["history"][1:]))


def test_evaluator_refuses_budget_overrun_and_invalid_binary():
    evaluator = Evaluator(small_problem(), 2, 10)
    evaluator.evaluate(np.zeros((2, 6), dtype=bool))
    with pytest.raises(ValueError, match="budget"):
        evaluator.evaluate(np.zeros((1, 6), dtype=bool))
    with pytest.raises(ValueError, match="binary"):
        small_problem().validate(np.full(6, 2))


@pytest.mark.parametrize("key,value", [
    ("mutation_probability", -0.1), ("mutation_probability", 1.1),
    ("crossover_probability", float("nan")), ("penalty", -1),
    ("tournament", 0), ("population", 10.5),
])
def test_invalid_search_parameters_are_rejected(key, value):
    config = {"population": 10, "budget": 100, "penalty": 10,
              "tournament": 3, "crossover_probability": 0.9, "mutation_probability": 0.1}
    config[key] = value
    method = {"name": "test", "algorithm": "ga", "constraint": "repair", "crossover": "uniform"}
    with pytest.raises(ValueError):
        solve(small_problem(), config, method, 44)


def test_duplicate_ids_and_nonpositive_resources_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        Problem(("A", "A"), np.array([1, 2]), np.array([1, 2]), np.array([3, 4]), 5, 5)
    for weights in ([0, 2], [-1, 2]):
        with pytest.raises(ValueError, match="positive"):
            Problem(("A", "B"), np.array(weights), np.array([1, 2]), np.array([3, 4]), 5, 5)


def test_narrow_integer_inputs_cannot_overflow_objective_or_constraints():
    problem = Problem(("A", "B"), np.array([100,100],dtype=np.int8),
                      np.array([100,100],dtype=np.int8), np.array([100,100],dtype=np.int8),150,150)
    result = problem.describe(np.ones(2,dtype=bool))
    assert result["mass"]==result["volume"]==result["value"]==200
    assert result["feasible"] is False
    score, feasible = Evaluator(problem,1,10).evaluate(np.ones((1,2)))
    assert score[0]==-800 and not feasible[0]
    assert exact_dynamic_programming(problem)["value"]==100


def test_integer_domain_and_combined_penalty_avoid_overflow():
    resource = np.array([2**62,2**62-1],dtype=np.int64)
    problem = Problem(("A","B"),resource,resource,np.array([3,5]),1,1)
    score, feasible = Evaluator(problem,1,1).evaluate(np.ones((1,2),dtype=bool))
    assert not feasible[0] and score[0]<0
    with pytest.raises(ValueError,match="64-bit"):
        Problem(("A","B"),np.array([2**62,2**62]),resource,np.array([3,5]),1,1)
    with pytest.raises(ValueError,match="64-bit"):
        Problem(("A",),np.array([1]),np.array([1]),np.array([2**63],dtype=np.uint64),1,1)


def test_float_binary_vectors_retain_exact_integer_objective():
    problem = Problem(("A","B"),np.array([1,1]),np.array([1,1]),np.array([2**53,1]),2,2)
    expected = 2**53+1
    assert problem.describe(np.array([1.,1.]))["value"]==expected
    evaluator = Evaluator(problem,1,0)
    evaluator.evaluate(np.array([[1.,1.]]))
    assert evaluator.best_value==expected


def test_duplicate_seeds_are_not_reported_as_independent():
    config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text(encoding="utf-8"))
    config["seeds"] = [41, 41]
    with pytest.raises(ValueError, match="unique"):
        validate_config(config)


def test_redundant_constraints_still_produce_two_infeasible_examples():
    problem = Problem(tuple(map(str, range(30))), np.full(30, 4), np.full(30, 4),
                      np.arange(30) + 1, 60, 60)
    examples = make_examples(problem, exact_dynamic_programming(problem))
    assert [row["feasible"] for row in examples] == [True, True, False, False]
    assert examples[2]["bits"] != examples[3]["bits"]
    for row in examples[2:]:
        assert row["mass_excess"] == row["volume_excess"] > 0


def test_zero_optimum_summary_has_defined_gap():
    row = {"method": "test", "best_feasible_value": 0, "feasible_fraction": 1.,
           "selected_feasible": 1, "seconds": 0., "evaluations": 10}
    result = summarize([row], reference_value=0)[0]
    assert result["mean_gap_percent"] == 0 and result["optimum_fraction"] == 1


def test_zero_optimum_examples_are_distinct_or_reject_impossible_model():
    problem = Problem(("A","B","C","D"),np.ones(4,dtype=int),np.ones(4,dtype=int),
                      np.zeros(4,dtype=int),2,2)
    examples = make_examples(problem,exact_dynamic_programming(problem))
    assert examples[0]["feasible"] and examples[1]["feasible"]
    assert examples[0]["bits"]!=examples[1]["bits"]
    impossible = Problem(("A","B"),np.array([2,3]),np.array([2,3]),np.array([0,0]),1,1)
    with pytest.raises(ValueError,match="two distinct feasible"):
        make_examples(impossible,exact_dynamic_programming(impossible))


def test_cli_nondefault_model_and_imported_data_have_truthful_isolated_reports(tmp_path):
    lab = Path(__file__).resolve().parents[1]
    canonical_report = lab / "report.md"
    canonical_report_bytes = canonical_report.read_bytes() if canonical_report.exists() else None
    config = json.loads((lab / "config.json").read_text(encoding="utf-8"))
    config.update(objects=32, mass_limit=140, volume_limit=120, penalty=0.5,
                  population=16, budget=64, seeds=[41, 57])
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    data_file = tmp_path / "items.csv"
    output = tmp_path / "generated_results"
    command = [sys.executable, str(lab / "main.py"), "--config", str(config_file),
               "--data", str(data_file), "--output", str(output)]
    first = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=60)
    assert first.returncode == 0, first.stderr
    report = (output / "report.md").read_text(encoding="utf-8")
    for text in ("32 неделимых", "{0,1}^32", "x_i ≤ 140", "x_i ≤ 120",
                 "m_i/140 + v_i/120", "F(x) - 0.5", "O(n·140·120)", "41, 57"):
        assert text in report
    assert "x_i ≤ 200" not in report and "40 неделимых" not in report
    meta = json.loads((output / "input_data.json").read_text(encoding="utf-8"))
    assert meta["matches_config_generator"] is True and meta["generated_during_run"] is True
    assert meta["sha256"] == hashlib.sha256(data_file.read_bytes()).hexdigest()
    with data_file.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["value"] = str(int(row["value"]) + 17)
    with data_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item", "mass", "volume", "value"])
        writer.writeheader()
        writer.writerows(rows)
    data_hash = hashlib.sha256(data_file.read_bytes()).hexdigest()
    imported_output = tmp_path / "imported_results"
    command[-1] = str(imported_output)
    second = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=60)
    assert second.returncode == 0, second.stderr
    imported_report = (imported_output / "report.md").read_text(encoding="utf-8")
    assert "Использована предоставленная входная таблица" in imported_report
    assert "Данные соответствуют генератору" not in imported_report
    assert "один синтетический экземпляр" not in imported_report
    assert "у синтетических предметов" not in imported_report
    imported_meta = json.loads((imported_output / "input_data.json").read_text(encoding="utf-8"))
    assert imported_meta["matches_config_generator"] is False
    assert imported_meta["generated_during_run"] is False
    assert hashlib.sha256(data_file.read_bytes()).hexdigest() == data_hash == imported_meta["sha256"]
    after = canonical_report.read_bytes() if canonical_report.exists() else None
    assert after == canonical_report_bytes
    for result_dir in (output, imported_output):
        with (result_dir / "runs.csv").open(encoding="utf-8", newline="") as stream:
            runs = list(csv.DictReader(stream))
        assert len(runs) == 8 and all(int(row["evaluations"]) == 64 for row in runs)
        for target in ("figures/algorithm.png", "figures/convergence.png", "figures/distribution.png"):
            assert (result_dir / target).exists()

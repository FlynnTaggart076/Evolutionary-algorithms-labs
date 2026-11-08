import numpy as np
import pytest
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
import ga as ga_module

from ga import GAConfig, genetic_search, random_search, reflect
from objectives import get_problem


@pytest.mark.parametrize("variant",[1,2,3,4,5,7,8,9,11,12,13,15,16,17,18,19,20])
def test_objective_at_known_minimum(variant):
    problem = get_problem(variant)
    d = problem.dimension
    point = np.zeros(d)
    if variant in [2,7]:
        point[:] = 1
    elif variant == 9:
        point = 2.0 ** (-(2.0 ** np.arange(d)-1) / 2.0 ** np.arange(d))
    elif variant == 12:
        point = np.sqrt(np.arange(1,d+1))
    elif variant in [19,20]:
        point[:] = -1
    assert problem.evaluate(point[None,:])[0] == pytest.approx(0,abs=1e-10)


def test_reflection_large_overshoots_and_boundaries():
    points = np.array([-1000.25,-2,-1,0,1,2,1000.25])
    repaired = reflect(points,-1,1)
    assert np.all((-1<=repaired)&(repaired<=1))
    np.testing.assert_allclose(repaired,[-0.25,0,-1,0,1,0,0.25])
    np.testing.assert_allclose(reflect(np.array([-1.,0.,1.]),-1,1),[-1,0,1])


def test_exact_budget_elitism_and_reproducibility():
    problem = get_problem(1)
    config = GAConfig(population=30,elite=2)
    one = genetic_search(problem,42,217,config)
    two = genetic_search(problem,42,217,config)
    assert one.evaluations == two.evaluations == 217
    assert one.history == two.history
    np.testing.assert_array_equal(one.x,two.x)
    assert np.all(np.diff([h["best"] for h in one.history])<=0)
    assert problem.evaluate(one.x[None,:])[0] == one.value
    baseline = random_search(problem,42,217,[h["evaluations"] for h in one.history])
    assert baseline.evaluations == 217
    assert baseline.history[0]["best"] == one.history[0]["best"]
    assert np.all(np.diff([h["best"] for h in baseline.history])<=0)


def test_budget_too_small_is_rejected():
    with pytest.raises(ValueError):
        genetic_search(get_problem(1),1,29,GAConfig(population=30))


def test_budget_matches_actual_objective_calls():
    class CountingProblem:
        def __init__(self):
            self.problem = get_problem(1)
            self.count = 0

        def __getattr__(self,name):
            return getattr(self.problem,name)

        def evaluate(self,points):
            self.count += len(points)
            return self.problem.evaluate(points)

    problem = CountingProblem()
    result = genetic_search(problem,5,217,GAConfig(population=30))
    assert problem.count == result.evaluations == 217
    problem.count = 0
    result = random_search(problem,5,217,[row["evaluations"] for row in result.history])
    assert problem.count == result.evaluations == 217


@pytest.mark.parametrize("field", ["mutation_scale", "blx_alpha", "crossover_probability"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_operator_parameters_are_rejected_before_evaluation(field, value):
    class NoEvaluation:
        def evaluate(self, points):
            pytest.fail("Invalid parameters must be rejected before fitness evaluation")

    with pytest.raises(ValueError, match="finite"):
        genetic_search(NoEvaluation(), 1, 130, replace(GAConfig(), **{field:value}))


@pytest.mark.parametrize("budget", [31, 32, 59, 217])
def test_partial_terminal_batch_preserves_configured_population_size(budget, monkeypatch):
    concatenate = np.concatenate
    population_sizes = []

    def observe_population(arrays, *args, **kwargs):
        result = concatenate(arrays, *args, **kwargs)
        if result.ndim == 2 and result.shape[1] == 6:
            population_sizes.append(len(result))
        return result

    monkeypatch.setattr(ga_module.np, "concatenate", observe_population)
    result = genetic_search(get_problem(1), 42, budget, GAConfig(population=30,elite=2))
    assert result.evaluations == budget
    assert population_sizes and set(population_sizes) == {30}
    assert np.all(np.diff([row['best'] for row in result.history]) <= 0)


def load_runner():
    specification = importlib.util.spec_from_file_location("lab1_test_runner", Path(__file__).with_name("run.py"))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_duplicate_comparison_factor_is_rejected_before_writing_results(tmp_path, monkeypatch):
    runner = load_runner()
    config = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))
    config["comparison_mutation_scale"] = config["ga"]["mutation_scale"]
    source = tmp_path/"config.json"
    source.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path/"output"
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(source), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not output.exists()


def test_other_variant_config_is_not_presented_as_the_assigned_experiment(tmp_path, monkeypatch):
    runner = load_runner()
    config = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))
    config['variant'] = 14
    source = tmp_path/'config.json'
    source.write_text(json.dumps(config), encoding='utf-8')
    output = tmp_path/'output'
    monkeypatch.setattr(sys,'argv',['run.py','--config',str(source),'--output',str(output)])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("runs,budget,population,elite,tournament,probability,alpha", [
    (2, 217, 30, 1, 5, 0.2, 0.7),
    (1, 30000, 100, 2, 3, 0.9, 0.3),
])
def test_report_uses_actual_parameters_and_actual_success_counts(
    tmp_path, runs, budget, population, elite, tournament, probability, alpha,
):
    runner = load_runner()
    config = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))
    ga = GAConfig(population=population, elite=elite, tournament=tournament,
                  crossover_probability=probability, blx_alpha=alpha)
    config.update({"runs":runs, "budget":budget, "ga":asdict(ga)})
    problem = get_problem(1)
    methods = ["GA sigma=0.05", "GA sigma=0.15", "random"]
    results = []
    for seed in range(1000,1000+runs):
        first = genetic_search(problem,seed,budget,ga)
        second = genetic_search(problem,seed,budget,replace(ga,mutation_scale=0.15))
        baseline = random_search(problem,seed,budget,[h['evaluations'] for h in first.history])
        results.extend(zip(methods,[seed]*3,[first,second,baseline]))
    summaries = []
    for method in methods:
        selected = [r[2] for r in results if r[0]==method]
        values = np.array([r.value for r in selected])
        summaries.append({"method":method,"runs":runs,"best":values.min(),"mean":values.mean(),
                          "median":np.median(values),"std":values.std(ddof=1) if runs>1 else 0.,
                          "worst":values.max(),"success_rate":np.mean(values<=1e-6),
                          "mean_seconds":np.mean([r.seconds for r in selected])})
    output = tmp_path/"report.md"
    runner.make_report(problem,config,summaries,results,output)
    text = output.read_text(encoding="utf-8")
    assert f"турниром из {tournament} особей" in text
    assert f"α = {alpha}" in text
    assert f"Его вероятность равна {probability}" in text
    assert f"Число переносимых элитных особей: {elite}" in text
    assert f"первые {population} точек" in text
    for summary in summaries[:2]:
        count = int(round(summary['success_rate']*runs))
        assert f"{count} из {runs} запусков" in text
    assert "во всех 20 запусках" not in text


@pytest.mark.parametrize("field,value", [
    ("runs", 1.5), ("runs", True), ("seed_start", -1), ("budget", 29),
    ("success_tolerance", float("nan")), ("success_tolerance", float("inf")),
    ("success_tolerance", -1.), ("group", "507"), ("student_number", 12),
])
def test_invalid_experiment_inputs_rejected_before_output_creation(tmp_path, monkeypatch, field, value):
    runner = load_runner()
    config = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))
    config[field] = value
    source = tmp_path / "config.json"
    source.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", str(source), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("budget,checkpoints", [(30.5, [30.5]), (True, [True]), (30, [30.0])])
def test_random_search_rejects_noninteger_budget_and_checkpoints_before_sampling(budget, checkpoints):
    class NoSampling:
        @property
        def lower(self):
            pytest.fail("Bad budget or checkpoints must be rejected before sampling")
    with pytest.raises(ValueError, match="integer"):
        random_search(NoSampling(), 1, budget, checkpoints)


@pytest.mark.parametrize("seed", [None, -1, 1.5, True])
def test_both_searches_require_reproducible_integer_seed(seed):
    with pytest.raises(ValueError, match="Seed"):
        genetic_search(get_problem(1), seed, 30, GAConfig(population=30))
    with pytest.raises(ValueError, match="Seed"):
        random_search(get_problem(1), seed, 30, [30])


@pytest.mark.parametrize("variant", [True, 1.0, 1.5])
def test_objective_variant_requires_integer_identifier(variant):
    with pytest.raises(ValueError, match="Variant"):
        get_problem(variant)

"""Regression checks for report provenance, CSV IDs and invalid parameters."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from portfolio import Problem, metrics, optimize

SPEC = importlib.util.spec_from_file_location("lab3_report_runner", Path(__file__).with_name("main.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def config():
    return json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["project", "effect", "cost", "risk", "staff"])
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("changes", [
    {"weights": []}, {"weights": [[np.nan, .5, .5]]},
    {"weights": [[0, 0, 0]]}, {"weights": [[-.1, .6, .5]]},
    {"crossover_probability": -.1}, {"mutation_probability": 1.1},
    {"population": 0}, {"evaluations": 399},
])
def test_reject_invalid_search_config(changes):
    cfg = {**config(), **changes}
    with pytest.raises(ValueError):
        runner.validate_config(cfg)


def test_csv_order_does_not_change_project_identity(tmp_path):
    cfg = config()
    rows = runner.generated_rows(cfg)
    path = tmp_path / "reordered.csv"
    write_rows(path, list(reversed(rows)))
    problem = runner.load_problem(cfg, path)
    expected = np.array([[row[k] for k in ["effect", "cost", "risk", "staff"]] for row in rows])
    np.testing.assert_array_equal(problem.data, expected)
    assert problem.source_info["matches_configured_generator"]
    rows[0]["project"] = rows[1]["project"]
    write_rows(path, rows)
    with pytest.raises(ValueError, match="unique"):
        runner.load_problem(cfg, path)


def test_nonfinite_csv_is_rejected(tmp_path):
    cfg = config()
    rows = runner.generated_rows(cfg)
    rows[0]["risk"] = "NaN"
    path = tmp_path / "nan.csv"
    write_rows(path, rows)
    with pytest.raises(ValueError, match="finite"):
        runner.load_problem(cfg, path)


def test_report_uses_actual_problem_weights_and_provenance(tmp_path, monkeypatch):
    cfg = {**config(), "projects": 3, "cost_limit": 10, "staff_limit": 2,
           "data_seed": 110518, "dependencies": [], "weights": [[.8, .1, .1], [.1, .8, .1]], "runs": 1}
    problem = Problem(np.array([[10.4,3.2,5.1,1],[20.3,6.1,4.2,1],[30.2,9.3,3.3,1]]),10,2,
                      source_info={"matches_configured_generator":False})
    population = np.eye(3,dtype=bool)
    values = problem.evaluate(population)
    archives = {label:(population,values) for label in ["NSGA-II","Weighted GA"]}
    stats = {"igd_min":0,"igd_mean":0,"igd_median":0,"igd_std":0,"igd_max":0,
             "coverage_mean":1,"epsilon_mean":0}
    summary = {label:dict(stats) for label in archives}
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    output = tmp_path/"results"
    output.mkdir()
    runner.report(output,cfg,problem,summary,values,7,archives)
    text = (tmp_path/"report.md").read_text(encoding="utf-8")
    assert "Σcᵢxᵢ ≤ 10; Σsᵢxᵢ ≤ 2" in text
    assert "2^3 = 8" in text and "объединяет 2 скалярных" in text
    assert "seed = 110517" not in text and "происхождение не приписывается" in text
    assert "вероятностью 0.35" in text and "Турнир включает два" in text
    assert "делит бюджет между 2 поисками" in text and "нельзя приписать только способу отбора" in text
    assert "| Экономичный | 1 | 10.4 | 3.2 | 5.1 |" in text
    assert "Для одиночного запуска разброс не оценивается" in text
    reps = json.loads((output/"representatives.json").read_text(encoding="utf-8"))
    assert len(reps)==len({tuple(row["projects"]) for row in reps})==3


def test_repair_supports_feasible_dependency_cycle():
    problem = Problem(np.array([[5.,1,2,1],[7,1,1,1]]),2,2,((0,1),(1,0)))
    repaired = problem.repair(np.zeros((1,2),dtype=bool),np.random.default_rng(3))
    assert repaired.all() and problem.feasible(repaired).all()
    with pytest.raises(ValueError, match="binary"):
        problem.feasible([2,1])


def test_metrics_count_unique_vectors_and_reject_nonfinite():
    duplicate = np.array([[0.,1],[0,1]])
    result = metrics(duplicate,np.array([[0.,1]]),np.ones(2))
    assert result["exact_coverage"]==result["precision"]==1 and result["front_size"]==1
    with pytest.raises(ValueError):
        metrics(np.array([[np.nan,1]]),duplicate,np.ones(2))
    with pytest.raises(ValueError):
        metrics(duplicate,duplicate,np.array([0.,1.]))


def test_direct_optimizer_rejects_invalid_probability():
    problem = Problem(np.array([[5.,1,2,1],[7,1,1,1]]),2,2)
    with pytest.raises(ValueError, match="probabilities"):
        optimize(problem,1,10,2,-.1,.5)


def test_external_cwd_and_alternate_output_preserve_report(tmp_path):
    lab = Path(__file__).resolve().parent
    report_path = lab/"report.md"
    before = report_path.read_bytes() if report_path.exists() else None
    result = subprocess.run([sys.executable,str(lab/"main.py"),"--runs","1","--budget","403",
                             "--output",str(tmp_path/"results")],cwd=tmp_path,
                            capture_output=True,text=True,encoding="utf-8",timeout=30)
    assert result.returncode==0,result.stderr
    after = report_path.read_bytes() if report_path.exists() else None
    assert after==before
    with (tmp_path/"results"/"runs.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows)==2 and all(int(row["evaluations"])==403 for row in rows)
    assert (tmp_path/"results"/"data_manifest.json").exists()


def test_cli_rejects_empty_weights_without_traceback(tmp_path):
    cfg = {**config(),"weights":[]}
    path = tmp_path/"invalid.json"
    path.write_text(json.dumps(cfg),encoding="utf-8")
    script = Path(__file__).with_name("main.py").resolve()
    result = subprocess.run([sys.executable,str(script),"--config",str(path),"--output",str(tmp_path/"out")],
                            cwd=tmp_path,capture_output=True,text=True,encoding="utf-8",timeout=30)
    assert result.returncode==2 and "weights" in result.stderr and "Traceback" not in result.stderr

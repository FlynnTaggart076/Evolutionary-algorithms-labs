"""Independently validate the four submitted experiment/result/PDF packages.

This checks saved evidence without calling the laboratory optimizers. Full seed
replays and operator checks belong to the separate audit and unit-test suites.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from pypdf import PdfReader
import pymupdf

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, context):
    check(np.isclose(float(actual), float(expected), rtol=1e-10, atol=1e-12),
          f"{context}: {actual} != {expected}")


def stats(values):
    a = np.array(values, dtype=float)
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
            "median": float(np.median(a)), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.}


def compare_stats(summary, rows, value_key, fields, context):
    computed = stats([float(row[value_key]) for row in rows])
    for stored, calculated in fields.items():
        close(summary[stored], computed[calculated], f"{context}/{stored}")


def common(lab, config, key, repeats, budget, minimum):
    for name in ["README.md", "config.json", "report.md", "report.pdf"]:
        check((lab / name).is_file(), f"Missing {lab.name}/{name}")
    check(repeats >= minimum, f"Too few independent runs in {lab.name}")
    rows = read_csv(lab / "results/runs.csv")
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        pair = (row[key], int(row["seed"]))
        check(pair not in seen, f"Duplicate run {lab.name}/{pair}")
        seen.add(pair)
        check(int(row["evaluations"]) == budget, f"Wrong budget {lab.name}/{pair}")
        check(float(row["seconds"]) >= 0, f"Invalid time {lab.name}/{pair}")
        groups[row[key]].append(row)
    check(all(len(group) == repeats for group in groups.values()), f"Run counts in {lab.name}")
    source = (lab / "report.md").read_text(encoding="utf-8")
    check(not re.search(r"\bTODO\b|\bTBD\b|PLACEHOLDER", source, flags=re.I), f"Unfinished report {lab.name}")
    for image in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", source):
        check((lab / image).is_file(), f"Missing report graphic {lab.name}/{image}")
    reader = PdfReader(lab / "report.pdf")
    expected_digest = "source-sha256:" + hashlib.sha256((lab / "report.md").read_bytes()).hexdigest()
    check(reader.metadata.get("/Keywords", "") == expected_digest,
          f"PDF is not synchronized with its Markdown source: {lab.name}")
    with pymupdf.open(lab / "report.pdf") as doc:
        check(len(reader.pages) == len(doc), f"PDF reader disagreement {lab.name}")
        for i, page in enumerate(doc):
            text = page.get_text()
            check(len(text) > 250, f"Empty/orphan PDF page {lab.name}/{i+1}")
            check("\ufffd" not in text and "\x00" not in text, f"Broken PDF glyphs {lab.name}/{i+1}")
            for block in page.get_text("dict")["blocks"]:
                if block["type"] == 0:
                    x0, y0, x1, y1 = block["bbox"]
                    check(x0 >= 20 and y0 >= 20 and x1 <= page.rect.width-20 and y1 <= page.rect.height-20,
                          f"Text outside safe bounds {lab.name}/{i+1}")
    return rows, groups, {"runs": len(rows), "methods": {k: len(v) for k, v in groups.items()},
                          "budget": budget, "pdf_pages": len(reader.pages)}


def histories(path, rows, method_key, score_key, increasing):
    expected = {(r[method_key], int(r["seed"])): r for r in rows}
    groups = defaultdict(list)
    for row in read_csv(path):
        groups[(row[method_key], int(row["seed"]))].append(row)
    check(set(groups) == set(expected), f"Missing/extra histories: {path}")
    for key, group in groups.items():
        evaluations = [int(r["evaluations"]) for r in group]
        scores = [float(r[score_key]) for r in group]
        check(all(b > a for a, b in zip(evaluations, evaluations[1:])), f"Evaluation ordering {key}")
        check(all((b >= a if increasing else b <= a) for a, b in zip(scores, scores[1:])), f"Best-so-far ordering {key}")
        check(evaluations[-1] == int(expected[key]["evaluations"]), f"Truncated history {key}")
        final_key = "best_feasible_value" if score_key == "best_feasible_value" else "best"
        close(scores[-1], expected[key][final_key], f"History final {key}")


def lab1():
    lab = ROOT / "Lab1"
    c = read_json(lab / "config.json")
    rows, groups, result = common(lab, c, "method", c["runs"], c["budget"], 20)
    check(len(groups) == 3, "Lab1 must have two GA settings and a random baseline")
    summary = {r["method"]: r for r in read_json(lab / "results/summary.json")}
    solutions = read_json(lab / "results/solutions.json")
    check(len(solutions) == len(rows), "Lab1 missing solutions")
    indexed = {(r["method"], int(r["seed"])): r for r in rows}
    for s in solutions:
        x = np.array(s["x"])
        check(x.shape == (6,) and np.all(np.abs(x) <= 100), "Lab1 invalid phenotype")
        value = float(sum(float(v)**2 for v in x))
        close(value, s["value"], "Lab1 saved objective")
        close(value, indexed[(s["method"], s["seed"])]["best"], "Lab1 CSV objective")
    for method, group in groups.items():
        compare_stats(summary[method], group, "best",
                      {"best": "min", "mean": "mean", "median": "median", "std": "std", "worst": "max"}, method)
    histories(lab / "results/histories.csv", rows, "method", "best", False)
    return result


def lab2():
    lab = ROOT / "Lab2"
    c = read_json(lab / "config.json")
    check(len(set(c["seeds"])) == len(c["seeds"]), "Lab2 duplicated independent seeds")
    rows, groups, result = common(lab, c, "method", len(c["seeds"]), c["budget"], 20)
    check(set(groups) == {m["name"] for m in c["methods"]}, "Lab2 methods differ from config")
    data = read_csv(lab / "data/items.csv")
    check(len(data) >= 30, "Lab2 too few objects")
    by_id = {r["item"]: r for r in data}
    check(len(by_id) == len(data), "Lab2 duplicate item IDs")
    reference = read_json(lab / "results/exact_reference.json")
    summary = {r["method"]: r for r in read_csv(lab / "results/summary.csv")}
    for method, group in groups.items():
        compare_stats(summary[method], group, "best_feasible_value",
                      {"best": "max", "mean": "mean", "median": "median", "sample_sd": "std", "worst": "min"}, method)
        check(all(float(r["best_feasible_value"]) <= reference["value"] for r in group), "Lab2 exceeds exact optimum")
        close(summary[method]["mean_feasible_fraction"], np.mean([float(r["feasible_fraction"]) for r in group]), method)
    best = read_json(lab / "results/best_solution.json")
    selected = [by_id[name] for name in best["items"]]
    for key in ["mass", "volume", "value"]:
        close(sum(int(r[key]) for r in selected), best[key], f"Lab2 best set {key}")
        close(sum(int(r[key]) for r in read_csv(lab / "results/best_set.csv")), best[key], f"Lab2 displayed set {key}")
    check(best["mass"] <= c["mass_limit"] and best["volume"] <= c["volume_limit"], "Lab2 best is infeasible")
    check(best["value"] == max(int(r["best_feasible_value"]) for r in rows), "Lab2 wrong best selection")
    histories(lab / "results/convergence.csv", rows, "method", "best_feasible_value", True)
    result["exact_optimum"] = reference["value"]
    return result


def lab3():
    lab = ROOT / "Lab3"
    c = read_json(lab / "config.json")
    rows, groups, result = common(lab, c, "algorithm", c["runs"], c["evaluations"], 20)
    check(set(groups) == {"NSGA-II", "Weighted GA"}, "Lab3 comparison methods")
    data = read_csv(lab / "data/projects.csv")
    coefficients = np.array([[float(r[k]) for k in ["effect", "cost", "risk", "staff"]] for r in data])
    n = len(data)
    scales = coefficients[:, :3].sum(axis=0)
    def objectives(csv_rows):
        values = []
        for row in csv_rows:
            ids = [int(i)-1 for i in row["projects"].split()]
            check(len(ids) == len(set(ids)) and all(0 <= i < n for i in ids) and ids, "Lab3 invalid portfolio IDs")
            x = np.zeros(n, dtype=int)
            x[ids] = 1
            total = x @ coefficients
            check(total[1] <= c["cost_limit"] and total[3] <= c["staff_limit"], "Lab3 resource violation")
            check(all(not x[child] or x[parent] for child, parent in c["dependencies"]), "Lab3 dependency violation")
            for k, value in zip(["effect", "cost", "risk"], total[:3]):
                close(row[k], value, "Lab3 reported phenotype")
            values.append([-total[0], total[1], total[2]])
        return np.array(values)
    exact = objectives(read_csv(lab / "results/exact_front.csv"))
    check(len({tuple(v) for v in exact}) == len(exact), "Lab3 exact front has duplicate vectors")
    masks = ((np.arange(2**n)[:,None] >> np.arange(n)) & 1)
    total = masks @ coefficients
    valid = (masks.sum(axis=1) > 0) & (total[:,1] <= c["cost_limit"]) & (total[:,3] <= c["staff_limit"])
    for child, parent in c["dependencies"]:
        valid &= (masks[:,child] <= masks[:,parent])
    all_f = total[valid,:3].copy()
    all_f[:,0] *= -1
    check(np.all(np.any(np.all(exact[None,:,:] <= all_f[:,None,:], axis=2), axis=1)), "Lab3 exact front misses a true optimum")
    check(not np.any(np.all(exact[:,None,:] <= exact[None,:,:],axis=2) & np.any(exact[:,None,:] < exact[None,:,:],axis=2)), "Lab3 exact front contains dominated vectors")
    exact_set = {tuple(v) for v in exact}
    for row in rows:
        method = "nsga2" if row["algorithm"] == "NSGA-II" else "weighted"
        f = objectives(read_csv(lab / f"results/front_{method}_{row['seed']}.csv"))
        check(len({tuple(v) for v in f}) == len(f), "Lab3 archive has duplicate vectors")
        normalized, reference = f/scales, exact/scales
        distances = np.linalg.norm(reference[:,None,:]-normalized[None,:,:],axis=2)
        close(row["igd"], distances.min(axis=1).mean(), "Lab3 IGD")
        close(row["epsilon"], np.max(np.min(np.max(normalized[None,:,:]-reference[:,None,:],axis=2),axis=1)), "Lab3 epsilon")
        hits = sum(tuple(v) in exact_set for v in f)
        close(row["exact_coverage"], hits/len(exact), "Lab3 coverage")
        close(row["precision"], hits/len(f), "Lab3 precision")
        check(len(f) == int(row["front_size"]), "Lab3 archive size")
    summary = read_json(lab / "results/summary.json")
    for method, group in groups.items():
        for key in ["igd", "epsilon", "exact_coverage", "precision", "front_size", "seconds"]:
            for suffix, value in stats([float(r[key]) for r in group]).items():
                close(summary[method][key+"_"+suffix], value, f"Lab3/{method}/{key}")
    result.update(exact_front_size=len(exact), feasible_portfolios=int(valid.sum()))
    return result


def lab4():
    lab = ROOT / "Lab4"
    c = read_json(lab / "config.json")
    rows, groups, result = common(lab, c, "algorithm", c["runs"], c["budget"], 30)
    check(set(groups) == {"ga", "memetic", "memetic_sparse"}, "Lab4 comparison methods")
    s = read_json(lab / "results/summary.json")
    for summary in s["summary"]:
        compare_stats(summary, groups[summary["algorithm"]], "best",
                      {"best": "min", "mean": "mean", "median": "median", "std": "std", "worst": "max"}, "Lab4")
    points = np.array([[float(r["x"]),float(r["y"])] for r in read_csv(lab / "data.csv")])
    d = np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2)
    best = read_json(lab / "results/best_solution.json")
    tour = best["tour"]
    check(sorted(tour) == list(range(len(points))), "Lab4 best permutation")
    length = sum(float(d[tour[i],tour[(i+1)%len(tour)]]) for i in range(len(tour)))
    close(length, best["best"], "Lab4 tour recomputed")
    close(length, min(float(r["best"]) for r in rows), "Lab4 global best")
    displayed = read_csv(lab / "results/best_route.csv")
    check([int(r["city"]) for r in displayed] == tour, "Lab4 displayed order")
    close(sum(float(r["edge_length"]) for r in displayed), length, "Lab4 displayed edge sum")
    for i, row in enumerate(displayed):
        check(int(row["next_city"]) == tour[(i+1)%len(tour)], "Lab4 route not closed")
    histories(lab / "results/traces.csv", rows, "algorithm", "best", False)
    result["pilot_runs"] = len(read_csv(lab / "results/pilot.csv"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optionally save the verified result manifest")
    args = parser.parse_args()
    results = {f"Lab{i}": func() for i, func in enumerate([lab1, lab2, lab3, lab4], 1)}
    hashes = {}
    for lab in [ROOT / f"Lab{i}" for i in range(1,5)]:
        for path in sorted(lab.rglob("*")):
            if any(part in {"__pycache__", ".pytest_cache", ".qa"} for part in path.relative_to(lab).parts):
                continue
            if path.is_file() and path.suffix in {".py", ".json", ".md", ".pdf", ".csv", ".png"}:
                hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [ROOT / "reproduce.py", ROOT / "requirements.txt", ROOT / "README.md", *sorted((ROOT / "tools").glob("*.py"))]:
        hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"status": "PASS", "total_main_runs": sum(r["runs"] for r in results.values()),
              "checks": results, "sha256": hashes}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")
    print(json.dumps({k:v for k,v in result.items() if k != "sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

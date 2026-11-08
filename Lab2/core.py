"""Binary genetic algorithm for a knapsack with mass and volume constraints."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter

import numpy as np


def positive_integer(value, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def nonnegative_finite(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")


def validate_search_parameters(config: dict, method: dict) -> None:
    positive_integer(config["population"], "population", minimum=2)
    positive_integer(config["budget"], "budget")
    if config["budget"] < config["population"] or config["budget"] % config["population"]:
        raise ValueError("The positive budget must be a multiple of population >=2.")
    positive_integer(config["tournament"], "tournament")
    nonnegative_finite(config["penalty"], "penalty")
    for key in ("crossover_probability", "mutation_probability"):
        nonnegative_finite(config[key], key)
        if config[key] > 1:
            raise ValueError(f"{key} must be in [0,1].")
    if method["constraint"] not in {"repair", "penalty"}:
        raise ValueError("Unknown constraint strategy.")
    if method["algorithm"] not in {"ga", "random"}:
        raise ValueError("Unknown search algorithm.")
    if method["algorithm"] == "ga" and method["crossover"] not in {"uniform", "two_point"}:
        raise ValueError("Unknown crossover.")
    if not isinstance(method["name"], str) or not method["name"].strip():
        raise ValueError("Method name must be nonempty.")


@dataclass(frozen=True)
class Problem:
    names: tuple[str, ...]
    mass: np.ndarray
    volume: np.ndarray
    value: np.ndarray
    mass_limit: int
    volume_limit: int

    def __post_init__(self) -> None:
        if not self.names or any(not isinstance(name, str) or not name.strip() for name in self.names):
            raise ValueError("Item identifiers must be nonempty strings.")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Item identifiers must be unique.")
        for key in ("mass_limit", "volume_limit"):
            positive_integer(getattr(self, key), key)
            if int(getattr(self, key)) > np.iinfo(np.int64).max:
                raise ValueError(f"{key} must fit signed 64-bit integer arithmetic.")
            object.__setattr__(self, key, int(getattr(self, key)))
        for key in ("mass", "volume", "value"):
            values = np.asarray(getattr(self, key))
            if values.shape != (len(self.names),) or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"{key} must contain one integer per item.")
            if np.any(values < 0) or (key != "value" and np.any(values == 0)):
                raise ValueError("Mass and volume must be positive; item values must be nonnegative.")
            # NumPy preserves narrow integer dtypes in dot products: int8 sums
            # can otherwise turn a resource overload into a negative total.
            if sum(int(value) for value in values) > np.iinfo(np.int64).max:
                raise ValueError(f"The total {key} must fit signed 64-bit integer arithmetic.")
            object.__setattr__(self, key, values.astype(np.int64, copy=True))

    @property
    def n(self) -> int:
        return len(self.names)

    @classmethod
    def load(cls, path: Path, mass_limit: int, volume_limit: int) -> "Problem":
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"item", "mass", "volume", "value"}.issubset(reader.fieldnames or ()):
                raise ValueError("Data CSV must have item,mass,volume,value columns.")
            rows = list(reader)
        return cls(
            tuple(row["item"] for row in rows),
            np.array([int(row["mass"]) for row in rows], dtype=np.int64),
            np.array([int(row["volume"]) for row in rows], dtype=np.int64),
            np.array([int(row["value"]) for row in rows], dtype=np.int64),
            mass_limit, volume_limit,
        )

    def validate(self, x: np.ndarray) -> None:
        x = np.asarray(x)
        if x.shape != (self.n,) or not np.all((x == 0) | (x == 1)):
            raise ValueError("A solution must be a binary vector with one gene per item.")

    def describe(self, x: np.ndarray) -> dict:
        self.validate(x)
        x = np.asarray(x, dtype=bool)
        mass = int(x @ self.mass)
        volume = int(x @ self.volume)
        return {
            "items": [name for name, present in zip(self.names, x) if present],
            "bits": [int(bit) for bit in x],
            "mass": mass, "volume": volume, "value": int(x @ self.value),
            "mass_excess": max(0, mass - self.mass_limit),
            "volume_excess": max(0, volume - self.volume_limit),
            "feasible": mass <= self.mass_limit and volume <= self.volume_limit,
        }


def repair(x: np.ndarray, problem: Problem) -> np.ndarray:
    """Remove low-density selected items until both resource bounds are met.

    Density uses the input item values, not an extra evaluation of a complete
    candidate. Running resource totals are updated after each removed item.
    """
    problem.validate(x)
    result = np.asarray(x, dtype=bool).copy()
    mass = int(result @ problem.mass)
    volume = int(result @ problem.volume)
    density = problem.value / (
        problem.mass / problem.mass_limit + problem.volume / problem.volume_limit
    )
    for index in np.argsort(density, kind="stable"):
        if mass <= problem.mass_limit and volume <= problem.volume_limit:
            break
        if result[index]:
            result[index] = False
            mass -= int(problem.mass[index])
            volume -= int(problem.volume[index])
    return result


def crossover(
    first: np.ndarray, second: np.ndarray, kind: str, rng: np.random.Generator,
) -> np.ndarray:
    """Binary uniform and two-point crossovers preserve the gene alphabet."""
    if kind == "uniform":
        return np.where(rng.random(first.shape) < 0.5, first, second)
    if kind == "two_point":
        left, right = sorted(rng.choice(len(first) + 1, size=2, replace=False))
        child = first.copy()
        child[left:right] = second[left:right]
        return child
    raise ValueError(f"Unknown crossover: {kind}")


class Evaluator:
    """Count every processed candidate, including repeated bit strings."""

    def __init__(self, problem: Problem, budget: int, penalty: float):
        positive_integer(budget, "budget")
        nonnegative_finite(penalty, "penalty")
        self.problem = problem
        self.budget = budget
        self.penalty = penalty
        self.calls = 0
        self.feasible_calls = 0
        self.best_value = -1
        self.best_x: np.ndarray | None = None

    def evaluate(self, population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        population = np.asarray(population)
        if population.ndim != 2 or population.shape[1] != self.problem.n or len(population) == 0:
            raise ValueError("Invalid population shape.")
        if self.calls + len(population) > self.budget:
            raise ValueError("The fitness budget would be exceeded.")
        if not np.all((population == 0) | (population == 1)):
            raise ValueError("The population must be binary.")
        population = population.astype(bool, copy=False)
        mass = population @ self.problem.mass
        volume = population @ self.problem.volume
        value = population @ self.problem.value
        mass_excess = np.maximum(mass - self.problem.mass_limit, 0)
        volume_excess = np.maximum(volume - self.problem.volume_limit, 0)
        feasible = (mass_excess == 0) & (volume_excess == 0)
        # Each integer resource total is exact. Convert only the penalty sum:
        # adding two individually valid int64 excesses can overflow int64.
        violation = mass_excess.astype(float) + volume_excess.astype(float)
        self.calls += len(population)
        self.feasible_calls += int(np.count_nonzero(feasible))
        candidate_values = np.where(feasible, value, -1)
        index = int(np.argmax(candidate_values))
        if int(candidate_values[index]) > self.best_value:
            self.best_value = int(candidate_values[index])
            self.best_x = population[index].copy()
        return value - self.penalty * violation, feasible


def solve(problem: Problem, config: dict, method: dict, seed: int) -> dict:
    """Run either an elitist (mu+lambda) GA or independent repaired sampling."""
    validate_search_parameters(config, method)
    positive_integer(seed, "seed", minimum=0)
    population_size = config["population"]
    budget = config["budget"]
    rng = np.random.default_rng(seed)
    evaluator = Evaluator(problem, budget, float(config["penalty"]))
    history = []
    started = perf_counter()

    def prepare(population: np.ndarray) -> np.ndarray:
        if method["constraint"] == "repair":
            return np.array([repair(x, problem) for x in population], dtype=bool)
        return population

    population = rng.random((population_size, problem.n)) < 0.5
    population[0] = False  # A known feasible anchor, also used by the baseline.
    population = prepare(population)
    score, _ = evaluator.evaluate(population)
    history.append((evaluator.calls, evaluator.best_value))
    while evaluator.calls < budget:
        if method["algorithm"] == "random":
            children = rng.random((population_size, problem.n)) < 0.5
        elif method["algorithm"] == "ga":
            children = np.empty_like(population)
            for index in range(population_size):
                parents = []
                for _ in range(2):
                    candidates = rng.integers(0, population_size, size=config["tournament"])
                    parents.append(population[candidates[np.argmax(score[candidates])]])
                child = parents[0].copy()
                if rng.random() < config["crossover_probability"]:
                    child = crossover(parents[0], parents[1], method["crossover"], rng)
                child ^= rng.random(problem.n) < config["mutation_probability"]
                children[index] = child
        else:
            raise ValueError("Unknown search algorithm.")
        children = prepare(children)
        child_score, _ = evaluator.evaluate(children)
        if method["algorithm"] == "ga":
            combined = np.concatenate((population, children))
            combined_score = np.concatenate((score, child_score))
            keep = np.argsort(-combined_score, kind="stable")[:population_size]
            population = combined[keep]
            score = combined_score[keep]
        else:
            population, score = children, child_score
        history.append((evaluator.calls, evaluator.best_value))
    seconds = perf_counter() - started
    selected_index = int(np.argmax(score))
    selected = problem.describe(population[selected_index])
    assert evaluator.best_x is not None
    # Independently recompute the archived feasible objective and both constraints.
    verified = problem.describe(evaluator.best_x)
    if not verified["feasible"] or verified["value"] != evaluator.best_value:
        raise AssertionError("The feasible incumbent archive is inconsistent.")
    return {
        "method": method["name"], "seed": seed,
        "best_feasible_value": evaluator.best_value,
        "evaluations": evaluator.calls,
        "feasible_fraction": evaluator.feasible_calls / evaluator.calls,
        "seconds": seconds,
        "selected_feasible": int(selected["feasible"]),
        "selected_value": selected["value"],
        "selected_score": float(score[selected_index]),
        "selected_mass_excess": selected["mass_excess"],
        "selected_volume_excess": selected["volume_excess"],
        "solution": verified, "history": history,
    }


def exact_dynamic_programming(problem: Problem) -> dict:
    """Independent integer resource DP used only as a quality reference."""
    table = np.zeros((problem.mass_limit + 1, problem.volume_limit + 1), dtype=np.int64)
    choices = np.zeros((problem.n, *table.shape), dtype=bool)
    for index, (mass, volume, value) in enumerate(zip(problem.mass, problem.volume, problem.value)):
        if mass > problem.mass_limit or volume > problem.volume_limit:
            continue
        previous = table.copy()
        candidate = previous[:-mass, :-volume] + value
        improved = candidate > previous[mass:, volume:]
        table[mass:, volume:] = np.maximum(previous[mass:, volume:], candidate)
        choices[index, mass:, volume:] = improved
    mass, volume = problem.mass_limit, problem.volume_limit
    selected = np.zeros(problem.n, dtype=bool)
    for index in range(problem.n - 1, -1, -1):
        if choices[index, mass, volume]:
            selected[index] = True
            mass -= int(problem.mass[index])
            volume -= int(problem.volume[index])
    result = problem.describe(selected)
    assert result["feasible"] and result["value"] == int(table[-1, -1])
    return result

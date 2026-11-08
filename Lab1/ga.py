"""Генетический и случайный поиск с точным учётом вычислений функции."""
from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter

import numpy as np

from objectives import Problem


@dataclass(frozen=True)
class GAConfig:
    population: int = 100
    elite: int = 2
    tournament: int = 3
    crossover_probability: float = 0.9
    blx_alpha: float = 0.3
    mutation_scale: float = 0.05

    def validate(self) -> None:
        """Reject malformed parameters before sampling or evaluating candidates."""
        for name in ("population", "elite", "tournament"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.population < 30 or not 1 <= self.elite < self.population:
            raise ValueError("population >= 30 and 1 <= elite < population required")
        for name in ("crossover_probability", "blx_alpha", "mutation_scale"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.tournament < 1 or self.mutation_scale <= 0 or not 0 <= self.crossover_probability <= 1 or self.blx_alpha < 0:
            raise ValueError("Invalid operator parameters")


@dataclass
class SearchResult:
    x: np.ndarray
    value: float
    evaluations: int
    seconds: float
    history: list[dict]


def reflect(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Repeated reflection, valid even for arbitrarily large overshoots."""
    width = upper - lower
    if width <= 0:
        raise ValueError("Empty interval")
    y = np.remainder(x - lower, 2 * width)
    return lower + np.where(y <= width, y, 2 * width - y)


def tournament_select(population: np.ndarray, values: np.ndarray, n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    candidates = rng.integers(0, len(population), (n, size))
    winners = candidates[np.arange(n), np.argmin(values[candidates], axis=1)]
    return population[winners].copy()


def genetic_search(problem: Problem, seed: int, budget: int, config: GAConfig) -> SearchResult:
    config.validate()
    if isinstance(budget, bool) or not isinstance(budget, Integral):
        raise ValueError("Budget must be an integer")
    if budget < config.population:
        raise ValueError("Budget must cover initialization")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    start = perf_counter()
    rng = np.random.default_rng(seed)
    p = rng.uniform(problem.lower, problem.upper, (config.population, problem.dimension))
    values = problem.evaluate(p)
    evaluations, generation = len(p), 0
    history = [{"generation": generation, "evaluations": evaluations, "best": float(values.min())}]
    while evaluations < budget:
        count = min(config.population - config.elite, budget - evaluations)
        elites = np.argsort(values)[:config.elite]
        first = tournament_select(p, values, count, config.tournament, rng)
        second = tournament_select(p, values, count, config.tournament, rng)
        low, high = np.minimum(first, second), np.maximum(first, second)
        span = high - low
        children = rng.uniform(low - config.blx_alpha * span, high + config.blx_alpha * span)
        crossover = rng.random((count, 1)) < config.crossover_probability
        children = np.where(crossover, children, first)
        progress = evaluations / budget
        # Prespecified deterministic annealing; no fitness-dependent adaptation.
        scale = config.mutation_scale * (0.0001 + 0.9999 * (1 - progress) ** 2)
        mutation = rng.random(children.shape) < 1 / problem.dimension
        children += mutation * rng.normal(0, scale * (problem.upper - problem.lower), children.shape)
        children = reflect(children, problem.lower, problem.upper)
        if not np.all(np.isfinite(children)):
            raise ValueError("Variation produced non-finite candidates; reduce operator parameters")
        child_values = problem.evaluate(children)
        evaluations += count
        # Keep the configured population size even when the terminal offspring
        # batch is partial. Fill unused positions with already evaluated,
        # non-elite parents; this consumes neither fitness calls nor RNG draws.
        missing = config.population - config.elite - count
        non_elites = np.ones(len(p), dtype=bool)
        non_elites[elites] = False
        retained = np.flatnonzero(non_elites)[:missing]
        p = np.concatenate([p[elites], children, p[retained]])
        values = np.concatenate([values[elites], child_values, values[retained]])
        assert len(p) == config.population
        generation += 1
        history.append({"generation": generation, "evaluations": evaluations, "best": float(values.min())})
    index = int(np.argmin(values))
    return SearchResult(p[index].copy(), float(values[index]), evaluations, perf_counter()-start, history)


def random_search(problem: Problem, seed: int, budget: int, checkpoints: list[int]) -> SearchResult:
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 1:
        raise ValueError("Budget must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if any(isinstance(point, bool) or not isinstance(point, Integral) for point in checkpoints):
        raise ValueError("Checkpoints must be integers")
    if not checkpoints or checkpoints[-1] != budget or checkpoints != sorted(set(checkpoints)):
        raise ValueError("Checkpoints must be strictly increasing and end at budget")
    start = perf_counter()
    rng = np.random.default_rng(seed)
    best, best_x, used = np.inf, None, 0
    history = []
    for generation, target in enumerate(checkpoints):
        if target <= used:
            raise ValueError("Invalid checkpoint")
        p = rng.uniform(problem.lower, problem.upper, (target-used, problem.dimension))
        values = problem.evaluate(p)
        index = int(np.argmin(values))
        if values[index] < best:
            best, best_x = float(values[index]), p[index].copy()
        used = target
        history.append({"generation": generation, "evaluations": used, "best": best})
    return SearchResult(best_x, best, used, perf_counter()-start, history)

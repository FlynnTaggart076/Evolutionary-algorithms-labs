"""Own permutation GA and memetic GA; every tested tour consumes one evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import random
from time import perf_counter

import numpy as np


def distance_matrix(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Require a finite two-dimensional coordinate table")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        delta = points[:, None, :] - points[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
        scale = np.max(np.abs(delta), axis=2)
        extreme = (scale > 0) & ((scale < 1e-150) | (scale > 1e150))
        # Keep ordinary-scale arithmetic unchanged for exact seed replays.
        distances[extreme] = np.hypot(delta[..., 0][extreme], delta[..., 1][extreme])
    if not np.isfinite(distances).all():
        raise ValueError("Distances exceed the floating-point range")
    return distances


def tour_length(tour: np.ndarray, distances: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(distances[tour, np.roll(tour, -1)].sum())
    if not np.isfinite(value):
        raise ValueError("Tour length exceeds the floating-point range")
    return value


def order_crossover(a: np.ndarray, b: np.ndarray, rng: random.Random) -> np.ndarray:
    """OX: retain an interval of a and circular order of remaining genes of b."""
    n = len(a)
    left, right = sorted(rng.sample(range(n), 2))
    right += 1
    child = np.full(n, -1, dtype=int)
    child[left:right] = a[left:right]
    used = set(map(int, a[left:right]))
    positions = list(range(right, n)) + list(range(left))
    values = [int(b[k % n]) for k in range(right, right + n) if int(b[k % n]) not in used]
    child[positions] = values
    return child


def inversion(tour: np.ndarray, rng: random.Random) -> np.ndarray:
    result = tour.copy()
    a, b = sorted(rng.sample(range(len(tour)), 2))
    result[a:b + 1] = result[a:b + 1][::-1]
    return result


def two_opt_neighbor(tour: np.ndarray, i: int, j: int) -> np.ndarray:
    """Replace edges (i,i+1), (j,j+1) by reversing tour[i+1:j+1]."""
    if not (0 <= i < j - 1 < len(tour) - 1) or (i == 0 and j == len(tour) - 1):
        raise ValueError("Invalid 2-opt edge pair")
    result = tour.copy()
    result[i + 1:j + 1] = result[i + 1:j + 1][::-1]
    return result


class BudgetExhausted(RuntimeError):
    pass


class Evaluator:
    def __init__(self, distances: np.ndarray, budget: int, checkpoint: int = 200):
        if not isinstance(budget, (int, np.integer)) or isinstance(budget, bool) or budget < 1:
            raise ValueError("budget must be a positive integer")
        if not isinstance(checkpoint, (int, np.integer)) or isinstance(checkpoint, bool) or checkpoint < 1:
            raise ValueError("checkpoint must be a positive integer")
        self.distances = distances
        self.budget = budget
        self.checkpoint = checkpoint
        self.evaluations = 0
        self.best = float("inf")
        self.best_tour: np.ndarray | None = None
        self.trace: list[tuple[int, float]] = []

    def __call__(self, tour: np.ndarray) -> float:
        if self.evaluations >= self.budget:
            raise BudgetExhausted
        value = tour_length(tour, self.distances)
        self.evaluations += 1
        if value < self.best:
            self.best, self.best_tour = value, tour.copy()
        if self.evaluations == 1 or self.evaluations % self.checkpoint == 0 or self.evaluations == self.budget:
            self.trace.append((self.evaluations, self.best))
        return value


def local_search(tour: np.ndarray, fitness: float, evaluate: Evaluator,
                 rng: random.Random, max_checks: int) -> tuple[np.ndarray, float]:
    """Random-order first improvement, bounded by full tour evaluations.

    No delta objective or uncounted objective calculation is used.
    """
    current, value = tour.copy(), fitness
    n = len(tour)
    pairs = [(i, j) for i in range(n - 2) for j in range(i + 2, n)
             if not (i == 0 and j == n - 1)]
    rng.shuffle(pairs)
    for i, j in pairs[:max_checks]:
        if evaluate.evaluations >= evaluate.budget:
            break
        candidate = two_opt_neighbor(current, i, j)
        score = evaluate(candidate)
        if score < value:
            current, value = candidate, score
    return current, value


@dataclass(frozen=True)
class Parameters:
    population: int = 80
    budget: int = 18000
    crossover_probability: float = 0.9
    mutation_probability: float = 0.3
    tournament_size: int = 3
    elites: int = 2
    local_probability: float = 0.2
    local_checks: int = 120
    trace_step: int = 200


def optimize(points: np.ndarray, seed: int, memetic: bool,
             parameters: Parameters = Parameters()) -> dict:
    p = parameters
    for name in ("population", "budget", "tournament_size", "elites", "local_checks", "trace_step"):
        value = getattr(p, name)
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Require a finite two-dimensional coordinate table")
    n = len(points)
    if n < 4 or p.population < 4 or p.budget < p.population:
        raise ValueError("Require n>=4, population>=4 and budget>=population")
    if not 1 <= p.elites < p.population or p.tournament_size < 1:
        raise ValueError("Invalid selection parameters")
    for probability in (p.crossover_probability, p.mutation_probability, p.local_probability):
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be in [0,1]")
    if p.local_checks < 0 or p.trace_step < 1:
        raise ValueError("Invalid local search or trace parameters")
    started = perf_counter()
    rng = random.Random(seed)
    evaluate = Evaluator(distance_matrix(points), p.budget, p.trace_step)
    population = [np.asarray(rng.sample(range(n), n), dtype=int) for _ in range(p.population)]
    fitness = [evaluate(tour) for tour in population]
    generations = 0
    local_evaluations = 0

    def parent() -> np.ndarray:
        candidates = [rng.randrange(p.population) for _ in range(p.tournament_size)]
        return population[min(candidates, key=fitness.__getitem__)]

    while evaluate.evaluations < p.budget:
        order = np.argsort(fitness)
        next_population = [population[int(i)].copy() for i in order[:p.elites]]
        next_fitness = [fitness[int(i)] for i in order[:p.elites]]
        while len(next_population) < p.population and evaluate.evaluations < p.budget:
            a, b = parent(), parent()
            child = order_crossover(a, b, rng) if rng.random() < p.crossover_probability else a.copy()
            if rng.random() < p.mutation_probability:
                child = inversion(child, rng)
            score = evaluate(child)
            if memetic and rng.random() < p.local_probability:
                before = evaluate.evaluations
                child, score = local_search(child, score, evaluate, rng, p.local_checks)
                local_evaluations += evaluate.evaluations - before
            next_population.append(child)
            next_fitness.append(score)
        # An incomplete last generation is unnecessary for the final best-so-far.
        if len(next_population) == p.population:
            population, fitness = next_population, next_fitness
        generations += 1
    return {
        "seed": seed, "algorithm": "memetic" if memetic else "ga",
        "best": evaluate.best, "tour": evaluate.best_tour.tolist(),
        "evaluations": evaluate.evaluations, "local_evaluations": local_evaluations,
        "generations": generations, "seconds": perf_counter() - started,
        "trace": evaluate.trace,
    }


def nearest_neighbor(points: np.ndarray) -> tuple[list[int], float]:
    """Constructive reference: best nearest-neighbor tour over every start vertex."""
    distances = distance_matrix(points)
    best_tour, best = [], float("inf")
    for start in range(len(points)):
        route, remaining = [start], set(range(len(points))) - {start}
        while remaining:
            nxt = min(remaining, key=lambda v: (distances[route[-1], v], v))
            remaining.remove(nxt)
            route.append(nxt)
        value = tour_length(np.asarray(route), distances)
        if value < best:
            best_tour, best = route, value
    return best_tour, best


def mst_lower_bound(points: np.ndarray) -> float:
    """Prim's algorithm. Deleting an edge from a tour yields a spanning tree."""
    d = distance_matrix(points)
    used = np.zeros(len(points), dtype=bool)
    cheapest = np.full(len(points), np.inf)
    cheapest[0] = 0
    total = 0.0
    for _ in points:
        u = int(np.argmin(np.where(used, np.inf, cheapest)))
        used[u] = True
        total += cheapest[u]
        cheapest = np.minimum(cheapest, d[u])
    return float(total)

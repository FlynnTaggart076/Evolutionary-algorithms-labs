"""Binary portfolio, explicit repair, Pareto sorting and NSGA-II operators."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Problem:
    data: np.ndarray  # columns: effect, cost, expected risk loss, staff
    cost_limit: float
    staff_limit: float
    dependencies: tuple[tuple[int, int], ...] = ()
    source_info: dict | None = None

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 2 or self.data.shape[1] != 4 or not len(self.data):
            raise ValueError("Project data must be a nonempty (n, 4) table")
        if not np.isfinite(self.data).all() or np.any(self.data < 0):
            raise ValueError("Project coefficients must be finite and nonnegative")
        if not np.isfinite([self.cost_limit, self.staff_limit]).all() or min(self.cost_limit, self.staff_limit) < 0:
            raise ValueError("Resource limits must be finite and nonnegative")
        for pair in self.dependencies:
            if len(pair) != 2 or any(not isinstance(i, (int, np.integer)) or isinstance(i, bool)
                                     or not 0 <= i < self.n for i in pair):
                raise ValueError("Dependencies must contain two valid zero-based project indices")

    def _population(self, x):
        rows = np.asarray(x)
        if rows.ndim not in (1, 2):
            raise ValueError("A portfolio must be a binary vector or a matrix of binary vectors")
        rows = np.atleast_2d(rows)
        if rows.shape[1] != self.n or not np.all((rows == 0) | (rows == 1)):
            raise ValueError("A portfolio must have one binary gene per project")
        return rows.astype(bool)

    @property
    def n(self):
        return len(self.data)

    @property
    def scales(self):
        # A zero criterion is constant: denominator one leaves it identically zero.
        totals = self.data[:, :3].sum(axis=0)
        return np.where(totals > 0, totals, 1.0)

    def feasible(self, x):
        x = self._population(x)
        good = (x.sum(axis=1) >= 1) & (x @ self.data[:, 1] <= self.cost_limit)
        good &= x @ self.data[:, 3] <= self.staff_limit
        for child, parent in self.dependencies:
            good &= (~x[:, child]) | x[:, parent]
        return good

    def repair(self, population, rng):
        """Delete dependent projects and random selected projects until feasible.

        Only constraint totals are computed here; no objectives are evaluated.
        """
        pop = self._population(population).copy()
        for x in pop:
            while True:
                changed = False
                for child, parent in self.dependencies:
                    if x[child] and not x[parent]:
                        x[child] = False
                        changed = True
                overloaded = (x @ self.data[:, 1] > self.cost_limit or
                              x @ self.data[:, 3] > self.staff_limit)
                if overloaded:
                    x[rng.choice(np.flatnonzero(x))] = False
                    changed = True
                if not changed:
                    break
            if not x.any():
                eligible = [j for j in range(self.n)
                            if all(child != j for child, _ in self.dependencies)
                            and self.data[j, 1] <= self.cost_limit
                            and self.data[j, 3] <= self.staff_limit]
                if not eligible:
                    # Cyclic dependencies may admit a group but no singleton.
                    groups = []
                    for start in range(self.n):
                        group = np.zeros(self.n, dtype=bool)
                        group[start] = True
                        changed = True
                        while changed:
                            changed = False
                            for child, parent in self.dependencies:
                                if group[child] and not group[parent]:
                                    group[parent] = True
                                    changed = True
                        if group @ self.data[:, 1] <= self.cost_limit and group @ self.data[:, 3] <= self.staff_limit:
                            groups.append(group)
                    if not groups:
                        raise ValueError("No feasible nonempty portfolio exists")
                    x[:] = groups[rng.choice(len(groups))]
                else:
                    x[rng.choice(eligible)] = True
        return pop

    def evaluate(self, x):
        """Each row is one objective-vector evaluation; all criteria minimize."""
        totals = self._population(x) @ self.data[:, :3]
        totals[:, 0] *= -1
        return totals


def nondominated_sort(values):
    """Deb's sorting: strict Pareto dominance, duplicate points do not dominate."""
    f = np.asarray(values)
    dominates = np.all(f[:, None, :] <= f[None, :, :], axis=2)
    dominates &= np.any(f[:, None, :] < f[None, :, :], axis=2)
    counts = dominates.sum(axis=0)
    ranks = np.full(len(f), -1, dtype=int)
    front = np.flatnonzero(counts == 0)
    fronts = []
    rank = 0
    while len(front):
        fronts.append(front)
        ranks[front] = rank
        counts -= dominates[front].sum(axis=0)
        counts[front] = -1
        front = np.flatnonzero(counts == 0)
        rank += 1
    return fronts, ranks


def crowding_distance(values, front):
    distances = np.zeros(len(front), dtype=float)
    if len(front) <= 2:
        distances[:] = np.inf
        return distances
    points = values[front]
    for j in range(points.shape[1]):
        order = np.argsort(points[:, j], kind="stable")
        span = points[order[-1], j] - points[order[0], j]
        if span == 0:
            continue
        distances[order[[0, -1]]] = np.inf
        distances[order[1:-1]] += (points[order[2:], j] - points[order[:-2], j]) / span
    return distances


def selection_data(values):
    fronts, ranks = nondominated_sort(values)
    distances = np.zeros(len(values))
    for front in fronts:
        distances[front] = crowding_distance(values, front)
    return fronts, ranks, distances


def environmental_selection(population, values, size):
    fronts, _, distances = selection_data(values)
    chosen = []
    for front in fronts:
        room = size - len(chosen)
        if len(front) <= room:
            chosen.extend(front)
        else:
            chosen.extend(front[np.argsort(-distances[front], kind="stable")[:room]])
            break
        if len(chosen) == size:
            break
    return population[chosen].copy(), values[chosen].copy()


def pareto_archive(population, values):
    """Incremental skyline avoids the quadratic matrix for large exact sets."""
    _, unique = np.unique(values, axis=0, return_index=True)
    indices = unique[np.argsort(values[unique, 0], kind="stable")]
    kept = []
    for index in indices:
        if kept:
            previous = values[kept]
            if np.any(np.all(previous <= values[index], axis=1)):
                continue
            survived = ~np.all(values[index] <= previous, axis=1)
            kept = [k for k, keep in zip(kept, survived) if keep]
        kept.append(index)
    return population[kept].copy(), values[kept].copy()


def update_archive(archive, pop, vals):
    if archive is None:
        return pareto_archive(pop, vals)
    return pareto_archive(np.vstack([archive[0], pop]), np.vstack([archive[1], vals]))


def breed(population, parent_indices, rng, crossover, mutation):
    parents = population[parent_indices]
    children = parents.copy()
    for i in range(0, len(children) - 1, 2):
        if rng.random() < crossover:
            mask = rng.random(population.shape[1]) < 0.5
            children[i] = np.where(mask, parents[i], parents[i + 1])
            children[i + 1] = np.where(mask, parents[i + 1], parents[i])
    children ^= rng.random(children.shape) < mutation
    return children


def optimize(problem, seed, budget, size, crossover, mutation, weights=None):
    """NSGA-II or one scalar GA; common variation, repair and full archive."""
    if not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size < 2:
        raise ValueError("population size must be an integer >= 2")
    if not isinstance(budget, (int, np.integer)) or isinstance(budget, bool) or budget < size:
        raise ValueError("budget must be an integer at least population size")
    if not np.isfinite([crossover, mutation]).all() or not (0 <= crossover <= 1 and 0 <= mutation <= 1):
        raise ValueError("Crossover and mutation probabilities must be finite and in [0, 1]")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("Weights must be three finite nonnegative numbers with positive sum")
    rng = np.random.default_rng(seed)
    pop = problem.repair(rng.random((size, problem.n)) < 0.35, rng)
    vals = problem.evaluate(pop)
    used = len(pop)
    archive = update_archive(None, pop, vals)
    history = [(used, len(archive[0]))]
    while used < budget:
        count = min(size, budget - used)
        pairs = rng.integers(0, len(pop), size=(count, 2))
        if weights is None:
            _, ranks, distances = selection_data(vals)
            a, b = pairs.T
            win = (ranks[a] < ranks[b]) | ((ranks[a] == ranks[b]) &
                                                        (distances[a] >= distances[b]))
        else:
            scores = (vals / problem.scales) @ weights
            a, b = pairs.T
            win = scores[a] <= scores[b]
        indices = np.where(win, a, b)
        child = problem.repair(breed(pop, indices, rng, crossover, mutation), rng)
        child_vals = problem.evaluate(child)
        used += count
        archive = update_archive(archive, child, child_vals)
        combined = np.vstack([pop, child])
        combined_vals = np.vstack([vals, child_vals])
        if weights is None:
            pop, vals = environmental_selection(combined, combined_vals, size)
        else:
            order = np.argsort((combined_vals / problem.scales) @ weights, kind="stable")[:size]
            pop, vals = combined[order].copy(), combined_vals[order].copy()
        history.append((used, len(archive[0])))
    assert used == budget and problem.feasible(archive[0]).all()
    return archive, history, used


def exact_front(problem):
    """Independent exhaustive reference for the small educational instance."""
    indices = np.arange(2 ** problem.n, dtype=np.uint32)
    pop = ((indices[:, None] >> np.arange(problem.n)) & 1).astype(bool)
    pop = pop[problem.feasible(pop)]
    if not len(pop):
        raise ValueError("No feasible nonempty portfolio exists")
    values = problem.evaluate(pop)
    archive = pareto_archive(pop, values)
    return archive, len(pop)


def metrics(front, reference, scales):
    front, reference, scales = np.asarray(front), np.asarray(reference), np.asarray(scales)
    if (front.ndim != 2 or reference.ndim != 2 or not len(front) or not len(reference)
            or front.shape[1] != reference.shape[1] or scales.shape != (front.shape[1],)
            or not np.isfinite(front).all() or not np.isfinite(reference).all()
            or not np.isfinite(scales).all() or np.any(scales <= 0)):
        raise ValueError("Metrics require nonempty finite fronts and positive finite scales")
    f, r = front / scales, reference / scales
    distances = np.sqrt(np.sum((r[:, None, :] - f[None, :, :]) ** 2, axis=2))
    igd = float(distances.min(axis=1).mean())
    epsilon = float(np.max(np.min(np.max(f[None, :, :] - r[:, None, :], axis=2), axis=1)))
    exact_set = {tuple(row) for row in reference}
    actual_set = {tuple(row) for row in front}
    hits = len(actual_set & exact_set)
    return {"igd": igd, "epsilon": epsilon, "exact_coverage": hits / len(exact_set),
            "precision": hits / len(actual_set), "front_size": len(actual_set)}

"""Reproducible experiments for multiobjective project portfolio selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from portfolio import Problem, exact_front, metrics, optimize, update_archive

ROOT = Path(__file__).resolve().parent


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generated_rows(config):
    rng = np.random.default_rng(config["data_seed"])
    rows = []
    for i in range(config["projects"]):
        cost = int(rng.integers(7, 26))
        rows.append({"project": i + 1, "effect": int(rng.integers(15, 70)),
                     "cost": cost, "risk": int(rng.integers(2, 28)),
                     "staff": int(rng.integers(3, 13))})
    return rows


def validate_config(config):
    for field in ("projects", "population", "runs", "evaluations", "seed_start", "data_seed"):
        value = config[field]
        minimum = 2 if field == "population" else 0 if field in ("seed_start", "data_seed") else 1
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    for field in ("crossover_probability", "mutation_probability"):
        value = config[field]
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{field} must be finite and in [0, 1]")
    weights = np.asarray(config["weights"], dtype=float)
    if (weights.ndim != 2 or weights.shape[1] != 3 or not len(weights)
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or np.any(weights.sum(axis=1) <= 0)):
        raise ValueError("weights must be a nonempty list of three finite nonnegative weights with positive sum")
    if config["evaluations"] < len(weights) * config["population"]:
        raise ValueError("budget must initialize every weighted GA")
    group_indices = {507: 0, 517: 1, 527: 2}
    if group_indices.get(config["group"]) != config["group_index"]:
        raise ValueError("group and group_index must agree with the assignment")
    if (not isinstance(config["student_number"], int) or config["student_number"] < 1
            or not isinstance(config["year_suffix"], int)):
        raise ValueError("student_number and year_suffix must be valid integers")
    assigned = 1 + ((17 * config["student_number"] + 7 * config["group_index"] + config["year_suffix"]) % 20)
    if config["variant"] != assigned or config["variant"] != 1 or config["track"] != "A":
        raise ValueError("This implementation covers assigned variant 1, track A")


def load_problem(config, data_path=None):
    path = Path(data_path) if data_path is not None else ROOT / "data" / "projects.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(path, generated_rows(config))
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != config["projects"]:
        raise ValueError("Project file and configuration differ")
    ids = [int(row["project"]) for row in rows]
    if sorted(ids) != list(range(1, len(rows) + 1)):
        raise ValueError("Project IDs must be unique integers 1..n")
    rows = [row for _, row in sorted(zip(ids, rows), key=lambda pair: pair[0])]
    columns = ("effect", "cost", "risk", "staff")
    data = np.array([[float(r[k]) for k in columns] for r in rows])
    generated = np.array([[float(r[k]) for k in columns] for r in generated_rows(config)])
    source_info = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "matches_configured_generator": bool(np.array_equal(data, generated)),
                   "configured_data_seed": config["data_seed"]}
    return Problem(data, config["cost_limit"], config["staff_limit"],
                   tuple(tuple(pair) for pair in config["dependencies"]), source_info)


def save_front(path, archive):
    pop, vals = archive
    write_csv(path, [{"projects": " ".join(str(j + 1) for j in np.flatnonzero(x)),
                      "effect": -f[0], "cost": f[1], "risk": f[2]}
                     for x, f in zip(pop, vals)])


def figures(output, archives, exact, records):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (a, b, la, lb) in zip(axes, [(1, 0, "Стоимость", "Эффект"),
                                         (2, 0, "Риск", "Эффект"),
                                         (1, 2, "Стоимость", "Риск")]):
        ax.scatter(exact[:, a], (-exact[:, b] if b == 0 else exact[:, b]),
                   s=12, color="0.8", label="Точный фронт")
        for label, (_, f) in archives.items():
            ax.scatter(f[:, a], -f[:, b] if b == 0 else f[:, b],
                       s=14, alpha=0.7, label=label)
        ax.set(xlabel=la, ylabel=lb)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "pareto.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    labels = ["NSGA-II", "Weighted GA"]
    for ax, key, title in zip(axes, ["igd", "exact_coverage"], ["IGD (меньше лучше)", "Доля точного фронта"]):
        groups = [[r[key] for r in records if r["algorithm"] == label] for label in labels]
        ax.boxplot(groups, tick_labels=labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "stability.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    texts = ["Инициализация\nи ремонт", "Оценка трёх\nкритериев", "Турнир: ранг\nи crowding", "Кроссовер\nи мутация", "Отбор из\nродителей и детей"]
    for i, text in enumerate(texts):
        ax.text(0.09 + i * 0.2, 0.6, text, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "square,pad=0.5", "facecolor": "white"})
        if i < 4:
            ax.annotate("", xy=(0.19 + i * 0.2, 0.6), xytext=(0.16 + i * 0.2, 0.6),
                        arrowprops={"arrowstyle": "->"})
    ax.text(0.5, 0.16, "Повтор до исчерпания бюджета; внешний архив сохраняет недоминируемые оценки.",
            ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "algorithm.png", dpi=180)
    plt.close(fig)


def report(output, config, problem, summary, exact, valid_count, archives):
    # Representative solutions are taken from one identified run, not pooled seeds.
    pop, vals = archives["NSGA-II"]
    normalized = vals / problem.scales
    candidates = [("Экономичный", int(np.argmin(vals[:, 1]))),
                  ("Сбалансированный", int(np.argmin(np.linalg.norm(normalized - normalized.min(axis=0), axis=1)))),
                  ("Максимальный эффект", int(np.argmin(vals[:, 0])))]
    selected, labels = [], []
    for label, index in candidates:
        if index not in selected:
            selected.append(index)
            labels.append(label)
    for index in range(len(vals)):
        if len(selected) >= min(3, len(vals)):
            break
        if index not in selected:
            selected.append(index)
            labels.append("Дополнительный компромисс")
    representatives = []
    for label, index in zip(labels, selected):
        representatives.append({"type": label, "projects": [int(j + 1) for j in np.flatnonzero(pop[index])],
                                "effect": float(-vals[index, 0]), "cost": float(vals[index, 1]),
                                "risk": float(vals[index, 2])})
    (output / "representatives.json").write_text(json.dumps(representatives, ensure_ascii=False, indent=2), encoding="utf-8")
    dependencies = "; ".join(f"{child + 1} требует {parent + 1}" for child, parent in problem.dependencies) or "отсутствуют"
    weights = ", ".join("(" + ",".join(f"{number:g}" for number in weight) + ")" for weight in config["weights"])
    if problem.source_info and problem.source_info["matches_configured_generator"]:
        provenance = (f"Таблица совпадает с синтетическими данными NumPy default_rng при seed = {config['data_seed']}. "
                      "Для каждого проекта последовательно генерируются целочисленные стоимость U[7,25], эффект U[15,69], риск U[2,27], трудоёмкость U[3,12].")
    else:
        provenance = "Использована переданная входная таблица; её происхождение не приписывается генератору из конфигурации."
    source_description = (f" Источник и SHA-256 таблицы сохранены в results/data_manifest.json." if problem.source_info else "")
    n_weights = len(config["weights"])
    lines = ["# Лабораторная работа № 3", "", "## 1. Постановка задачи", "",
             f"Вариант {config['variant']}, трек {config['track']}: многокритериальный подбор портфеля проектов. N = {config['student_number']}, группа {config['group']}, Q = {config['group_index']}, Y = {config['year_suffix']}. V = 1 + ((17·{config['student_number']} + 7·{config['group_index']} + {config['year_suffix']}) mod 20) = {config['variant']}. Правило распределения треков в задании отсутствует; выбран трек A.", "",
             f"Выбирается непустое подмножество из {problem.n} проектов. Критерии: максимум суммарного эффекта E, минимум стоимости C и ожидаемых потерь R. Стоимость, эффект и риск измеряются в условных единицах; трудоёмкость - в человеко-днях. Аддитивность и отсутствие синергии являются допущениями модели.", "",
             "## 2. Модель и данные", "",
             f"Генотип x ∈ {{0,1}}^{problem.n} совпадает с фенотипом выбранного множества. Минимизируется вектор F(x) = (-Σeᵢxᵢ, Σcᵢxᵢ, Σrᵢxᵢ). Ограничения: Σcᵢxᵢ ≤ {problem.cost_limit:g}; Σsᵢxᵢ ≤ {problem.staff_limit:g}; Σxᵢ ≥ 1. Зависимости в нумерации проектов с 1: {dependencies}.", "",
             provenance + source_description + " Проекты упорядочены по уникальным ID 1…n. Ремонт удаляет проекты с невыполненными зависимостями и случайные выбранные проекты при перегрузке; пустой портфель заменяется допустимым непустым набором. При ремонте проверяются ресурсы; полный вектор F не вычисляется.", "",
             "## 3. Алгоритмы и параметры", "",
             "Реализован NSGA-II: строгая недоминируемая сортировка, турнир по рангу и расстоянию до скопления, равномерный бинарный кроссовер, побитовая мутация и элитистский отбор из объединения родителей и потомков. Инициализация: каждый бит независимо равен единице с вероятностью 0.35, затем применяется ремонт. Турнир включает два кандидата с возвращением. Расстояние до скопления нормируется по каждому непостоянному критерию; граничные точки сохраняются преимущественно. Равномерный кроссовер независимо наследует каждый бинарный выбор; побитовая мутация меняет включение отдельного проекта, а ремонт восстанавливает зависимости.", "",
             f"Размер популяции {config['population']}; вероятность кроссовера {config['crossover_probability']}; вероятность мутации каждого бита {config['mutation_probability']}. Бюджет {config['evaluations']} оценок вектора F на запуск, включая начальную популяцию и повторные генотипы. Остановка строго по бюджету. Выполнено {config['runs']} независимых запусков с seed {config['seed_start']}–{config['seed_start'] + config['runs'] - 1} для каждого метода.", "",
             f"Базовый метод объединяет {n_weights} скалярных ГА со взвешенными суммами нормированных критериев и теми же операторами. Веса: {weights}. Общий бюджет делится между весами с разницей не более одной оценки; все начальные популяции входят в бюджет. Нормирование - на суммы коэффициентов критериев по всем проектам; для тождественно нулевого критерия знаменатель равен единице. У обоих методов сохраняется внешний архив всех недоминируемых оценённых решений; проверка доминирования не требует нового вычисления F. Новый механизм относительно работ № 1–2 - сортировка по Парето и поддержание разнообразия по crowding distance.", "",
             "![Схема NSGA-II](results/algorithm.png)", "",
             "## 4. Методика оценки", "",
             f"Полным перебором 2^{problem.n} = {2 ** problem.n} генотипов получено {valid_count} допустимых портфелей и {len(exact)} различных векторов точного фронта. Перебор служит независимым эталоном, не участвует в инициализации или поиске; его затраты исключены из равных поисковых бюджетов. Бюджет измеряет число полных векторов F; ресурсные проверки учитываются во времени. Для данного малого экземпляра точный метод практически применим; при росте числа проектов его стоимость экспоненциальна.", "",
             "IGD - среднее по точному фронту расстояние до ближайшей найденной точки в нормированных критериях. Аддитивный ε-индикатор равен max по эталонным точкам минимума по найденным точкам максимального покоординатного превышения; меньшее значение лучше. Покрытие - доля точных векторов, присутствующих в архиве; точность - доля архива, принадлежащая точному фронту. Недоминируемость внутри выборки сама по себе не доказывает принадлежность точному фронту.", "",
             "## 5. Результаты", "",
             "| Метод | Лучший IGD | Средний | Медиана | s | Худший | Среднее покрытие | Средний ε |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['igd_min']:.5f} | {s['igd_mean']:.5f} | {s['igd_median']:.5f} | {s['igd_std']:.5f} | {s['igd_max']:.5f} | {s['coverage_mean']:.3f} | {s['epsilon_mean']:.5f} |")
    sd_note = (" Для одиночного запуска разброс не оценивается; сохранённый ноль является условным обозначением."
               if config['runs'] == 1 else "")
    lines += ["", "s - выборочное стандартное отклонение." + sd_note + " Значения каждого запуска, время и число оценок приведены в results/runs.csv; статистики остальных показателей - в results/summary.json. Все сохранённые портфели проходят проверку допустимости.", "",
              "![Проекции фронта Парето](results/pareto.png)", "",
              "На проекциях показаны архивы первого запуска обоих методов, seed = " + str(config['seed_start']) + ". Проекция может скрывать различия по третьему критерию; доминирование проверяется в трёхмерном пространстве.", "",
              "![Устойчивость по независимым запускам](results/stability.png)", "",
              f"Характерные решения из архива NSGA-II первого запуска ({len(representatives)}):", "",
              "| Тип | Проекты | Эффект | Стоимость | Риск |", "|---|---|---:|---:|---:|"]
    for r in representatives:
        lines.append(f"| {r['type']} | {', '.join(map(str, r['projects']))} | {r['effect']:.10g} | {r['cost']:.10g} | {r['risk']:.10g} |")
    ns, ws = summary["NSGA-II"], summary["Weighted GA"]
    better = "меньше" if ns["igd_mean"] < ws["igd_mean"] else "больше либо равен"
    quality = (f"Для данного экземпляра результат NSGA-II является хорошим: среднее покрытие точного фронта составляет {ns['coverage_mean']:.1%} при IGD {ns['igd_mean']:.6g}. "
               "Оценка опирается на известное множество компромиссов, а не на один экстремальный портфель."
               if ns["coverage_mean"] >= 0.95 else
               f"Среднее покрытие {ns['coverage_mean']:.1%} показывает неполноту найденного множества; для восстановления точного фронта этот результат недостаточен.")
    lines += ["", "Экономичный портфель минимизирует затраты, максимальный эффект ориентирован на суммарный результат проектов. Сбалансированное решение минимизирует нормированное евклидово расстояние до идеальной точки внутри найденного архива; это явно заданное предпочтение, а не универсальный оптимум.", "",
              "## 6. Выводы", "",
              f"Средний IGD NSGA-II {better}, чем у объединения взвешенных ГА; средние покрытия составили {ns['coverage_mean']:.1%} и {ws['coverage_mean']:.1%}. Сравниваются совокупные конфигурации: NSGA-II использует один поиск, а взвешенный метод делит бюджет между {n_weights} поисками. Поэтому наблюдаемую разницу нельзя приписать только способу отбора. Веса задают направления поиска; Парето-ранжирование с расстоянием до скопления поддерживает разнообразие без предварительного задания предпочтений.", "",
              quality + " Результат является приближением полного множества компромиссов; принадлежность отдельных найденных векторов эталону проверяется перебором. Обобщение на реальные проекты ограничено аддитивной моделью, единственным экземпляром и фиксированным набором весов. Возможные улучшения: больше экземпляров, нелинейные взаимодействия и анализ чувствительности к ограничению бюджета.", "",
              "Источник постановки: Р. В. Душкин. «Эволюционное программирование и генетические алгоритмы», комплекс лабораторных работ, с. 12–15; исходный PDF в корне проекта."]
    if len(representatives) < 3:
        lines += ["", "В этой серии найдено менее трёх различных компромиссов; такое сравнение не заменяет полного эксперимента лабораторной работы."]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--data", type=Path, help="CSV with unique project IDs 1..n; defaults to data/projects.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        for key, val in [("runs", args.runs), ("seed_start", args.seed_start), ("evaluations", args.budget)]:
            if val is not None:
                config[key] = val
        validate_config(config)
        problem = load_problem(config, args.data)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_manifest.json").write_text(json.dumps(problem.source_info, indent=2, ensure_ascii=False), encoding="utf-8")
    reference, valid_count = exact_front(problem)
    save_front(out / "exact_front.csv", reference)
    records, traces, illustrative = [], [], {}
    for seed in range(config["seed_start"], config["seed_start"] + config["runs"]):
        for label in ("NSGA-II", "Weighted GA"):
            start = time.perf_counter()
            if label == "NSGA-II":
                archive, history, used = optimize(problem, seed, config["evaluations"], config["population"],
                                                  config["crossover_probability"], config["mutation_probability"])
                for evaluations, size in history:
                    traces.append({"algorithm": label, "seed": seed, "evaluations": evaluations, "archive_size": size})
            else:
                archive, used = None, 0
                quotient, remainder = divmod(config["evaluations"], len(config["weights"]))
                children = np.random.SeedSequence(seed).spawn(len(config["weights"]))
                for i, weight in enumerate(config["weights"]):
                    part, history, count = optimize(problem, children[i], quotient + (i < remainder),
                                                   config["population"], config["crossover_probability"],
                                                   config["mutation_probability"], np.array(weight))
                    for evaluations, size in history:
                        traces.append({"algorithm": f"Weighted GA w{i + 1}", "seed": seed,
                                       "evaluations": used + evaluations, "archive_size": size})
                    used += count
                    archive = update_archive(archive, *part)
            elapsed = time.perf_counter() - start
            row = {"algorithm": label, "seed": seed, "evaluations": used, "seconds": elapsed,
                   **metrics(archive[1], reference[1], problem.scales)}
            assert problem.feasible(archive[0]).all()
            records.append(row)
            save_front(out / f"front_{'nsga2' if label == 'NSGA-II' else 'weighted'}_{seed}.csv", archive)
            if seed == config["seed_start"]:
                illustrative[label] = archive
            print(f"{label} seed={seed} IGD={row['igd']:.5f} coverage={row['exact_coverage']:.3f}", flush=True)
    write_csv(out / "runs.csv", records)
    write_csv(out / "history.csv", traces)
    summary = {}
    for label in ("NSGA-II", "Weighted GA"):
        rows = [r for r in records if r["algorithm"] == label]
        stats = {}
        for field in ["igd", "epsilon", "exact_coverage", "precision", "front_size", "seconds"]:
            values = np.array([r[field] for r in rows])
            for suffix, value in [("min", values.min()), ("mean", values.mean()), ("median", np.median(values)),
                                  ("std", values.std(ddof=1) if len(values) > 1 else 0), ("max", values.max())]:
                stats[f"{field}_{suffix}"] = float(value)
        stats["coverage_mean"] = stats["exact_coverage_mean"]
        stats["runs"] = len(rows)
        summary[label] = stats
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "config_used.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "environment.json").write_text(json.dumps({"python": sys.version, "numpy": np.__version__,
                                                    "matplotlib": matplotlib.__version__, "platform": platform.platform()}, indent=2), encoding="utf-8")
    figures(out, illustrative, reference[1], records)
    if out == (ROOT / "results").resolve():
        report(out, config, problem, summary, reference[1], valid_count, illustrative)


if __name__ == "__main__":
    main()

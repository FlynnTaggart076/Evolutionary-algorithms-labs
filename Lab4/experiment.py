"""Reproduce all experiment tables, plots and the academic Markdown report."""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields, replace
import json
import hashlib
import os
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

from algorithm import Parameters, distance_matrix, mst_lower_bound, nearest_neighbor, optimize

HERE = Path(__file__).resolve().parent


def write_csv(path, rows, columns=None):
    rows = list(rows)
    if columns is None:
        columns = list(rows[0])
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def generated_points(config):
    n = config["cities"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 4:
        raise ValueError("cities must be an integer >=4")
    rng = np.random.default_rng(config["data_seed"])
    centers = np.array([[20., 20.], [20., 80.], [80., 20.], [80., 80.]])
    quotient, remainder = divmod(n, len(centers))
    return np.vstack([rng.normal(center, 8, size=(quotient + (i < remainder), 2))
                      for i, center in enumerate(centers)])


def load_data(config, path=None):
    path = Path(path) if path is not None else HERE / "data.csv"
    if not path.exists():
        points = generated_points(config)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(path, ({"city": i, "x": float(x), "y": float(y)}
                         for i, (x, y) in enumerate(points)))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["city", "x", "y"]:
            raise ValueError("Coordinate columns must be city,x,y")
        rows = list(reader)
    identifiers = [int(row["city"]) for row in rows]
    if identifiers != list(range(len(rows))):
        raise ValueError("City identifiers must be consecutive 0..n-1 in row order")
    points = np.array([[float(row["x"]), float(row["y"])] for row in rows])
    if points.shape != (config["cities"], 2) or not np.all(np.isfinite(points)):
        raise ValueError("data.csv does not match cities or contains nonfinite coordinates")
    return points


def data_description(config, path):
    points = load_data(config, path)
    if np.array_equal(points, generated_points(config)):
        counts = [config["cities"] // 4 + (i < config["cities"] % 4) for i in range(4)]
        return (f"Данные синтетические: NumPy default_rng, seed {config['data_seed']}; "
                f"четыре центра (20,20), (20,80), (80,20), (80,80); числа точек {counts}. "
                f"Координаты независимы, нормальное распределение со стандартным отклонением 8. "
                f"Таблица: {Path(path).name}. Происхождение проверено повторной генерацией.")
    return (f"Использована входная таблица {Path(path).name}, {len(points)} пунктов. "
            "Координаты отличаются от генератора при указанном seed; "
            "синтетическое происхождение для этой таблицы не утверждается.")


def diagram(path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.set(xlim=(0, 10), ylim=(0, 7))
    ax.axis("off")
    boxes = [
        (2, 5.8, 6, .7, "Данные и параметры; фиксация seed"),
        (2, 4.7, 6, .7, "Случайная популяция; оценка маршрутов"),
        (2, 3.6, 6, .7, "Турнир; OX; инверсия; оценка потомка"),
        (2, 2.5, 6, .7, "Меметический вариант: вероятностный 2-opt"),
        (2, 1.4, 6, .7, "Элитизм; сохранение лучшего; учёт бюджета"),
        (2, .3, 6, .7, "Бюджет исчерпан: маршрут и статистика"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="black", linewidth=.8))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    for k in range(5):
        ax.add_patch(FancyArrowPatch((5, boxes[k][1]), (5, boxes[k + 1][1] + .7),
                                    arrowstyle="->", mutation_scale=10, color="black"))
    ax.plot([8, 9, 9, 8], [1.75, 1.75, 3.95, 3.95], color="black", linewidth=.8)
    ax.annotate("", xy=(8, 3.95), xytext=(9, 3.95), arrowprops={"arrowstyle": "->", "lw": .8})
    ax.text(9.15, 2.85, "Есть бюджет", rotation=90, fontsize=9, va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_plots(results, points, reference, output, config):
    labels = {"ga": "Базовый ГА", "memetic": f"2-opt, p={config['local_probability']}",
              "memetic_sparse": f"2-opt, p={config['sparse_local_probability']}"}
    colors = {"ga": "#555555", "memetic": "#125a85", "memetic_sparse": "#8a4b12"}
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    for method in labels:
        group = [r for r in results if r["algorithm"] == method]
        steps = np.array([x for x, _ in group[0]["trace"]])
        trajectories = np.array([[y for _, y in r["trace"]] for r in group])
        ax.plot(steps, trajectories.mean(axis=0), label=labels[method], color=colors[method])
        ax.fill_between(steps, trajectories.min(axis=0), trajectories.max(axis=0),
                        color=colors[method], alpha=.14)
    ax.axhline(reference["nearest_neighbor"], color="black", linestyle="--", linewidth=1,
               label="Ближайший сосед: ориентир")
    ax.set(xlabel="Число оценок маршрута", ylabel="Лучшая длина маршрута")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "convergence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    data = [[r["best"] for r in results if r["algorithm"] == m] for m in labels]
    ax.boxplot(data, tick_labels=list(labels.values()), showmeans=True)
    ax.axhline(reference["nearest_neighbor"], color="black", linestyle="--", linewidth=.9)
    ax.set(ylabel="Итоговая длина маршрута")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    best = min(results, key=lambda r: r["best"])
    method = best["algorithm"]
    tour = best["tour"] + best["tour"][:1]
    ax.plot(points[tour, 0], points[tour, 1], "-", color=colors[method], linewidth=1)
    ax.scatter(points[:, 0], points[:, 1], s=15, color="black")
    start = best["tour"][0]
    ax.scatter(*points[start], s=65, marker="s", facecolors="none", edgecolors="black")
    ax.annotate(f"Старт: {start}", points[start], xytext=(8, 8), textcoords="offset points", fontsize=9)
    ax.set(title=f"Лучший маршрут: L = {best['best']:.2f}", xlabel="x, усл. ед.", ylabel="y, усл. ед.")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output / "routes.png", dpi=180)
    plt.close(fig)
    diagram(output / "algorithm.png")


def make_report(config, summary, reference, results, output, data_path=None, report_path=None):
    def f(x):
        return f"{x:.3f}"
    ga, memetic, sparse = summary
    improvement = 100 * (ga["mean"] - memetic["mean"]) / ga["mean"] if ga["mean"] else 0.
    wins = sum(m["best"] < g["best"] for g, m in zip(results[::3], results[1::3]))
    sparse_improvement = 100 * (ga["mean"] - sparse["mean"]) / ga["mean"] if ga["mean"] else 0.
    sparse_wins = sum(m["best"] < g["best"] for g, m in zip(results[::3], results[2::3]))
    best = min(results, key=lambda r: r["best"])
    relative_lb = 100 * (best["best"] - reference["mst_lower_bound"]) / reference["mst_lower_bound"] if reference["mst_lower_bound"] else 0.
    relative_nn = 100 * (best["best"] - reference["nearest_neighbor"]) / reference["nearest_neighbor"] if reference["nearest_neighbor"] else 0.
    direction = "уменьшил" if improvement >= 0 else "увеличил"
    sd_direction = "выросло" if sparse["std"] > ga["std"] else "снизилось" if sparse["std"] < ga["std"] else "не изменилось"
    n = config["cities"]
    destination = Path(report_path or HERE / "report.md").resolve()
    relative_output = Path(os.path.relpath(Path(output).resolve(), destination.parent)).as_posix()
    provenance = data_description(config, data_path or HERE / "data.csv")
    quality = ("Лучший маршрут допустим и не длиннее конструктивного ориентира."
               if best["best"] <= reference["nearest_neighbor"] else
               "Лучший маршрут допустим, но уступает конструктивному ориентиру; качество этой настройки недостаточно.")
    coordinates = load_data(config, data_path or HERE / "data.csv")
    optimality = ("Нулевая длина при совпадающих координатах достигает неотрицательной нижней границы; оптимальность доказана."
                  if best["best"] == 0 and np.all(coordinates == coordinates[0]) else
                  "Глобальная оптимальность не доказана.")
    report = f"""# Лабораторная работа № 4

## Меметический алгоритм решения задачи маршрутизации

### 1. Постановка задачи

Индивидуальные параметры: N = 11, группа 517, Q = 1, Y = 26. Вариант V = 1 + ((17·11 + 7·1 + 26) mod 20) = 1. Тема 1: генетический алгоритм с локальным поиском 2-opt. Предварительная постановка зафиксирована в PROPOSAL.md.

Требуется построить замкнутый маршрут обслуживания {n} пунктов. Генотип π является перестановкой множества {{0, …, {n-1}}}; фенотип - последовательность посещения пунктов с возвратом в начало. Минимизируется L(π) = Σ d(π[k], π[(k+1) mod {n}]), k = 0, …, {n-1}, где d - евклидово расстояние. Ограничение: каждый пункт посещается ровно один раз. Единицы координат и расстояний условные. Вместимость, дорожная сеть и временные окна не моделируются.

{provenance}

### 2. Алгоритмы и эксперимент

Начальная популяция состоит из независимых равномерных случайных перестановок. Базовый ГА использует турнирную селекцию, OX-кроссовер, инверсию и элитизм. OX сохраняет порядок части родительского маршрута; инверсия меняет последовательность посещения сегмента. Оба оператора сохраняют перестановку, поэтому ремонт ограничений не требуется.

Меметический вариант дополнительно применяет к потомку ограниченный 2-opt: перебирает в случайном порядке до {config['local_checks']} пар несмежных рёбер и принимает каждое строгое улучшение. Полный локальный оптимум не гарантируется. После каждого принятия дальнейшие пары применяются к обновлённому маршруту.

| Параметр | Значение |
| --- | --- |
| Популяция; элита; размер турнира | {config['population']}; {config['elites']}; {config['tournament_size']} |
| Вероятность OX; вероятность инверсии | {config['crossover_probability']}; {config['mutation_probability']} |
| Вероятности 2-opt; предел проверок | {config['local_probability']} и {config['sparse_local_probability']}; {config['local_checks']} |
| Бюджет каждого запуска | {config['budget']} оценок маршрута |
| Независимые запуски каждого метода | {config['runs']} |
| Seed алгоритма | {config['seed_start']} … {config['seed_start'] + config['runs'] - 1} |

Остановка происходит точно при исчерпании бюджета. Одна оценка - полный расчёт длины одного маршрута. Учтены начальная популяция, потомки и все проверенные соседи 2-opt; копирование элиты не требует повторной оценки. При неполном последнем поколении сохраняется лучший из всех оценённых маршрутов. Число поколений методов различается. Бюджет и остальные параметры совпадают; различается наличие либо частота 2-opt. Одинаковые номера seed используются попарно; после включения локального поиска последовательности случайных обращений расходятся.

Исследуются частоты {config['local_probability']} и {config['sparse_local_probability']}. Редкая конфигурация добавлена разведочно после первоначального отрицательного результата. Три пробных запуска при seed {config['seed_start']} (частоты 0,005; 0,01; 0,02) записаны в results/pilot.csv и учтены отдельно от основных сравнений. Дополнительное сравнение не представляет независимого подтверждения выбранной настройки.

![Схема алгоритма]({relative_output}/algorithm.png)

### 3. Результаты

Полные результаты содержатся в {relative_output}/runs.csv; траектории - в {relative_output}/traces.csv. В таблице приведено выборочное стандартное отклонение с делителем n-1. Все полученные маршруты допустимы.

| Метод | Лучшее | Среднее | Медиана | Ст. откл. | Худшее |
| --- | --- | --- | --- | --- | --- |
| Базовый ГА | {f(ga['best'])} | {f(ga['mean'])} | {f(ga['median'])} | {f(ga['std'])} | {f(ga['worst'])} |
| ГА + 2-opt | {f(memetic['best'])} | {f(memetic['mean'])} | {f(memetic['median'])} | {f(memetic['std'])} | {f(memetic['worst'])} |
| ГА + редкий 2-opt | {f(sparse['best'])} | {f(sparse['mean'])} | {f(sparse['median'])} | {f(sparse['std'])} | {f(sparse['worst'])} |

Среднее время запуска: базовый ГА {f(ga['mean_seconds'])} с; ГА + 2-opt {f(memetic['mean_seconds'])} с. Время является характеристикой данного запуска и окружения. Среднее число оценок, использованных локальным поиском: {f(memetic['mean_local_evaluations'])}; эти оценки входят в общий лимит {config['budget']}.

Для редкого 2-opt среднее время составляет {f(sparse['mean_seconds'])} с, среднее число локальных оценок {f(sparse['mean_local_evaluations'])}. Все три конфигурации имеют одинаковый общий бюджет.

![Динамика лучшей длины по числу оценок]({relative_output}/convergence.png)

Линии обозначают средние лучшие найденные значения, области - минимум и максимум по независимым запускам. Они отражают разброс, а не доверительный интервал. Итоговое распределение показано ниже.

![Распределение итоговой длины]({relative_output}/distribution.png)

Частый локальный поиск 2-opt {direction} среднюю длину на {abs(improvement):.2f}%. В {wins} из {config['runs']} пар seed этот вариант получил более короткий маршрут. Это описательное сравнение на одном экземпляре, а не доказательство превосходства на всех задачах маршрутизации.

Редкий 2-opt изменил среднюю длину относительно базового ГА на {-sparse_improvement:+.2f}% и выиграл в {sparse_wins} из {config['runs']} пар seed. Сопоставление двух частот показывает цену локального улучшения при фиксированном числе полных оценок.

### 4. Предметная интерпретация и оценка качества

![Лучший замкнутый маршрут]({relative_output}/routes.png)

Лучший маршрут получен конфигурацией {best['algorithm']} при seed {best['seed']}, его длина {best['best']:.3f}. Порядок посещения и длины участков приведены в {relative_output}/best_route.csv; последнее ребро возвращает маршрут в начало.

Конструктивный ориентир - лучший маршрут ближайшего соседа по всем {n} стартовым пунктам: {reference['nearest_neighbor']:.3f}. Он построен отдельно и не включён в сравнение равных бюджетов. Минимальное остовное дерево даёт строгую нижнюю границу {reference['mst_lower_bound']:.3f}: удаление любого ребра гамильтонова цикла оставляет остовное дерево. Отклонение лучшего результата от этой границы составляет {relative_lb:.2f}%; изменение относительно ближайшего соседа составляет {relative_nn:+.2f}%. Разрыв с нижней границей не равен фактической ошибке относительно неизвестного оптимума.

### 5. Выводы и ограничения

Частый 2-opt {direction} среднюю длину на {abs(improvement):.2f}%. При частоте {config['sparse_local_probability']} изменение составило {-sparse_improvement:+.2f}%, стандартное отклонение {sd_direction}: {f(ga['std'])} и {f(sparse['std'])}. При равном бюджете локальный поиск сокращает число поколений, поэтому его интенсивность необходимо согласовывать с затратами эволюционного поиска. Разведочно выбранная редкая конфигурация требует независимой проверки.

{quality} {optimality} Выводы ограничены одним входным экземпляром. Дальнейшая работа: несколько размеров и наборов координат, настройка локального бюджета, контроль разнообразия и ограничения реальной доставки. Сценарий демонстрации на 6 минут приведён в DEMO.md.

Источник задания: Душкин Р. В. «Эволюционное программирование и генетические алгоритмы. Комплекс лабораторных работ», раздел лабораторной работы № 4, с. 18-20.
"""
    destination.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--data", type=Path, default=HERE / "data.csv")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("runs", "budget", "seed_start"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if config["runs"] < 1:
        parser.error("--runs must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    points = load_data(config, args.data)
    p = Parameters(**{field.name: config[field.name] for field in fields(Parameters)})
    results = []
    pilot = []
    for probability in (0.005, 0.01, 0.02):
        trial = optimize(points, config['seed_start'], True, replace(p, local_probability=probability))
        pilot.append({"seed": config['seed_start'], "local_probability": probability,
                      "best": trial["best"], "evaluations": trial["evaluations"]})
    write_csv(output / "pilot.csv", pilot)
    for seed in range(config["seed_start"], config["seed_start"] + config["runs"]):
        for method, memetic, parameters in (("ga", False, p), ("memetic", True, p),
                ("memetic_sparse", True, replace(p, local_probability=config['sparse_local_probability']))):
            result = optimize(points, seed, memetic, parameters)
            result["algorithm"] = method
            results.append(result)
            print(f"{result['algorithm']:7s} seed={seed} length={result['best']:.3f} evals={result['evaluations']}", flush=True)
    columns = ["algorithm", "seed", "best", "evaluations", "local_evaluations", "generations", "seconds"]
    write_csv(output / "runs.csv", ({key: r[key] for key in columns} for r in results))
    write_csv(output / "traces.csv", ({"algorithm": r["algorithm"], "seed": r["seed"],
                                     "evaluations": e, "best": score}
                                    for r in results for e, score in r["trace"]))
    summary = []
    for method in ("ga", "memetic", "memetic_sparse"):
        group = [r for r in results if r["algorithm"] == method]
        values = np.array([r["best"] for r in group])
        summary.append({"algorithm": method, "runs": len(group), "best": float(values.min()),
                        "mean": float(values.mean()), "median": float(np.median(values)),
                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.,
                        "worst": float(values.max()),
                        "mean_seconds": float(np.mean([r["seconds"] for r in group])),
                        "mean_local_evaluations": float(np.mean([r["local_evaluations"] for r in group]))})
    _, nn_value = nearest_neighbor(points)
    reference = {"nearest_neighbor": nn_value, "mst_lower_bound": mst_lower_bound(points)}
    write_csv(output / "summary.csv", summary)
    (output / "summary.json").write_text(json.dumps({"summary": summary, "reference": reference},
                                                    indent=2, ensure_ascii=False), encoding="utf-8")
    snapshot = {**config, "data_file": str(args.data.resolve()),
                "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest()}
    (output / "experiment_config.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "environment.json").write_text(json.dumps({"python": platform.python_version(),
                                                        "numpy": np.__version__,
                                                        "matplotlib": matplotlib.__version__}, indent=2), encoding="utf-8")
    best = min(results, key=lambda r: r["best"])
    (output / "best_by_method.json").write_text(json.dumps({method: {
        key: value for key, value in min((r for r in results if r['algorithm'] == method),
                                        key=lambda r: r['best']).items() if key != 'trace'}
        for method in ('ga', 'memetic', 'memetic_sparse')}, indent=2), encoding='utf-8')
    (output / "best_solution.json").write_text(json.dumps({key: value for key, value in best.items()
                                                          if key != "trace"}, indent=2), encoding="utf-8")
    d = distance_matrix(points)
    route = best["tour"]
    write_csv(output / "best_route.csv", ({"order": k, "city": city,
                                           "x": float(points[city, 0]), "y": float(points[city, 1]),
                                           "next_city": route[(k + 1) % len(route)],
                                           "edge_length": float(d[city, route[(k + 1) % len(route)]])}
                                          for k, city in enumerate(route)))
    make_plots(results, points, reference, output, config)
    if output == (HERE / "results").resolve():
        make_report(config, summary, reference, results, output, data_path=args.data)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

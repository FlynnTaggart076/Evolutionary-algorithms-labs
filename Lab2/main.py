"""Reproduce the data, four 20-run comparisons, and the laboratory report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
from statistics import mean, median, stdev
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

try:
    from .core import Problem, exact_dynamic_programming, positive_integer, solve, validate_search_parameters
except ImportError:
    from core import Problem, exact_dynamic_programming, positive_integer, solve, validate_search_parameters


BASE = Path(__file__).resolve().parent
LABELS = {
    "repair_uniform": "ГА: ремонт, равномерный",
    "penalty_uniform": "ГА: штраф, равномерный",
    "repair_two_point": "ГА: ремонт, двухточечный",
    "random_repair": "Случайная выборка с ремонтом",
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_data(config: dict, path: Path) -> None:
    rng = np.random.default_rng(config["data_seed"])
    n = config["objects"]
    mass = rng.integers(3, 23, n)
    volume = rng.integers(2, 19, n)
    value = rng.integers(12, 111, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, [
        {"item": f"I{i+1:02d}", "mass": int(mass[i]),
         "volume": int(volume[i]), "value": int(value[i])}
        for i in range(n)
    ])


def validate_config(config: dict) -> None:
    for key in ("mass_limit", "volume_limit"):
        positive_integer(config[key], key)
    positive_integer(config["objects"], "objects", minimum=30)
    positive_integer(config["data_seed"], "data_seed", minimum=0)
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a nonempty list.")
    for seed in seeds:
        positive_integer(seed, "seed", minimum=0)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique for independent runs.")
    methods = config["methods"]
    if not isinstance(methods, list) or not 1 <= len(methods) <= 4:
        raise ValueError("methods must contain between one and four configurations.")
    for method in methods:
        validate_search_parameters(config, method)
    if len({method["name"] for method in methods}) != len(methods):
        raise ValueError("Method names must be unique.")
    variant = config["variant"]
    positive_integer(variant["N"], "N")
    positive_integer(variant["Y"], "Y", minimum=0)
    if variant["V"] != 1 or 1 + ((17*variant["N"] + 7*variant["Q"] + variant["Y"]) % 20) != 1:
        raise ValueError("This implementation supports the assigned variant V=1.")
    if {507: 0, 517: 1, 527: 2}.get(variant["group"]) != variant["Q"]:
        raise ValueError("The group and Q parameters disagree.")


def data_provenance(config: dict, problem: Problem, path: Path, generated: bool) -> dict:
    rng = np.random.default_rng(config["data_seed"])
    expected = [rng.integers(lo, hi, problem.n) for lo, hi in ((3, 23), (2, 19), (12, 111))]
    matches = all(np.array_equal(actual, expected_column) for actual, expected_column in
                  zip((problem.mass, problem.volume, problem.value), expected))
    matches = matches and problem.names == tuple(f"I{i+1:02d}" for i in range(problem.n))
    return {"source_path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "objects": problem.n, "generated_during_run": generated,
            "matches_config_generator": matches, "data_seed": config["data_seed"]}


def make_examples(problem: Problem, reference: dict) -> list[dict]:
    empty = problem.describe(np.zeros(problem.n, dtype=bool))
    examples = [{"name": "empty_feasible", **empty}, {"name": "optimal_feasible", **reference}]
    if examples[0]["bits"] == examples[1]["bits"]:
        for index in range(problem.n):
            candidate = np.zeros(problem.n, dtype=bool)
            candidate[index] = True
            described = problem.describe(candidate)
            if described["feasible"]:
                examples[1] = {"name": "singleton_feasible", **described}
                break
        else:
            raise ValueError("Input data must admit at least two distinct feasible binary examples.")
    rng = np.random.default_rng(21991)
    wanted = {"mass_only": None, "volume_only": None}
    invalid = []
    for _ in range(100000):
        candidate = rng.random(problem.n) < 0.4
        mass = int(candidate @ problem.mass)
        volume = int(candidate @ problem.volume)
        if (mass > problem.mass_limit or volume > problem.volume_limit) and len(invalid) < 2:
            described = problem.describe(candidate)
            if not invalid or described["bits"] != invalid[0]["bits"]:
                invalid.append(described)
        if mass > problem.mass_limit and volume <= problem.volume_limit:
            wanted["mass_only"] = problem.describe(candidate)
        if volume > problem.volume_limit and mass <= problem.mass_limit:
            wanted["volume_only"] = problem.describe(candidate)
        if all(value is not None for value in wanted.values()):
            break
    if any(value is None for value in wanted.values()):
        # Separate violations need not exist when one resource constraint implies
        # the other. The assignment requires two infeasible examples, not isolated violations.
        if len(invalid) < 2:
            for omitted in range(-1, problem.n):
                candidate = np.ones(problem.n, dtype=bool)
                if omitted >= 0:
                    candidate[omitted] = False
                described = problem.describe(candidate)
                if not described["feasible"] and all(described["bits"] != row["bits"] for row in invalid):
                    invalid.append(described)
                if len(invalid) >= 2:
                    break
        if len(invalid) < 2:
            raise ValueError("Input data must admit at least two distinct infeasible binary examples.")
        return examples + [{"name": f"infeasible_{index+1}", **row} for index, row in enumerate(invalid[:2])]
    return examples + [{"name": name, **value} for name, value in wanted.items()]


def summarize(runs: list[dict], reference_value: int) -> list[dict]:
    summaries = []
    for method in dict.fromkeys(row["method"] for row in runs):
        selected = [row for row in runs if row["method"] == method]
        values = [row["best_feasible_value"] for row in selected]
        summaries.append({
            "method": method, "runs": len(values), "best": max(values),
            "mean": mean(values), "median": median(values),
            "sample_sd": stdev(values) if len(values) > 1 else 0.0,
            "worst": min(values),
            "mean_gap_percent": mean((reference_value - value) / reference_value * 100 for value in values) if reference_value else 0.0,
            "optimum_fraction": sum(value == reference_value for value in values) / len(values),
            "mean_feasible_fraction": mean(row["feasible_fraction"] for row in selected),
            "selected_feasible_fraction": mean(row["selected_feasible"] for row in selected),
            "mean_seconds": mean(row["seconds"] for row in selected),
            "evaluations_min": min(row["evaluations"] for row in selected),
            "evaluations_max": max(row["evaluations"] for row in selected),
        })
    return summaries


def method_labels(config: dict) -> dict[str, str]:
    labels = {}
    for method in config["methods"]:
        constraint = "ремонт" if method["constraint"] == "repair" else "штраф"
        if method["algorithm"] == "random":
            labels[method["name"]] = "Случайная выборка с ремонтом" if constraint == "ремонт" else "Случайная выборка со штрафом"
        else:
            operator = "равномерный" if method["crossover"] == "uniform" else "двухточечный"
            labels[method["name"]] = f"ГА: {constraint}, {operator}"
    return labels


def plot_results(results: list[dict], reference: dict, folder: Path, labels: dict[str, str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True, sharey=True)
    if len(labels) > 4:
        raise ValueError("At most four comparison methods are supported.")
    for ax in axes.flat[len(labels):]:
        ax.set_visible(False)
    for ax, method in zip(axes.flat, labels):
        runs = [result for result in results if result["method"] == method]
        if not runs:
            ax.set_visible(False)
            continue
        trajectories = np.array([[value for _, value in run["history"]] for run in runs])
        evaluations = [calls for calls, _ in runs[0]["history"]]
        ax.fill_between(evaluations, trajectories.min(axis=0), trajectories.max(axis=0), alpha=0.2)
        ax.plot(evaluations, trajectories.mean(axis=0), linewidth=1.6, label="Среднее; полоса min–max")
        ax.axhline(reference["value"], color="black", linestyle="--", linewidth=1, label="Точный оптимум")
        ax.set_title(labels[method], fontsize=10)
        ax.grid(alpha=0.22)
        ax.set_xlabel("Вычисления фитнес-функции")
        ax.set_ylabel("Лучшая допустимая ценность")
    axes.flat[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(folder / "convergence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.7))
    methods = [method for method in labels if any(result["method"] == method for result in results)]
    values = [[result["best_feasible_value"] for result in results if result["method"] == method] for method in methods]
    ax.boxplot(values, tick_labels=[labels[method].replace(": ", ":\n") for method in methods], showmeans=True)
    ax.axhline(reference["value"], color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Лучшая допустимая ценность")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(folder / "distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    labels = [
        "Инициализация бинарной популяции; seed",
        "Ремонт либо сохранение исходного генотипа",
        "Оценка ценности, массы и объёма; допустимый архив",
        "Турнир → кроссовер → битовая мутация",
        "Обработка ограничений и оценка потомков",
        "Элитистский отбор (μ + λ); проверка бюджета",
        "Лучший допустимый набор и статистика серии",
    ]
    for index, label in enumerate(labels):
        y = 9.2 - index * 1.4
        ax.add_patch(FancyBboxPatch((0.7, y - 0.48), 8.6, 0.85, boxstyle="round,pad=0.06", facecolor="white", edgecolor="black"))
        ax.text(5, y - 0.05, label, ha="center", va="center", fontsize=10)
        if index < len(labels) - 1:
            ax.annotate("", xy=(5, y - 0.98), xytext=(5, y - 0.5), arrowprops={"arrowstyle":"->"})
    ax.plot([0.65, 0.25, 0.25], [2.15, 2.15, 5.0], color="black", linewidth=1)
    ax.annotate("", xy=(0.65, 5.0), xytext=(0.25, 5.0), arrowprops={"arrowstyle":"->"})
    ax.text(0.05, 3.55, "бюджет не исчерпан", rotation=90, fontsize=8,
            ha="center", va="center")
    fig.tight_layout()
    fig.savefig(folder / "algorithm.png", dpi=180)
    plt.close(fig)


def build_report(config: dict, summaries: list[dict], reference: dict, examples: list[dict],
                 best: dict, output: Path, report_path: Path, problem: Problem,
                 provenance: dict) -> None:
    labels = method_labels(config)
    table = "\n".join(
        f"| {labels[row['method']]} | {row['best']} | {row['mean']:.2f} | {row['median']:.1f} | {row['sample_sd']:.2f} | {row['worst']} |"
        for row in summaries
    )
    extra_table = "\n".join(
        f"| {labels[row['method']]} | {100*row['mean_feasible_fraction']:.1f} | {100*row['selected_feasible_fraction']:.0f} | {100*row['optimum_fraction']:.0f} | {row['mean_gap_percent']:.2f} | {row['mean_seconds']:.3f} |"
        for row in summaries
    )
    example_table = "\n".join(
        f"| {row['name']} | {row['mass']} | {row['volume']} | {row['value']} | {row['mass_excess']} | {row['volume_excess']} | {'да' if row['feasible'] else 'нет'} |"
        for row in examples
    )
    by_name = {row["method"]: row for row in summaries}
    contrast, analysis = [], []
    ga_methods = [method for method in config["methods"] if method["algorithm"] == "ga"]
    for index, first in enumerate(ga_methods):
        for second in ga_methods[index + 1:]:
            difference = [key for key in ("constraint", "crossover") if first[key] != second[key]]
            if len(difference) == 1:
                factor = "способ обработки ограничений" if difference[0] == "constraint" else "оператор кроссовера"
                contrast.append(f"Сравнение «{labels[first['name']]}» и «{labels[second['name']]}» меняет только {factor}.")
                a, b = by_name[first["name"]]["mean"], by_name[second["name"]]["mean"]
                analysis.append(f"Для «{labels[first['name']]}» и «{labels[second['name']]}» средние равны {a:.2f} и {b:.2f}; разность второго и первого составляет {b-a:+.2f}.")
    baseline_text, baseline_analysis = [], []
    for method in config["methods"]:
        if method["algorithm"] == "random":
            processing = "применяет ремонт" if method["constraint"] == "repair" else "применяет штраф"
            baseline_text.append(f"«{labels[method['name']]}» независимо генерирует строки с вероятностью единицы 0,5, {processing} и сохраняет допустимый архив; её начальный пакет содержит пустой набор. Отбор родителей отсутствует, бюджет равен {config['budget']} оценкам.")
            for ga in ga_methods:
                if ga["constraint"] == method["constraint"]:
                    a, b = by_name[ga['name']]['mean'], by_name[method['name']]['mean']
                    baseline_analysis.append(f"Среднее «{labels[method['name']]}» составляет {b:.2f}; преимущество «{labels[ga['name']]}» по средней допустимой ценности равно {a-b:+.2f}.")
                    break
    relative = Path(__import__("os").path.relpath(output, report_path.parent)).as_posix()
    figure = lambda filename: f"{relative}/figures/{filename}"
    solution = best["solution"]
    mass_limit, volume_limit = problem.mass_limit, problem.volume_limit
    data_path = Path(__import__("os").path.relpath(provenance["source_path"], report_path.parent)).as_posix()
    if provenance["matches_config_generator"]:
        data_description = f"Данные соответствуют генератору NumPy PCG64 при seed = {config['data_seed']}: последовательно получены {problem.n} масс из целых [3,22], {problem.n} объёмов из [2,18] и {problem.n} ценностей из [12,110] равномерным выбором."
    else:
        data_description = "Использована предоставленная входная таблица; значения массы, объёма и ценности непосредственно прочитаны из CSV. Соответствие генератору из конфигурации не установлено."
    data_description += f" Таблица: `{data_path}`. Контрольная сумма и сведения о загрузке сохранены в `{relative}/input_data.json`."
    seed_text = ", ".join(map(str, config["seeds"]))
    gap = 100*(reference['value']-solution['value'])/reference['value'] if reference['value'] else 0.0
    variant = config["variant"]
    text = f'''# Лабораторная работа № 2

## 1. Постановка задачи

Дисциплина: «Эволюционное программирование и генетические алгоритмы». Индивидуальные параметры: N = {variant['N']}, группа {variant['group']}, Q = {variant['Q']}, Y = {variant['Y']}. Вариант V = 1 + ((17·{variant['N']} + 7·{variant['Q']} + {variant['Y']}) mod 20) = {variant['V']}. Рассматривается рюкзак с ограничениями массы и объёма.

Имеются {problem.n} неделимых предметов. Генотип x принадлежит {{0,1}}^{problem.n}; x_i = 1 означает включение предмета i. Фенотипом является выбранное множество предметов, дополнительное декодирование не требуется. Цель: максимизировать F(x) = Σ c_i x_i при ограничениях Σ m_i x_i ≤ {mass_limit} и Σ v_i x_i ≤ {volume_limit}. Масса, объём и ценность заданы в условных целочисленных единицах. Все ограничения жёсткие; штраф применяется только как механизм поиска.

{data_description} Экземпляр служит для сравнения методов, а не для моделирования конкретного склада.

## 2. Алгоритм и параметры

Использован собственный генетический алгоритм (μ + λ). Размеры родительской популяции и поколения потомков равны {config['population']}. Инициализация: {config['population']-1} независимых бинарных строк с вероятностью единицы 0,5 и пустой допустимый набор. Турнир состоит из {config['tournament']} кандидатов с выборкой с возвращением. Вероятность кроссовера равна {config['crossover_probability']}; вероятность инверсии каждого бита равна {config['mutation_probability']}. После оценки потомков сохраняются {config['population']} лучших особей из родителей и потомков; значения фитнеса родителей повторно не вычисляются. Отдельный архив сохраняет лучший допустимый набор.

Реализованы два способа обработки ограничений. Ремонт удаляет включённые предметы по возрастанию c_i/(m_i/{mass_limit} + v_i/{volume_limit}), пока оба ограничения не выполнены. Допустимые строки остаются неизменными. Удаление не нарушает бинарность, а пустое множество всегда допустимо, поэтому процедура завершается. Штрафной способ сохраняет генотип и максимизирует S(x) = F(x) - {config['penalty']:g}·[max(0,M(x)-{mass_limit}) + max(0,V(x)-{volume_limit})]. Величина S не подменяет исходную ценность F; качество всех методов измеряется по допустимому архиву.

Равномерный кроссовер выбирает каждый бит из одного из двух родителей с вероятностью 0,5. Двухточечный кроссовер переносит непрерывный сегмент второго родителя в копию первого. Оба оператора сохраняют алфавит {{0,1}}, но могут нарушать ресурсные ограничения; после них применяется выбранный способ обработки ограничений. Битовая мутация соответствует неделимому выбору предметов. Порядок строк не вводит дополнительных ограничений модели; равномерный кроссовер наследует выбор каждого предмета независимо от соседних индексов.

![Рисунок 1. Схема генетического алгоритма. Левый переход возвращает поиск к созданию потомков.]({figure("algorithm.png")})

## 3. Методика эксперимента

Для каждой из {len(config['methods'])} конфигураций проведено {len(config['seeds'])} независимых запусков с seed: {seed_text}. Внутри серии seed различаются; между методами используются одинаковые номера seed для сопоставления. Бюджет каждого запуска равен {config['budget']} вычислениям фитнес-функции: {config['population']} исходных особей и {config['budget']//config['population']-1} поколений по {config['population']} потомков. Ремонт использует исходную таблицу ценностей и текущие суммы ресурсов; дополнительных оценок полной целевой функции при ремонте нет. Каждая подготовленная строка оценивается один раз, включая повторяющиеся строки. Лимит относится к поиску; отдельный точный эталон в него не включён.

{' '.join(contrast)} {' '.join(baseline_text)}

Точный эталон получен собственной динамической программой по двум целочисленным ресурсам: D_i(a,b) = max(D_(i-1)(a,b), c_i + D_(i-1)(a-m_i,b-v_i)) для допустимых индексов; начальное D_0(a,b) = 0. Сложность O(n·{mass_limit}·{volume_limit}). Обратный проход восстанавливает набор. Получено F* = {reference['value']}; масса {reference['mass']}, объём {reference['volume']}. Рекуррентный перебор альтернатив «взять/не взять» даёт точную гарантию для предоставленного экземпляра и не является основным оптимизатором.

## 4. Проверка допустимости

| Пример | Масса | Объём | Ценность | Превышение массы | Превышение объёма | Допустим |
|---|---:|---:|---:|---:|---:|---|
{example_table}

Полные бинарные векторы и идентификаторы предметов приведены в `{relative}/examples.json`. Первые два примера допустимы; третий и четвёртый нарушают ограничения, как показано в таблице. При взаимозависимых ресурсных ограничениях изолированное нарушение каждого ресурса может быть невозможно. После поиска ценность и оба ресурса каждого архивного решения повторно вычислены из исходной таблицы независимо от поискового счётчика.

## 5. Результаты

В таблице приведена лучшая допустимая ценность каждого запуска. Стандартное отклонение выборочное, с делителем n - 1.

| Метод | Лучшее | Среднее | Медиана | Станд. откл. | Худшее |
|---|---:|---:|---:|---:|---:|
{table}

| Метод | Допустимые оценки, % | Допустимый лидер, % | Достижение F*, % | Средний разрыв, % | Время, с |
|---|---:|---:|---:|---:|---:|
{extra_table}

Доля допустимых оценок рассчитана по всем {config['budget']} обработанным кандидатам. Допустимый лидер означает допустимость особи с максимальным поисковым score в конечной популяции; допустимый архив при этом существует во всех запусках. Разрыв равен 100·(F* - F)/F*. Время усреднено по серии и включает поиск с ремонтом, но исключает генерацию данных, точный эталон, построение графиков и повторную проверку архива. Число оценок во всех запусках проверено программно.

![Рисунок 2. Сходимость лучшей допустимой ценности: среднее и диапазон min–max по серии.]({figure("convergence.png")})

![Рисунок 3. Распределения итоговой допустимой ценности; пунктиром показан точный оптимум.]({figure("distribution.png")})

Лучший найденный допустимый набор получен методом `{best['method']}`, seed = {best['seed']}: {', '.join(solution['items'])}. Ценность равна {solution['value']}, масса {solution['mass']} ≤ {mass_limit}, объём {solution['volume']} ≤ {volume_limit}. Полная предметная таблица выбранного набора сохранена в `{relative}/best_set.csv`, вектор и проверенные суммы - в `{relative}/best_solution.json`.

## 6. Анализ и вывод

{' '.join(analysis)} Это наблюдения на одной серии и одном экземпляре, а не доказательство универсального превосходства оператора.

{' '.join(baseline_analysis)} Точный эталон позволяет оценивать качество по измеримому разрыву, а не по субъективной близости результатов. Лучший результат имеет разрыв {gap:.2f}%; распределения и частота достижения F* характеризуют воспроизводимость качества. При нулевом F* и нулевой допустимой ценности относительный разрыв принимается равным нулю.

Ограничения исследования: один экземпляр; фиксированные коэффициент штрафа и вероятности операторов; условные единицы ресурсов; восстановление допустимости может смещать поиск к предметам высокой плотности. При увеличении задачи двухресурсная динамическая программа становится дорогой. Дальнейшее исследование требует нескольких экземпляров, анализа коэффициента штрафа и альтернативных процедур ремонта. Строгая гарантия относится к DP-эталону; результаты ГА являются экспериментальными.

## 7. Воспроизводимость

Реализация проверена с Python {platform.python_version()} и NumPy {np.__version__}. Фактические параметры находятся в `{relative}/config_snapshot.json`; результаты отдельных запусков - в `{relative}/runs.csv`, траектории - в `{relative}/convergence.csv`, сводка - в `{relative}/summary.csv`. Для повторения используются эта конфигурация и входная таблица `{data_path}` через параметры --config и --data. Инструкция установки и команды запуска приведены в README.

Источник постановки: Душкин Р. В. «Эволюционное программирование и генетические алгоритмы. Комплекс лабораторных работ», лабораторная работа № 2, вариант 1.
'''
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE / "config.json")
    parser.add_argument("--output", type=Path, default=BASE / "results")
    parser.add_argument("--data", type=Path, default=BASE / "data" / "items.csv")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        validate_config(config)
        generated = not args.data.exists()
        if generated:
            generate_data(config, args.data)
        problem = Problem.load(args.data, config["mass_limit"], config["volume_limit"])
        if problem.n != config["objects"]:
            raise ValueError("Data/config object counts disagree.")
        provenance = data_provenance(config, problem, args.data, generated)
        reference = exact_dynamic_programming(problem)
        examples = make_examples(problem, reference)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config_snapshot.json", config)
    write_json(output / "input_data.json", provenance)
    write_json(output / "exact_reference.json", reference)
    write_json(output / "examples.json", examples)
    results = []
    total_started = perf_counter()
    for method in config["methods"]:
        for seed in config["seeds"]:
            result = solve(problem, config, method, seed)
            if result["evaluations"] != config["budget"]:
                raise AssertionError("Unequal search budget.")
            results.append(result)
        print(f"{method['name']}: {len(config['seeds'])} runs completed", flush=True)
    runs = [{key: value for key, value in result.items() if key not in {"solution", "history"}} for result in results]
    write_csv(output / "runs.csv", runs)
    summaries = summarize(runs, reference["value"])
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "convergence.csv", [
        {"method": result["method"], "seed": result["seed"], "evaluations": calls, "best_feasible_value": value}
        for result in results for calls, value in result["history"]
    ])
    best = max(results, key=lambda result: result["best_feasible_value"])
    write_json(output / "best_solution.json", {"method": best["method"], "seed": best["seed"], **best["solution"]})
    write_csv(output / "best_set.csv", [
        {"item": name, "mass": int(problem.mass[index]), "volume": int(problem.volume[index]), "value": int(problem.value[index])}
        for index, name in enumerate(problem.names) if best["solution"]["bits"][index]
    ], fieldnames=["item", "mass", "volume", "value"])
    plot_results(results, reference, output / "figures", method_labels(config))
    report_path = BASE / "report.md" if output == (BASE / "results").resolve() else output / "report.md"
    build_report(config, summaries, reference, examples, best, output, report_path, problem, provenance)
    write_json(output / "verification.json", {
        "runs": len(runs), "expected_runs": len(config["seeds"]) * len(config["methods"]),
        "equal_budget": all(row["evaluations"] == config["budget"] for row in runs),
        "all_archives_feasible": all(result["solution"]["feasible"] for result in results),
        "all_values_bounded_by_exact_optimum": all(row["best_feasible_value"] <= reference["value"] for row in runs),
        "exact_optimum": reference["value"], "total_seconds": perf_counter() - total_started,
    })
    print(f"Verified {len(runs)} runs; exact optimum = {reference['value']}; report: {report_path}", flush=True)


if __name__ == "__main__":
    main()

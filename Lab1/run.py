"""Запуск воспроизводимого эксперимента лабораторной работы № 1."""
import argparse
import csv
from dataclasses import asdict, replace
import json
from numbers import Integral, Real
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from ga import GAConfig, genetic_search, random_search
from objectives import get_problem

HERE = Path(__file__).resolve().parent


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagram(path, elite=2):
    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.axis("off")
    labels = ["Инициализация допустимой популяции", "Вычисление f(x), учёт бюджета", f"Сохранение элитных особей: {elite}", "Турнирный отбор, BLX-α-кроссовер", "Гауссовская мутация, отражение границ", "Оценка потомков, замена популяции", "Бюджет исчерпан? Возврат лучшей особи"]
    ys = np.linspace(9.1, 0.9, len(labels))
    for y, label in zip(ys, labels):
        ax.add_patch(FancyBboxPatch((1.0,y-0.38),8,0.76,boxstyle="round,pad=0.08",facecolor="#f4f4f4",edgecolor="black",linewidth=0.8))
        ax.text(5,y,label,ha="center",va="center",fontsize=10)
    for a,b in zip(ys[:-1],ys[1:]):
        ax.annotate("",(5,b+0.48),(5,a-0.48),arrowprops={"arrowstyle":"->","lw":1})
    ax.plot([9.13,9.6,9.6,9.1],[ys[-1],ys[-1],ys[2],ys[2]],color="black",lw=1)
    ax.annotate("",(9.1,ys[2]),(9.6,ys[2]),arrowprops={"arrowstyle":"->","lw":1})
    ax.text(9.7,5,"Нет",rotation=90,va="center",fontsize=9)
    fig.tight_layout()
    fig.savefig(path,dpi=180,bbox_inches="tight")
    plt.close(fig)


def make_plots(results, methods, out, elite=2):
    fig, ax = plt.subplots(figsize=(7.4,4.4))
    for method in methods:
        series = [r[2].history for r in results if r[0] == method]
        values = np.array([[h["best"] for h in s] for s in series])
        generations = np.array([h["generation"] for h in series[0]])
        mean = values.mean(axis=0)
        line, = ax.plot(generations,np.maximum(mean,1e-300),label=method,linewidth=1.4)
        ax.fill_between(generations,np.maximum(values.min(axis=0),1e-300),np.maximum(values.max(axis=0),1e-300),alpha=0.13,color=line.get_color())
    ax.set(xlabel="Поколение (для случайного поиска: контрольный шаг)",ylabel="Лучшее найденное f(x)",yscale="log")
    ax.grid(True,alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out/"convergence.png",dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4,4.1))
    arrays = [[r[2].value for r in results if r[0]==m] for m in methods]
    ax.boxplot(arrays,tick_labels=methods,showmeans=True)
    ax.set(ylabel="Итоговое f(x)",yscale="log")
    ax.grid(True,axis="y",alpha=0.25)
    fig.tight_layout()
    fig.savefig(out/"distribution.png",dpi=180)
    plt.close(fig)
    diagram(out/"algorithm.png", elite)


def number(value):
    return f"{value:.6g}"


def ratio_text(numerator, denominator):
    return f"{numerator / denominator:.3g}" if denominator != 0 else "не определено при нулевом знаменателе"


def validate_experiment_config(config):
    for name, minimum in (("runs", 1), ("seed_start", 0), ("budget", 1),
                          ("student_number", 1), ("Q", 0), ("Y", 0), ("variant", 1)):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if config["variant"] != 1:
        raise ValueError("This experiment and report support assigned variant 1 only; other functions are available in objectives.py")
    if {"507": 0, "517": 1, "527": 2}.get(str(config["group"])) != config["Q"]:
        raise ValueError("group and Q disagree with the assignment")
    if config["Y"] > 99:
        raise ValueError("Y must contain the final two digits of the year")
    if 1 + (17*config["student_number"] + 7*config["Q"] + config["Y"]) % 20 != config["variant"]:
        raise ValueError("Student parameters do not calculate assigned variant 1")
    tolerance = config["success_tolerance"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("success_tolerance must be finite and nonnegative")
    ga1 = GAConfig(**config["ga"])
    ga2 = replace(ga1, mutation_scale=config["comparison_mutation_scale"])
    ga1.validate()
    ga2.validate()
    if config["budget"] < ga1.population:
        raise ValueError("Budget must cover initialization")
    if ga1.mutation_scale == ga2.mutation_scale:
        raise ValueError("comparison_mutation_scale must differ from ga.mutation_scale")
    return ga1, ga2


def make_report(problem, config, summaries, results, path):
    by_method = {row["method"]:row for row in summaries}
    baseline, first, second = [by_method[m] for m in ["random", "GA sigma=0.05", "GA sigma=0.15"]] if config["ga"]["mutation_scale"]==0.05 and config["comparison_mutation_scale"]==0.15 else [summaries[2],summaries[0],summaries[1]]
    winner = min(summaries[:2],key=lambda row:row["mean"])
    best = min(results,key=lambda r:r[2].value)
    success_text = lambda row: "не определена без численного ориентира" if row["success_rate"] is None else f"{row['success_rate']:.0%}"
    seeds = f"{config['seed_start']}–{config['seed_start']+config['runs']-1}"
    rows = "\n".join("| " + " | ".join([s["method"],number(s["best"]),number(s["mean"]),number(s["median"]),number(s["std"]),number(s["worst"])]) + " |" for s in summaries)
    quality = "" if problem.optimum is None else f"Теоретический ориентир: f* = {number(problem.optimum)}. "
    calculated_variant = 1 + (17 * config['student_number'] + 7 * config['Q'] + config['Y']) % 20
    vline = f"N = {config['student_number']}, группа {config['group']}, Q = {config['Q']}, Y = {config['Y']}. V = 1 + ((17·{config['student_number']} + 7·{config['Q']} + {config['Y']}) mod 20) = {calculated_variant}."
    if problem.variant != calculated_variant:
        vline += f" Явно выбран вариант V = {problem.variant}."
    ga = config['ga']
    if problem.variant == 1:
        successes = lambda row: int(round(row['success_rate'] * row['runs']))
        if winner['success_rate'] == 1:
            assessment = f"Результат {winner['method']} является хорошим для данной задачи: точность {config['success_tolerance']:g} достигнута во всех {winner['runs']} запусках, а значения сопоставлены с доказанным минимумом."
        elif winner['success_rate'] > 0:
            assessment = f"{winner['method']} приближает доказанный минимум, однако достижение точности {config['success_tolerance']:g} неустойчиво."
        else:
            assessment = f"Результат недостаточен относительно заданной точности {config['success_tolerance']:g}: она не достигнута ни в одном запуске."
        assessment += f" Для {first['method']} точность достигнута в {successes(first)} из {first['runs']} запусков; для {second['method']} - в {successes(second)} из {second['runs']} запусков. Сфера является простой унимодальной функцией, поэтому эксперимент не доказывает эффективности ГА на мультимодальных задачах. Для сферы градиентные или аналитические методы экономичнее."
    else:
        assessment = "Качество оценено относительно случайного поиска при одинаковом бюджете и теоретического ориентира, если он известен. Результаты являются эмпирическими; глобальная оптимальность найденных векторов не доказана."
    report = f"""# Лабораторная работа № 1

## Многомерная оптимизация генетическим алгоритмом

### 1. Постановка задачи

{vline} Размерность d = 5 + (V mod 6) = {problem.dimension}. Требуется минимизировать функцию «{problem.name}» на области [{problem.lower:g}; {problem.upper:g}]^{problem.dimension}.

`{problem.formula}`.

{quality}{'Каждое слагаемое неотрицательно, а при x = (0, 0, 0, 0, 0, 0) сумма равна нулю; следовательно, глобальный минимум равен нулю.' if problem.variant == 1 else ''}

### 2. Представление и алгоритм

Генотип и фенотип совпадают: особь представляет собой вещественный вектор длины d. Начальная популяция равномерно распределена в допустимой области. Селекция проводится турниром из {ga['tournament']} особей. BLX-α-кроссовер выбирает каждый ген равномерно на интервале [min(a,b) - α|a-b|; max(a,b) + α|a-b|], α = {ga['blx_alpha']}. Его вероятность равна {ga['crossover_probability']}. При отсутствии кроссовера копируется первый родитель.

Каждый ген мутирует независимо с вероятностью 1/d. Добавляется нормальная случайная величина с нулевым средним и стандартным отклонением σ(g)(u-l), где σ(g) = σ₀[0.0001 + 0.9999(1-E/B)²], E - уже использованный бюджет, B - полный бюджет. Убывание масштаба задано заранее и одинаково в обеих конфигурациях. Выход за границы устраняется повторным отражением. Число переносимых элитных особей: {ga['elite']}; их оценки повторно не вычисляются. Остановка наступает при точном исчерпании бюджета. Последний пакет потомков при необходимости неполный; оставшиеся позиции заполняются ранее оценёнными неэлитными родителями без новых вычислений функции. Размер популяции сохраняется равным {ga['population']}.

![Блок-схема генетического алгоритма](results/algorithm.png)

Турнир не требует масштабирования функции. BLX-α выбирает значения между родителями, а при α > 0 также допускает выход за родительский интервал. Мутация восстанавливает разнообразие, отражение сохраняет допустимость, а элитизм гарантирует невозрастание лучшего значения.

### 3. Методика эксперимента

| Параметр | Значение |
| --- | --- |
| Число независимых запусков каждого метода | {config['runs']} |
| Начальные seed | {seeds} |
| Вычислений функции на запуск | {config['budget']} |
| Размер популяции / число элит | {config['ga']['population']} / {config['ga']['elite']} |
| Размер турнира / вероятность кроссовера | {ga['tournament']} / {ga['crossover_probability']} |
| Параметр BLX-α | {ga['blx_alpha']} |
| Вероятность мутации одного гена | 1/d = {1/problem.dimension:.6g} |
| Сравниваемый фактор | Начальный масштаб σ₀: {config['ga']['mutation_scale']} и {config['comparison_mutation_scale']} |
| Условие практического успеха | f(x) - f* ≤ {config['success_tolerance']:g} |

Сравниваются две конфигурации ГА, различающиеся только σ₀, и равномерная случайная выборка. Во всех методах учитывается одно вычисление на каждый оценённый вектор, включая начальную популяцию. Используются парные seed; первые {ga['population']} точек одинаковы у всех методов. Это сохраняет независимость запусков с различными seed и уменьшает влияние различий старта. Нулевой вектор специально не добавляется. Для случайного поиска контрольные шаги совпадают по числу оценок с поколениями ГА. Стандартное отклонение рассчитано с делителем n - 1; для единственного запуска в машиночитаемых файлах сохраняется 0 как условное значение, не оценка разброса. Время измеряется отдельно и зависит от среды выполнения.

### 4. Результаты

| Метод | Лучшее | Среднее | Медиана | Ст. откл. | Худшее |
| --- | --- | --- | --- | --- | --- |
{rows}

Наименьшее среднее среди ГА получено для {winner['method']}: {number(winner['mean'])}. Отношение среднего результата случайного поиска к этому значению равно {ratio_text(baseline['mean'],winner['mean'])}. Во всех запусках израсходовано ровно {config['budget']} оценок. Все итоговые решения допустимы.

Доля достижения заданной точности: {first['method']} - {success_text(first)}; {second['method']} - {success_text(second)}; случайный поиск - {success_text(baseline)}. Среднее время запуска составило соответственно {first['mean_seconds']:.4f}, {second['mean_seconds']:.4f} и {baseline['mean_seconds']:.4f} с.

![Сходимость: среднее, минимум и максимум по серии](results/convergence.png)

Линии показывают среднее лучшее значение, полупрозрачная область - минимум и максимум по серии. Логарифмическая шкала позволяет сопоставить порядок ошибки. Для случайного поиска показан лучший результат накопленной выборки.

![Распределение итоговых значений](results/distribution.png)

Лучшее решение получено методом {best[0]}, seed = {best[1]}, f(x) = {best[2].value:.10g}. Его координаты:

`[{', '.join(f'{x:.9g}' for x in best[2].x)}]`.

Все координаты с полной точностью, индивидуальные результаты и траектории представлены в results/solutions.json, results/runs.csv и results/histories.csv.

### 5. Анализ и вывод

{problem.difficulty} На выбранной функции первый начальный масштаб мутации {'улучшил' if first['mean'] < second['mean'] else 'не улучшил'} среднее итоговое значение по сравнению со вторым масштабом. Отношение средних результатов второго ГА к первому равно {ratio_text(second['mean'],first['mean'])}. Это наблюдение относится к указанному бюджету, представлению и серии seed; оно не устанавливает универсальное преимущество параметра.

{assessment} Дополнительное требование выполнено реализацией BLX-α-кроссовера. Дальнейшее исследование может включать чувствительность к размеру популяции и сравнение с методами, использующими производные.

### Источник

Душкин Р. В. Эволюционное программирование и генетические алгоритмы: комплекс лабораторных работ. Общие требования, лабораторная работа № 1, приложение с формулами. С. 1–8, 25–31.
"""
    path.write_text(report,encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=HERE/"config.json")
    parser.add_argument("--variant",type=int,choices=[1],help="Assigned experiment variant (1); other objectives are library helpers")
    parser.add_argument("--runs",type=int)
    parser.add_argument("--seed-start",type=int)
    parser.add_argument("--budget",type=int)
    parser.add_argument("--output",type=Path,default=HERE/"results")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        for field in ["variant","runs","seed_start","budget"]:
            value = getattr(args,field)
            if value is not None:
                config[field] = value
        ga1, ga2 = validate_experiment_config(config)
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))
    problem = get_problem(config["variant"])
    methods = [f"GA sigma={ga1.mutation_scale}",f"GA sigma={ga2.mutation_scale}","random"]
    if len(set(methods)) != len(methods):
        parser.error("Experiment method labels must be unique")
    results, runs, histories, solutions, summaries = [], [], [], [], []
    args.output.mkdir(parents=True,exist_ok=True)
    for seed in range(config["seed_start"],config["seed_start"]+config["runs"]):
        one = genetic_search(problem,seed,config["budget"],ga1)
        two = genetic_search(problem,seed,config["budget"],ga2)
        random = random_search(problem,seed,config["budget"],[h["evaluations"] for h in one.history])
        for method,result in zip(methods,[one,two,random]):
            results.append((method,seed,result))
            runs.append({"method":method,"seed":seed,"best":result.value,"evaluations":result.evaluations,"seconds":result.seconds,"feasible":bool(np.all((result.x>=problem.lower)&(result.x<=problem.upper)))})
            histories.extend({"method":method,"seed":seed,**row} for row in result.history)
            solutions.append({"method":method,"seed":seed,"value":result.value,"x":result.x.tolist()})
    for method in methods:
        subset = [r for r in runs if r["method"]==method]
        values = np.array([r["best"] for r in subset])
        summaries.append({"method":method,"runs":len(values),"best":float(values.min()),"mean":float(values.mean()),"median":float(np.median(values)),"std":float(values.std(ddof=1)) if len(values)>1 else 0.0,"worst":float(values.max()),"success_rate":float(np.mean(values-problem.optimum<=config["success_tolerance"])) if problem.optimum is not None else None,"mean_seconds":float(np.mean([r["seconds"] for r in subset])),"evaluations":config["budget"],"feasible_rate":float(np.mean([r["feasible"] for r in subset]))})
    write_csv(args.output/"runs.csv",runs)
    write_csv(args.output/"histories.csv",histories)
    write_csv(args.output/"summary.csv",summaries)
    for name,value in [("solutions.json",solutions),("summary.json",summaries),("experiment.json",{"parameters":config,"ga_configurations":[asdict(ga1),asdict(ga2)],"python":platform.python_version(),"numpy":np.__version__,"problem":{"name":problem.name,"dimension":problem.dimension,"bounds":[problem.lower,problem.upper],"optimum":problem.optimum}})]:
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    make_plots(results,methods,args.output,ga1.elite)
    # The report references the canonical results directory to keep relative links stable.
    if args.output.resolve() == (HERE/"results").resolve():
        make_report(problem,config,summaries,results,HERE/"report.md")
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()

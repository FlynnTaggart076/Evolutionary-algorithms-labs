"""Векторизованные целевые функции из приложения к заданию."""
from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class Problem:
    variant: int
    name: str
    lower: float
    upper: float
    formula: str
    optimum: float | None
    difficulty: str

    @property
    def dimension(self) -> int:
        return 5 + self.variant % 6

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """One output for every row; an individual is always a real vector."""
        x = np.asarray(points, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError(f"Expected an (n, {self.dimension}) matrix")
        d, v = self.dimension, self.variant
        i = np.arange(1, d + 1)
        squares = np.sum(x * x, axis=1)
        if v == 1:
            return squares
        if v == 2:
            return np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1) ** 2, axis=1)
        if v == 3:
            return 10 * d + np.sum(x * x - 10 * np.cos(2 * np.pi * x), axis=1)
        if v == 4:
            return -20 * np.exp(-0.2 * np.sqrt(squares / d)) - np.exp(np.mean(np.cos(2 * np.pi * x), axis=1)) + 20 + np.e
        if v == 5:
            return 1 + squares / 4000 - np.prod(np.cos(x / np.sqrt(i)), axis=1)
        if v == 6:
            return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)
        if v == 7:
            w = 1 + (x - 1) / 4
            return np.sin(np.pi * w[:, 0]) ** 2 + np.sum((w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2), axis=1) + (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)
        if v == 8:
            s = np.sum(0.5 * i * x, axis=1)
            return squares + s * s + s ** 4
        if v == 9:
            return (x[:, 0] - 1) ** 2 + np.sum(i[1:] * (2 * x[:, 1:] ** 2 - x[:, :-1]) ** 2, axis=1)
        if v == 10:
            return -np.sum(np.sin(x) * np.sin(i * x * x / np.pi) ** 20, axis=1)
        if v == 11:
            return np.sum(np.abs(x * np.sin(x) + 0.1 * x), axis=1)
        if v == 12:
            return np.sum((x * x - i) ** 2, axis=1)
        if v == 13:
            radius = np.sqrt(squares)
            return 1 - np.cos(2 * np.pi * radius) + 0.1 * radius
        if v == 14:
            return 0.5 * np.sum(x ** 4 - 16 * x * x + 5 * x, axis=1)
        if v == 15:
            return np.sum(i * x * x, axis=1)
        if v == 16:
            return x[:, 0] ** 2 + 1e6 * np.sum(x[:, 1:] ** 2, axis=1)
        if v == 17:
            return np.sum(np.abs(x) ** (2 + 4 * (i - 1) / (d - 1)), axis=1)
        if v == 18:
            return 1e6 * x[:, 0] ** 2 + np.sum(x[:, 1:] ** 2, axis=1)
        if v == 19:
            return np.abs(squares - d) ** 0.25 + (0.5 * squares + np.sum(x, axis=1)) / d + 0.5
        return np.abs(squares ** 2 - np.sum(x, axis=1) ** 2) ** 0.5 + (0.5 * squares + np.sum(x, axis=1)) / d + 0.5


def get_problem(variant: int) -> Problem:
    if isinstance(variant, bool) or not isinstance(variant, Integral) or not 1 <= variant <= 20:
        raise ValueError("Variant must be in 1..20")
    names = ["Сфера", "Розенброк", "Растригин", "Экли", "Гриванк", "Швефель", "Леви", "Захаров", "Диксон-Прайс", "Михалевич", "Альпин № 1", "Цин", "Саломон", "Стыблинский-Танг", "Сумма квадратов", "Изогнутая сигара", "Различные степени", "Диск", "Счастливая кошка", "HGBat"]
    bounds = [(-100,100),(-5,10),(-5.12,5.12),(-32.768,32.768),(-600,600),(-500,500),(-10,10),(-5,10),(-10,10),(0,np.pi),(-10,10),(-500,500),(-100,100),(-5,5),(-10,10),(-100,100),(-1,1),(-100,100),(-20,20),(-15,15)]
    formulas = [
        "f(x) = sum(x_i^2)",
        "f(x) = sum_{i=1}^{d-1} [100(x_{i+1}-x_i^2)^2 + (x_i-1)^2]",
        "f(x) = 10d + sum[x_i^2 - 10 cos(2 pi x_i)]",
        "f(x) = -20 exp(-0.2 sqrt(sum(x_i^2)/d)) - exp(sum cos(2 pi x_i)/d) + 20 + e",
        "f(x) = 1 + sum(x_i^2)/4000 - product cos(x_i/sqrt(i))",
        "f(x) = 418.9829d - sum[x_i sin(sqrt(abs(x_i)))]",
        "w_i=1+(x_i-1)/4; f(x)=sin^2(pi w_1)+sum_{i=1}^{d-1}(w_i-1)^2[1+10sin^2(pi w_i+1)]+(w_d-1)^2[1+sin^2(2pi w_d)]",
        "f(x) = sum(x_i^2) + [sum(0.5 i x_i)]^2 + [sum(0.5 i x_i)]^4",
        "f(x) = (x_1-1)^2 + sum_{i=2}^d i(2x_i^2-x_{i-1})^2",
        "f(x) = -sum[sin(x_i) sin(i x_i^2/pi)^20]",
        "f(x) = sum abs(x_i sin(x_i)+0.1x_i)",
        "f(x) = sum(x_i^2-i)^2",
        "f(x) = 1-cos(2pi sqrt(sum x_i^2))+0.1sqrt(sum x_i^2)",
        "f(x) = 0.5 sum(x_i^4-16x_i^2+5x_i)",
        "f(x) = sum(i x_i^2)",
        "f(x) = x_1^2 + 10^6 sum_{i=2}^d x_i^2",
        "f(x) = sum abs(x_i)^(2+4(i-1)/(d-1))",
        "f(x) = 10^6 x_1^2 + sum_{i=2}^d x_i^2",
        "s=sum x_i^2; f(x)=abs(s-d)^(1/4)+(0.5s+sum x_i)/d+0.5",
        "s=sum x_i^2; f(x)=sqrt(abs(s^2-(sum x_i)^2))+(0.5s+sum x_i)/d+0.5",
    ]
    difficulties = [
        "Функция выпукла и не затрудняет градиентный поиск. Трудность случайного поиска связана с малой относительной мерой окрестности минимума при росте размерности.",
        "Узкая искривлённая долина и сильная связь соседних координат замедляют продвижение к минимуму; независимые большие мутации часто выводят из долины.",
        "Периодические слагаемые создают множество локальных минимумов. Локальный поиск зависит от начальной точки, а случайная выборка редко попадает в малую область глобального минимума.",
        "Периодические локальные минимумы сочетаются с узкой областью глобального минимума; это осложняет локальный и случайный поиск.",
        "Произведение косинусов создаёт множество локальных минимумов и взаимодействие координат на широкой области поиска.",
        "Большое число удалённых локальных минимумов и расположение глобального минимума вдали от центра области осложняют поиск.",
        "Периодические слагаемые порождают локальные минимумы. Функция сепарабельна; масштаб мутации влияет на переход между впадинами каждой координаты.",
        "Функция унимодальна, но связь координат и слагаемое четвёртой степени образуют плохо обусловленный ландшафт.",
        "Нелинейная связь соседних координат создаёт узкую область малых значений и осложняет независимую настройку координат.",
        "Степень 20 создаёт узкие впадины с многочисленными локальными экстремумами. Малая мера хороших областей затрудняет случайный поиск.",
        "Модуль и синус образуют негладкий мультимодальный ландшафт с множественными локальными минимумами.",
        "В каждой координате имеются два минимума; широкая область поиска и четвёртая степень снижают эффективность случайной выборки.",
        "Радиальные периодические впадины создают локальные минимумы, а глобальный минимум находится в центре малой области.",
        "В каждой координате имеются два локальных минимума; возможна преждевременная фиксация в неверной впадине.",
        "Функция выпукла; различающиеся веса координат влияют на точность. Случайный поиск не использует направленность улучшений.",
        "Отношение кривизн равно 10^6; одинаковый масштаб мутации не соответствует резко различающейся чувствительности координат.",
        "Разные степени приводят к различной чувствительности координат около нуля и затрудняют выбор общего масштаба мутации.",
        "Одна координата имеет кривизну в 10^6 раз выше остальных; изотропная мутация плохо согласована с геометрией функции.",
        "Негладкое слагаемое четвёртого корня формирует узкую область малых значений около сферы; минимум требует согласованных изменений координат.",
        "Негладкий корень и связь сумм координат создают узкую долину, требующую согласованного изменения генов.",
    ]
    d = 5 + variant % 6
    optimum = 0.0
    if variant == 6:
        # Precision follows the rounded constant in the assignment.
        z = 420.9687462275036
        optimum = float(d * (418.9829 - z * np.sin(np.sqrt(z))))
    elif variant == 10:
        optimum = None
    elif variant == 14:
        roots = np.roots([4,0,-32,5])
        z = min(roots, key=lambda t: t**4 - 16*t*t + 5*t)
        optimum = float(0.5*d*(z**4-16*z*z+5*z))
    return Problem(variant, names[variant-1], *bounds[variant-1], formulas[variant-1], optimum, difficulties[variant-1])

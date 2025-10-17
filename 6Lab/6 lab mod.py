from __future__ import annotations
import math
import random
from typing import List, Tuple

K_DEFAULT = 5
P_DEFAULT = [0.15, 0.20, 0.25, 0.25, 0.15]  # сумма = 1.00 (без промаха)
N_DEFAULT = 1_000_000  # число прогонов Монте-Карло (можно уменьшить для быстрого запуска)
SEED_DEFAULT = 42      # фиксируем генератор

def validate_probs(p: List[float]) -> Tuple[List[float], float]:
    """Проверка вероятностей: допускаем, что sum(p) <= 1. Остаток трактуем как 'промах'.
    Возвращаем (p, p_miss)."""
    if len(p) < 3:
        raise ValueError("Требуется k > 2: укажите как минимум 3 бака (k>=3).")
    if any(x < 0 for x in p):
        raise ValueError("Вероятности не могут быть отрицательными.")
    s = sum(p)
    if s > 1 + 1e-12:
        raise ValueError(f"Сумма вероятностей по бакам = {s:.6f} > 1. Уменьшите значения.")
    p_miss = max(0.0, 1.0 - s)
    return p, p_miss

def analytic_probability(p: List[float]) -> float:
    """Точная вероятность воспламенения при двух выстрелах.
    Событие = (оба в один бак) ИЛИ (в соседние баки). Крайние баки не 'замыкаются'.
    Промах (если есть) не влияет, как отдельная категория без соседей.
    Формула: sum_i p_i^2 + 2 * sum_{i=1..k-1} p_i * p_{i+1}.
    """
    # игнорируем промах, он не добавляет шансов
    same = sum(pi * pi for pi in p)
    adjacent = 2.0 * sum(p[i] * p[i + 1] for i in range(len(p) - 1))
    return same + adjacent

def simulate_mc(p: List[float], n: int, seed: int | None = None) -> float:
    """Монте-Карло моделирование. Разрешаем промах с вероятностью 1 - sum(p)."""
    p, p_miss = validate_probs(p)
    rng = random.Random(seed)
    # Построим кумулятив для выборки индекса исхода: 0..k-1 — баки, k — 'промах'
    weights = p + ([p_miss] if p_miss > 0 else [])
    cdf = []
    acc = 0.0
    for w in weights:
        acc += w
        cdf.append(acc)

    def sample_outcome() -> int:
        """Возвращает индекс попадания: 0..k-1 — номер бака, k — промах (если есть)."""
        u = rng.random()
        # двоичный поиск тут не обязателен; линейного достаточно.
        for idx, cut in enumerate(cdf):
            if u <= cut:
                return idx
        return len(cdf) - 1  # на случай округл. хвоста

    k = len(p)
    hits = 0
    miss_index = len(p) if p_miss > 0 else None

    for _ in range(n):
        a = sample_outcome()
        b = sample_outcome()
        # Условия воспламенения:
        # 1) оба выстрела в один и тот же бак (не промах)
        if a == b and (miss_index is None or a != miss_index):
            hits += 1
            continue
        # 2) по соседним бакам (i, i+1) в любом порядке
        if (a < k and b < k) and (abs(a - b) == 1):
            hits += 1

    return hits / n

def ci_wald(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Классический ДИ Вальда: p̂ ± z * sqrt(p̂(1-p̂)/n)"""
    se = math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / n)
    lo = max(0.0, p_hat - z * se)
    hi = min(1.0, p_hat + z * se)
    return lo, hi

def ci_wilson(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """ДИ Уилсона (лучше при крайних p̂ и умеренных n)."""
    if n <= 0:
        return (0.0, 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n)) / n)
    return max(0.0, center - margin), min(1.0, center + margin)


def main():
    print("=== Моделирование воспламенения баков (два выстрела, баки в линию) ===")
    try:
        use_defaults = input("Использовать параметры по умолчанию? [Y/n]: ").strip().lower()
    except EOFError:
        use_defaults = "y"

    if use_defaults in ("", "y", "yes", "д", "да"):
        k = K_DEFAULT
        p = P_DEFAULT[:]
        n = N_DEFAULT
        seed = SEED_DEFAULT
    else:
        k = int(input("Введите k (k > 2): ").strip())
        print(f"Введите {k} вероятностей p1..pk через пробел (сумма ≤ 1, остаток = промах):")
        p_str = input().strip().split()
        if len(p_str) != k:
            raise ValueError(f"Ожидалось {k} чисел, получено {len(p_str)}.")
        p = [float(x) for x in p_str]
        n = int(input("Число прогонов Монте-Карло N (например, 500000): ").strip())
        seed_in = input("Seed (пусто — случайный): ").strip()
        seed = int(seed_in) if seed_in else None

    # Проверим корректность и вычислим аналитическое значение
    p, p_miss = validate_probs(p)
    p_true = analytic_probability(p)

    print("\n--- Параметры ---")
    print(f"k = {len(p)} баков")
    print("p =", ", ".join(f"{x:.6f}" for x in p), end="")
    if p_miss > 0:
        print(f"  |  промах = {p_miss:.6f}")
    else:
        print("  |  промаха нет (сумма p_i = 1)")
    print(f"N = {n:,}")
    print(f"seed = {seed}")

    # Монте-Карло
    p_hat = simulate_mc(p, n, seed=seed)

    # Доверительные интервалы (95%)
    z = 1.96
    lo_wald, hi_wald = ci_wald(p_hat, n, z)
    lo_wil, hi_wil = ci_wilson(p_hat, n, z)

    # Вывод результатов
    print("\n--- Результаты ---")
    print(f"Аналитическая вероятность P_true = {p_true:.10f}")
    print(f"Монте-Карло оценка     p_hat   = {p_hat:.10f}")
    se = math.sqrt(p_hat * (1 - p_hat) / n)
    print(f"Стандартная ошибка SE ≈ {se:.10f}")

    print("\n95% ДИ Вальда:  [{:.10f}, {:.10f}]  => содержит истину? {}".format(
        lo_wald, hi_wald, "ДА" if (lo_wald <= p_true <= hi_wald) else "НЕТ"
    ))
    print("95% ДИ Уилсона: [{:.10f}, {:.10f}]  => содержит истину? {}".format(
        lo_wil, hi_wil, "ДА" if (lo_wil <= p_true <= hi_wil) else "НЕТ"
    ))

    # Краткая справка по аналитике:
    print("\n--- Справка по аналитическому решению ---")
    same = sum(pi * pi for pi in p)
    adjacent = 2.0 * sum(p[i] * p[i + 1] for i in range(len(p) - 1))
    print(f"Слагаемое 'один и тот же бак' : sum p_i^2 = {same:.10f}")
    print(f"Слагаемое 'соседние баки'     : 2*sum p_i*p_(i+1) = {adjacent:.10f}")
    print(f"Итого P = sum p_i^2 + 2*sum p_i p_(i+1) = {p_true:.10f}")

if __name__ == "__main__":
    main()

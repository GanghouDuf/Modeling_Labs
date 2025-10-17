
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

# --- Универсальный генератор (формула 1.12) ---
def universal_generator(n, 
                        a=[1176, 1476, 1776], 
                        b=[8191, 131071, 8388607], 
                        c=[123, 321, 231], 
                        y0=[100, 200, 300]):
    k = len(a)
    y = np.zeros((k, n+1), dtype=int)
    for i in range(k):
        y[i, 0] = y0[i]
    for j in range(n):
        for i in range(k):
            y[i, j+1] = abs(a[i] * (y[i, j] % b[i]) - (c[i] * y[i, j]) // b[i])
    u = ((y[0,1:]/b[0]) + (y[1,1:]/b[1]) + (y[2,1:]/b[2])) % 1
    return u

# --- Теоретическая функция распределения ---
def F_theor(x):
    if 0 <= x < 0.5:
        return np.sqrt(0.25 - (x - 0.5)**2)
    elif 0.5 <= x < 1.3:
        return 0.3125*x + 0.34375
    elif 1.3 <= x <= 1.5:
        return 1.25*x - 0.875
    elif x < 0:
        return 0
    else:
        return 1

F_theor_vec = np.vectorize(F_theor)

# --- Теоретическая плотность распределения ---
def f_theor(x):
    if 0 <= x < 0.5:
        denom = np.sqrt(0.25 - (x - 0.5)**2)
        if denom == 0:
            return 0
        return (0.5 - x) / denom
    elif 0.5 <= x < 1.3:
        return 0.3125
    elif 1.3 <= x <= 1.5:
        return 1.25
    else:
        return 0

f_theor_vec = np.vectorize(f_theor)

# --- Обратная функция распределения (метод обратных функций) ---
def F_inv(u):
    x = np.zeros_like(u)

    # [0;0.5] → F ∈ [0;0.5]
    mask1 = (u >= 0) & (u < 0.5)
    x[mask1] = 0.5 - np.sqrt(0.25 - u[mask1]**2)

    # [0.5;1.3] → F ∈ [0.5;0.75]
    mask2 = (u >= 0.5) & (u < 0.75)
    x[mask2] = (u[mask2] - 0.34375) / 0.3125

    # [1.3;1.5] → F ∈ [0.75;1.0]
    mask3 = (u >= 0.75) & (u <= 1.0)
    x[mask3] = (u[mask3] + 0.875) / 1.25

    return x

# --- Генерация выборки ---
N = 2000
u_sample = universal_generator(N)   # равномерная выборка из твоей программы
sample = F_inv(u_sample)            # преобразование в X по варианту 2

# --- Выборочные характеристики ---
mean = np.mean(sample)
var = np.var(sample)

print("Оценка математического ожидания:", mean)
print("Оценка дисперсии:", var)

# --- Гистограмма частот с наложением теоретической плотности ---
plt.hist(sample, bins=25, range=(0, 1.5), density=True, 
         edgecolor='black', alpha=0.6, label="Эмпирическая гистограмма")

x_vals = np.linspace(0, 1.5, 400)
plt.plot(x_vals, f_theor_vec(x_vals), 'r-', lw=2, label="Теоретическая плотность")

plt.title("Гистограмма распределения с теоретической плотностью")
plt.xlabel("x")
plt.ylabel("Плотность")
plt.legend()
plt.show()

# --- Эмпирическая и теоретическая CDF ---
sorted_sample = np.sort(sample)
empirical_cdf = np.arange(1, N+1) / N

plt.step(sorted_sample, empirical_cdf, where='post', label="Эмпирическая CDF")
plt.plot(x_vals, F_theor_vec(x_vals), 'r-', lw=2, label="Теоретическая CDF")

plt.title("Функция распределения")
plt.xlabel("x")
plt.ylabel("F(x)")
plt.legend()
plt.show()

# --- Критерий Колмогорова–Смирнова ---
D, p_value = kstest(sample, F_theor_vec)

print("Статистика Колмогорова–Смирнова D =", D)
print("p-value =", p_value)

alpha = 0.05
if p_value > alpha:
    print("✅ Гипотеза H0 НЕ отвергается: выборка соответствует теоретическому распределению")
else:
    print("❌ Гипотеза H0 отвергается: выборка не соответствует теоретическому распределению")


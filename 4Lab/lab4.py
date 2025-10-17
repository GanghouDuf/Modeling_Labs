import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, norm

# ------------------------------
# ПАРАМЕТРЫ РАСПРЕДЕЛЕНИЯ
# ------------------------------
mu = 0       # математическое ожидание
sigma = 1    # стандартное отклонение
n = 1000     # объем выборки
k = 15       # число интервалов для гистограммы

# ------------------------------
# 1. Метод ЦПТ
# ------------------------------
def normal_clt(mu, sigma, n, m=12):
    samples = np.sum(np.random.uniform(0, 1, (n, m)), axis=1) - m/2
    return mu + sigma * samples
def normal_box_muller(mu, sigma, n):
    u1 = np.random.uniform(0, 1, n)
    u2 = np.random.uniform(0, 1, n)
    z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return mu + sigma * z


data_clt = normal_clt(mu, sigma, n)
data_bm  = normal_box_muller(mu, sigma, n)


def print_stats(data, name):
    print(f"\n===== {name} =====")
    print(f"Мат. ожидание = {np.mean(data):.4f}")
    print(f"Дисперсия     = {np.var(data):.4f}")
    ks = kstest(data, 'norm', args=(mu, sigma))
    print("Критерий Колмогорова:", ks)

# Вывод статистик
print_stats(data_clt, "Метод ЦПТ")
print_stats(data_bm, "Метод Бокса–Маллера")


plt.figure(figsize=(10,5))
plt.hist(data_bm, bins=k, density=True, alpha=0.6, label="Выборка (Box-Muller)")
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
plt.plot(x, norm.pdf(x, mu, sigma), label="Теоретическая плотность N(0,1)")
plt.title("Гистограмма и теоретическая нормальная плотность")
plt.xlabel("x")
plt.ylabel("Плотность")
plt.legend()
plt.grid(True)
plt.show()


data_sorted = np.sort(data_bm)
ecdf = np.arange(1, n + 1) / n
cdf_theor = norm.cdf(data_sorted, mu, sigma)

plt.figure(figsize=(10,5))
plt.step(data_sorted, ecdf, where='post', label='Эмпирическая F_n(x)')
plt.plot(data_sorted, cdf_theor, label='Теоретическая F(x) N(0,1)')
plt.title("Статистическая функция распределения (ЭФР)")
plt.xlabel("x")
plt.ylabel("F(x)")
plt.legend()
plt.grid(True)
plt.show()


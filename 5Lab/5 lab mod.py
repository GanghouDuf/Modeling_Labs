import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, lognorm, kstest


N = 1000        # объем выборки
k_bins = 15

#Параметры гамма-распределения
shape_gamma = 2.0   # k (shape)
scale_gamma = 2.0   # θ (scale)

#Параметры логнормального распределения
sigma_logn = 0.5    # стандартное отклонение log(X)
mu_logn = 0


gamma_sample = gamma.rvs(a=shape_gamma, scale=scale_gamma, size=N)

print("=== Гамма-распределение ===")
print("Эмпирическое матожидание:", np.mean(gamma_sample))
print("Эмпирическая дисперсия:", np.var(gamma_sample))

#гистограмма
plt.hist(gamma_sample, bins=k_bins, density=True, alpha=0.6)
x = np.linspace(min(gamma_sample), max(gamma_sample), 200)
plt.plot(x, gamma.pdf(x, a=shape_gamma, scale=scale_gamma))
plt.title("Гистограмма (Gamma)")
plt.show()

#CDF
gamma_sample_sorted = np.sort(gamma_sample)
ecdf = np.arange(1, N + 1) / N
plt.step(gamma_sample_sorted, ecdf, where="post", label="Эмпирическая CDF")
plt.plot(x, gamma.cdf(x, a=shape_gamma, scale=scale_gamma), label="Теоретическая CDF")
plt.title("Функция распределения (Gamma)")
plt.legend()
plt.show()

#logCDF
logn_sample = lognorm.rvs(s=sigma_logn, scale=np.exp(mu_logn), size=N)

print("\n=== Логнормальное распределение ===")
print("Эмпирическое матожидание:", np.mean(logn_sample))
print("Эмпирическая дисперсия:", np.var(logn_sample))

#гистограмма
plt.hist(logn_sample, bins=k_bins, density=True, alpha=0.6)
x = np.linspace(min(logn_sample), max(logn_sample), 300)
plt.plot(x, lognorm.pdf(x, s=sigma_logn, scale=np.exp(mu_logn)))
plt.title("Гистограмма (Lognormal)")
plt.show()


logn_sorted = np.sort(logn_sample)
ecdf2 = np.arange(1, N + 1) / N
plt.step(logn_sorted, ecdf2, where="post", label="Эмпирическая CDF")
plt.plot(x, lognorm.cdf(x, s=sigma_logn, scale=np.exp(mu_logn)), label="Теоретическая CDF")
plt.title("Функция распределения (Lognormal)")
plt.legend()
plt.show()

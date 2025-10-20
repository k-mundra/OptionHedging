import numpy as np
import matplotlib.pyplot as plt

def heston_sim(S0, v0, r, kappa, theta, sigma, rho, T, N, M, h):
    dt = T / N
    S = np.zeros((M, N + 1))
    v = np.zeros((M, N + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    S_plus = np.zeros((M, N + 1))
    S_plus[:, 0] = S0 + h
    S_minus = np.zeros((M, N + 1))
    S_minus[:, 0] = S0 - h

    for t in range(1, N + 1):
        Z1 = np.random.normal(size=M)
        Z2 = np.random.normal(size=M)
        W1 = Z1 * np.sqrt(dt)
        W2 = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)

        v[:, t] = np.abs(v[:, t - 1] + kappa * (theta - v[:, t - 1]) * dt + sigma * np.sqrt(v[:, t - 1]) * W2)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * v[:, t - 1]) * dt + np.sqrt(v[:, t - 1]) * W1)
        S_plus[:, t] = S_plus[:, t - 1] * np.exp((r - 0.5 * v[:, t - 1]) * dt + np.sqrt(v[:, t - 1]) * W1)
        S_minus[:, t] = S_minus[:, t - 1] * np.exp((r - 0.5 * v[:, t - 1]) * dt + np.sqrt(v[:, t - 1]) * W1)
    return S.mean(axis=0), S_plus.mean(axis=0), S_minus.mean(axis=0), v.mean(axis=0)

def option_price_heston(S0, v0, r, kappa, theta, sigma, rho, T, N, M, K):
    t = np.linspace(0, T, N + 1)
    S, S_plus, S_minus, _ = heston_sim(S0, v0, r, kappa, theta, sigma, rho, T, N, M, h=1e-4)
    payoff = np.maximum(S - K, 0)
    payoff_plus = np.maximum(S_plus- K, 0)
    payoff_minus = np.maximum(S_minus - K, 0)
    discounted_payoff = np.exp(-r * t) * payoff
    discounted_payoff_plus = np.exp(-r * t) * payoff_plus
    discounted_payoff_minus = np.exp(-r * t) * payoff_minus
    # option_price = np.mean(discounted_payoff)
    return discounted_payoff, discounted_payoff_plus, discounted_payoff_minus

def heston_greeks(S0, v0, r, kappa, theta, sigma, rho, T, N, M, h):
    price, price_plus, price_minus = option_price_heston(S0, v0, r, kappa, theta, sigma, rho, T, N, M, K=S0)
    delta = (price_plus - price_minus) / (2 * h)
    gamma = (price_plus - 2 * price + price_minus) / (h ** 2)
    return delta, gamma



# Example usage
S0 = 100
v0 = 0.04
r = 0.05
kappa = 2.0
theta = 0.04
sigma = 0.3
rho = -0.7
T = 1.0
N = 365
M = 10000
h = 1e-3
plt.figure(figsize=(12, 8))
s, s_minus, s_plus, v = heston_sim(S0, v0, r, kappa, theta, sigma, rho, T, N, M, h)
delta, gamma = heston_greeks(S0, v0, r, kappa, theta, sigma, rho, T, N, M, h)
print(f"Simulated Delta: {delta}")
print(f"Simulated Gamma: {gamma.shape}")
print(f"Simulated Asset Price: {s.shape}")
print(f"Simulated Variance: {v.shape}")

plt.plot(s, label='Asset Price')
# plt.plot(v, label='Variance')
plt.plot(s_plus, label='Asset Price + h')
plt.plot(s_minus, label='Asset Price - h')
plt.title('Heston Model Simulation')
plt.xlabel('Time Steps')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def heston_sim(S0, v0, r, kappa, theta, sigma, rho, T, N, M, h):
    dt = T / N # timestep
    S = np.zeros((M, N + 1)) # container for underlying price
    S_plus = S.copy()
    S_minus = S.copy()
    S[:, 0] = S0
    S_plus[:, 0] = S0 + h
    S_minus[:, 0] = S0 - h

    v = np.zeros((M, N + 1)) # container for stochastic volatility
    v[:, 0] = v0
    
    np.random.seed(0)
    for t in range(1, N + 1):
        Z1 = np.random.normal(size=M)
        Z2 = np.random.normal(size=M)
        W1 = Z1 * np.sqrt(dt)
        W2 = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)

        v[:, t] = np.abs(v[:, t - 1] + kappa * (theta - v[:, t - 1]) * dt + sigma * np.sqrt(v[:, t - 1]) * W2)

        # compute all 3 together so we can use CRN (common random numbers) and only find v once
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * v[:, t - 1]) * dt + np.sqrt(v[:, t - 1]) * W1)
        S_plus[:, t] = S_plus[:, t - 1] * np.exp((r - 0.5 * v[:, t - 1]) * dt + np.sqrt(v[:, t - 1]) * W1)
        S_minus[:, t] = S_minus[:, t - 1] * np.exp((r - 0.5 * v[:, t - 1]) * dt + np.sqrt(v[:, t - 1]) * W1)

    return S, S_plus, S_minus, v

def option_price_heston(S, T, K):
    payoffs = np.maximum(S[:,-1] - K, 0) # compare underyling price at option maturity (last column) to strike price 
    discounted_payoffs = np.exp(-r * T) * payoffs # find present value
    option_price = np.mean(discounted_payoffs) # average over all paths
    return option_price

def heston_greeks(price, price_plus, price_minus, h):
    delta = (price_plus - price_minus) / (2 * h) # first derivative
    gamma = (price_plus - 2 * price + price_minus) / (h ** 2) # second derivative
    return delta, gamma

# Example usage
S0 = 100
v0 = 0.04
r = 0.05 # risk neutral rate
kappa = 2.0
theta = 0.04
sigma = 0.3
rho = -0.7
T = 1.0 # maturity in years
K = 110 # strike price
N = 365 # steps
M = 10000 # paths
h = 1e-3 

s, s_plus, s_minus, v = heston_sim(S0, v0, r, kappa, theta, sigma, rho, T, N, M, h)
price = option_price_heston(s, T, K)
price_plus = option_price_heston(s_plus, T, K)
price_minus = option_price_heston(s_minus, T, K)
delta, gamma = heston_greeks(price, price_plus, price_minus, h)

print(f"Simulated Delta: {delta}")
print(f"Simulated Gamma: {gamma}")
print(f"Simulated Asset Price: {np.mean(s[:,-1])}")
print(f"Simulated Variance: {np.mean(v[:,-1])}")

time = np.linspace(0,T,N+1)
fig, axes = plt.subplots(1,2, figsize=(15,5))

i=0
while i<M:
    axes[0].plot(time, s[i,:], linewidth=1.5,alpha=0.3, color='red')
    axes[1].plot(time, v[i,:], linewidth=1.5,alpha=0.3, color='red')
    i+=100

axes[0].plot(time, np.mean(s, axis=0), linewidth=2, color='black')
axes[1].plot(time, np.mean(v, axis=0), linewidth=2, color='black')

axes[0].set_title('Underlying Price')
axes[1].set_title('Volatility')

plt.show()

import numpy as np

def sim_heston_paths(S0, v0, r, kappa, theta, sigma, rho, dt, num_steps, num_paths, Z_future):
    S = np.zeros((num_paths, num_steps + 1)) # container for underlying price
    S[:, 0] = S0

    v = np.zeros((num_paths, num_steps + 1)) # container for stochastic volatility
    v[:, 0] = v0

    for t in range(1, num_steps + 1):
        Z1 = Z_future[:,0,t-1]
        Z2 = Z_future[:,1,t-1]
        W1 = Z1 * np.sqrt(dt)
        W2 = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)

        v[:, t] = np.abs(v[:, t - 1] + kappa * (theta - v[:, t - 1]) * dt + sigma * np.sqrt(v[:, t - 1]) * W2)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * v[:, t - 1]) * dt + np.sqrt(v[:, t - 1]) * W1)

    return S, v

def heston_option_price(S, K, discount_period):
    payoffs = np.maximum(S[:,-1] - K, 0) # compare underyling price at option maturity (last column) to strike price 
    discounted_payoffs = np.exp(-r * discount_period) * payoffs # find present value
    option_price = np.mean(discounted_payoffs) # average over all paths
    return option_price

def heston_greeks(price, price_plus, price_minus, h):
    delta = (price_plus - price_minus) / (2 * h) # first derivative
    gamma = (price_plus - 2 * price + price_minus) / (h ** 2) # second derivative
    return delta, gamma


'''
----------------------------------------------------------------------
----------------------------------------------------------------------
FUNCTIONAL CODE ENDS HERE, BELOW IS JUST AN ILLUSTRATIVE EXAMPLE
----------------------------------------------------------------------
----------------------------------------------------------------------
'''

import matplotlib.pyplot as plt

# Example usage
S0 = 100 # initial value of stock price
v0 = 0.04 # initial value of variance
r = 0.05 # risk free rate
kappa = 2.0 # rate of mean inversion of variance
theta = 0.04 # long term mean of variance
sigma = 0.3 # volatility of volatility
rho = -0.7 # correlation between Weiner processes
T = 1.0 # maturity in years
K = 110 # strike price
num_steps = 365 # steps
num_paths = 10000 # paths
h = 1e-3 # perturbation to stock price for greek computation

np.random.seed(0)
Z_future = np.random.randn(num_paths, 2, num_steps) # sample all the random values at once before starting the sim
dt = T / num_steps # timestep

s, v= sim_heston_paths(S0, v0, r, kappa, theta, sigma, rho, dt, num_steps, num_paths, Z_future)
s_plus, _= sim_heston_paths(S0+h, v0, r, kappa, theta, sigma, rho, dt, num_steps, num_paths, Z_future)
s_minus, _= sim_heston_paths(S0-h, v0, r, kappa, theta, sigma, rho, dt, num_steps, num_paths, Z_future)

price = heston_option_price(s, K, T)
price_plus = heston_option_price(s_plus, K, T)
price_minus = heston_option_price(s_minus, K, T)
delta, gamma = heston_greeks(price, price_plus, price_minus, h)

print(f"Simulated Delta: {delta}")
print(f"Simulated Gamma: {gamma}")
print(f"Simulated Asset Price: {np.mean(s[:,-1])}") # final value averaged across paths
print(f"Simulated Variance: {np.mean(v[:,-1])}") # final value averaged across paths

time = np.linspace(0,T,num_steps+1)
fig, axes = plt.subplots(1,2, figsize=(15,5))

i=0
while i<num_paths:
    axes[0].plot(time, s[i,:], linewidth=1.5,alpha=0.3, color='red')
    axes[1].plot(time, v[i,:], linewidth=1.5,alpha=0.3, color='red')
    i+=100

axes[0].plot(time, np.mean(s, axis=0), linewidth=2, color='black')
axes[1].plot(time, np.mean(v, axis=0), linewidth=2, color='black')

axes[0].set_title('Underlying Price')
axes[1].set_title('Volatility')

plt.show()

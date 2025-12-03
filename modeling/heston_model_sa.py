import numpy as np
from scipy.integrate import quad

def sim_heston_paths(S0, v0, r, kappa, theta, sigma, rho, dt, num_steps, num_paths, Z_future):
    S = np.zeros((num_paths, num_steps + 1)) # container for underlying price
    S[:, 0] = S0

    v = np.zeros((num_paths, num_steps + 1)) # container for stochastic volatility
    v[:, 0] = np.maximum(v0, 0.0)

    for t in range(1, num_steps + 1):
        Z1 = Z_future[:,0,t-1]
        Z2 = Z_future[:,1,t-1]
        W1 = Z1 * np.sqrt(dt)
        W2 = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)

        v[:, t] = np.maximum((v[:, t - 1] + kappa * (theta - v[:, t - 1]) * dt + sigma * np.sqrt(v[:, t - 1]) * W2), 0.0)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * v[:, t - 1]) * dt + np.sqrt(v[:, t - 1]) * W1)

    return S, v

def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):

    # constants
    a = kappa*theta
    b = kappa+lambd

    # common terms w.r.t phi
    rspi = rho*sigma*phi*1j

    # define d parameter given phi and b
    d = np.sqrt( (rho*sigma*phi*1j - b)**2 + (phi*1j+phi**2)*sigma**2 )

    # define g parameter given phi, b and d
    g = (b-rspi+d)/(b-rspi-d)

    # calculate characteristic function by components
    exp1 = np.exp(r*phi*1j*tau)
    term2 = S0**(phi*1j) * ( (1-g*np.exp(d*tau))/(1-g) )**(-2*a/sigma**2)
    exp2 = np.exp(a*tau*(b-rspi+d)/sigma**2 + v0*(b-rspi+d)*( (1-np.exp(d*tau))/(1-g*np.exp(d*tau)) )/sigma**2)

    return exp1*term2*exp2
def integrand(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    numerator = np.exp(r*tau)*heston_charfunc(phi-1j,*args) - K*heston_charfunc(phi,*args)
    denominator = 1j*phi*K**(1j*phi)
    return numerator/denominator

def heston_option_price(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

    P, umax, N = 0, 100, 10000
    dphi=umax/N #dphi is width

    for i in range(1,N):
        # rectangular integration
        phi = dphi * (2*i + 1)/2 # midpoint to calculate height
        numerator = np.exp(r*tau)*heston_charfunc(phi-1j,*args) - K * heston_charfunc(phi,*args)
        denominator = 1j*phi*K**(1j*phi)

        P += dphi * numerator/denominator

    return np.real((S0 - K*np.exp(-r*tau))/2 + P/np.pi)
def heston_price_scipy(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

    real_integral, _ = np.real( quad(integrand, 0, 100, args=args) )

    return (S0 - K*np.exp(-r*tau))/2 + real_integral/np.pi

def heston_greeks(price_plus, price_minus, h, mode: str, price=0):
    if mode == 'delta':
        delta = (price_plus - price_minus) / (2 * h) # first derivative
        return delta
    elif mode == 'vega':
        vega = (price_plus - price_minus) / (2 * h) # first derivative
        return vega
    elif mode == 'gamma':
        gamma = (price_plus - 2 * price + price_minus) / (h ** 2) # second derivative
        return gamma


'''
----------------------------------------------------------------------
----------------------------------------------------------------------
FUNCTIONAL CODE ENDS HERE, BELOW IS JUST AN ILLUSTRATIVE EXAMPLE
----------------------------------------------------------------------
----------------------------------------------------------------------
'''

if __name__ == "__main__":
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
    lambd = 0
    np.random.seed(0)
    Z_future = np.random.randn(num_paths, 2, num_steps) # sample all the random values at once before starting the sim
    dt = T / num_steps # timestep

    price = heston_option_price(S0, K, v0, kappa, theta, sigma, rho, lambd, T, r)
    price_plus = heston_option_price(S0+h, K, v0, kappa, theta, sigma, rho, lambd, T, r)
    price_minus = heston_option_price(S0-h, K, v0, kappa, theta, sigma, rho, lambd, T, r)
    delta= heston_greeks(price_plus, price_minus, h, mode='delta')
    gamma= heston_greeks(price_plus, price_minus, h, mode='gamma', price=price)

    print(f"Simulated Delta: {delta}")
    print(f"Simulated Gamma: {gamma}")
    # for semi analytical solution

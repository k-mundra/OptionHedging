from modeling.heston_model_mc import sim_heston_paths, heston_greeks, heston_option_price
import numpy as np
import matplotlib.pyplot as plt

def dynamic_hedge(hedge_freq=10):
    S0 = 100 # initial value of stock price
    v0 = 0.04 # initial value of variance
    r = 0.05 # risk free rate
    kappa = 2.0 # rate of mean inversion of variance
    theta = 0.04 # long term mean of variance
    sigma = 0.3 # volatility of volatility
    rho = -0.7 # correlation between Weiner processes
    T = 1.0 # maturity in years
    K = 110 # strike price
    num_steps_outer = 365 # steps
    num_paths_outer = 100 # paths for outer (baseline) simulation
    h = 1e-3 # perturbation to stock price for greek computation

    num_paths_inner = 500

    '''
    input to function: hedge_freq -> this is the hedging frequency in days,
    use this to figure out what num_steps to use as inputs sim_heston_paths and how often we call it
    '''
 
    '''TASK:hedge a portfolio short 1 option and long delta stock, with any cash earning interest at r'''
    '''implement this function by
    1. running MC sims at every time step we want to hedge at (using num_paths_inner)
    2. finding the option prices discounted to that time and corresponding greeks
    3. update portfolio
    
    finally return PnL -> should be (M,1) array where M is num of paths'''
 
    hedge_step_days = max(1, int(hedge_freq)) # guarantees hedging freq is at least 1 day
    dt_outer = T / num_steps_outer 

    steps_per_day = num_steps_outer / (T * 365)
    hedge_step = int(round(hedge_step_days * steps_per_day))
    hedge_indices = np.arange(0, num_steps_outer + 1, hedge_step) # time steps to hedge at 
 
    # generate outer simulation
    np.random.seed(42)
    Z_outer = np.random.randn(num_paths_outer, 2, num_steps_outer)
    S_paths, V_paths = sim_heston_paths(S0, v0, r, kappa, theta, sigma, rho,
                                        dt_outer, num_steps_outer, num_paths_outer, Z_outer)
 

    cash = np.zeros(num_paths_outer)        # initialize cash account for each outer path
    stock_pos = np.zeros(num_paths_outer)   # initialize stock position

    for t_idx in hedge_indices:
        tau = T - t_idx * dt_outer # time remaining until option expires 
 
        if t_idx > 0:
            time_elapsed = hedge_step_days * dt_outer  
            cash *= np.exp(r * time_elapsed) # accrue interest 
 
        S_t = S_paths[:, t_idx] # get current stock price
        V_t = V_paths[:, t_idx] # get current variance 
 
        deltas = np.zeros_like(S_t)
        prices = np.zeros_like(S_t)
 
        num_steps_remaining = num_steps_outer - t_idx # how many time steps until maturity 
 
        for i in range(num_paths_outer):
            # generate inner simulation
            Z_inner = np.random.randn(num_paths_inner, 2, num_steps_remaining)
 
            s_main, _ = sim_heston_paths(S_t[i], V_t[i], r, kappa, theta, sigma, rho,
                                         dt_outer, num_steps_remaining, num_paths_inner, Z_inner)
            s_plus, _ = sim_heston_paths(S_t[i] + h, V_t[i], r, kappa, theta, sigma, rho,
                                         dt_outer, num_steps_remaining, num_paths_inner, Z_inner)
            s_minus, _ = sim_heston_paths(S_t[i] - h, V_t[i], r, kappa, theta, sigma, rho,
                                          dt_outer, num_steps_remaining, num_paths_inner, Z_inner)
 
            price = heston_option_price(s_main, K, r, tau)
            price_plus = heston_option_price(s_plus, K, r, tau)
            price_minus = heston_option_price(s_minus, K, r, tau)
 
            delta = heston_greeks(price_plus, price_minus, h, mode='delta') # compute delta
 
            deltas[i] = delta
            prices[i] = price
 
        if t_idx == 0:
            stock_pos = deltas
            cash = prices.reshape(-1) - deltas * S_t
        else:
            delta_diff = deltas - stock_pos
            cash -= delta_diff * S_t
            stock_pos = deltas 
 
    S_T = S_paths[:, -1] # get final stock price 
    payoff = np.maximum(S_T - K, 0) # option payoff at maturity 
    
    cash *= np.exp(r * hedge_step_days * dt_outer) # accrue interest for final period
    
    pnl = stock_pos * S_T + cash - payoff # calculate final pnl
 
    print("Mean PnL:", np.mean(pnl))
    print("Std PnL:", np.std(pnl))
 
    plt.hist(pnl, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(np.mean(pnl), color='red', linestyle='--', linewidth=2,
                label=f'Mean = ${np.mean(pnl):.2f}')
    plt.axvline(0, color='black', linestyle='-', linewidth=1)
    plt.xlabel('P&L (dollars $)')
    plt.ylabel('Freq')
    plt.title(f'PnL Distribution, Delta Hedging')
    plt.legend()
    plt.grid(True)
    plt.show()
 
    return pnl
 
if __name__ == "__main__":
    dynamic_hedge(hedge_freq=10)
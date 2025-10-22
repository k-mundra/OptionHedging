from heston_model import sim_heston_paths, heston_greeks, heston_option_price
import numpy as np
import matplotlib.pyplot as plt

def dynamic_hedge(hedge_freq = 10):
    pass

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
    num_paths_outer = 10000 # paths for outer (baseline) simulation
    h = 1e-3 # perturbation to stock price for greek computation

    num_paths_inner = 


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

# FEEL FREE TO ADD ANY HELPER FUNCTIONS OR PLOTTING CODE AS NECESSARY